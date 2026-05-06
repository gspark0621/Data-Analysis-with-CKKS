import os
import numpy as np
from time import time
from sklearn.metrics import adjusted_rand_score

from core.plaintext.MultipartyOwner_plain import prepare_owner_blocks_plain
from core.plaintext.GridIndex_plain import (
    generate_public_grid_centers_nd,
    normalize_grid_centers,
    compute_axis_cell_counts,
)
from core.plaintext.MultipartyServer_plain import run_multiparty_point_dbscan_plain
from core.plaintext.FinalClient_plain import reconstruct_results_plain

DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/hepta.arff"

def load_arff_to_pts_with_labels(filepath: str):
    pts = []
    true_labels = []
    data_section = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@data'):
                data_section = True
                continue
            if data_section:
                line = line.replace('\t', ' ').replace(',', ' ')
                values = line.split()
                if len(values) < 2:
                    continue

                row = [float(v) for v in values[:-1]]
                label = int(float(values[-1]))
                pts.append(row)
                true_labels.append(label)

    if not pts:
        raise ValueError("데이터를 찾을 수 없습니다. 파일 포맷을 확인해주세요.")

    return np.array(pts, dtype=np.float64), np.array(true_labels, dtype=int)


def save_timings_txt(filename, timings_dict):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for k, v in timings_dict.items():
                f.write(f"{k}: {float(v):.6f}\n")
        print(f"✅ 타이밍 파일 저장 완료: {filename}")
    except Exception as e:
        print(f"❌ 타이밍 파일 저장 실패: {e}")


def remap_labels_to_sequential(labels):
    unique_valid = sorted([x for x in set(labels) if x != -1])
    mapping = {old: idx + 1 for idx, old in enumerate(unique_valid)}

    out = []
    for x in labels:
        if x == -1:
            out.append(-1)
        else:
            out.append(mapping[x])
    return out


def main():
    print("==================================================")
    print("      대화형 Plaintext B-Model DBSCAN 전체 데이터 테스트      ")
    print("==================================================\n")

    print(f"▶ 데이터셋 경로: {DATASET_PATH}\n")

    eps_val = float(input("eps 값을 입력하세요 (예: 0.5) > "))
    min_pts_val = int(float(input("min_pts 값을 입력하세요 (예: 3) > ")))

    print("\n--------------------------------------------------")
    print(f"▶ 실행 파라미터: eps = {eps_val}, min_pts = {min_pts_val}")
    print("--------------------------------------------------\n")

    print("데이터 전체 로딩 중...")
    pts, true_labels = load_arff_to_pts_with_labels(DATASET_PATH)
    N = len(pts)
    dimension = pts.shape[1]

    global_min = np.min(pts)
    global_max = np.max(pts)
    scale_factor = global_max - global_min if (global_max - global_min) != 0.0 else 1.0

    query_epsilon_norm = eps_val / scale_factor
    domain_mins_norm = [0.0] * dimension
    domain_maxs_norm = [1.0] * dimension
    base_epsilon_norm = min((min_pts_val / N) ** (1 / dimension), 1.0)

    print(f"완료! (총 {N}개의 점, {dimension}차원)")
    print(f"global_min = {global_min:.6f}, global_max = {global_max:.6f}")
    print(f"Base epsilon(norm, grid용) = {base_epsilon_norm:.6f}")
    print(f"Query epsilon(norm, DBSCAN용) = {query_epsilon_norm:.6f}\n")

    if N > 10000:
        print(f"⚠️ 경고: 데이터 개수({N}개)가 많아 block 비교 비용이 커질 수 있습니다.\n")

    # ---------------------------------------------------------
    # 1. Final Client: 고정 0~1 도메인 기반 public grid 설정
    # ---------------------------------------------------------
    print("================ Public Grid 준비 =======================")

    grid_centers_norm = generate_public_grid_centers_nd(
        domain_mins=domain_mins_norm,
        domain_maxs=domain_maxs_norm,
        epsilon=base_epsilon_norm
    )

    axis_cell_counts = compute_axis_cell_counts(
        domain_mins_norm=domain_mins_norm,
        domain_maxs_norm=domain_maxs_norm,
        epsilon_norm=base_epsilon_norm
    )

    num_grids = 1
    for c in axis_cell_counts:
        num_grids *= c

    bucket_size = 32
    max_blocks_per_grid = max(1, int(np.ceil(N / bucket_size)))

    print(f"축별 cell 수: {axis_cell_counts}")
    print(f"생성된 public grid 수: {len(grid_centers_norm)}")
    print(f"계산된 전체 grid 수: {num_grids}")
    print(f"bucket_size = {bucket_size}, max_blocks_per_grid = {max_blocks_per_grid}\n")

    if len(grid_centers_norm) != num_grids:
        raise ValueError("grid center 개수와 축별 cell 수의 곱이 일치하지 않습니다.")

    if num_grids > 200000:
        print("⚠️ 경고: n차원 grid 수가 매우 큽니다. 메모리/시간이 급격히 증가할 수 있습니다.\n")

    # ---------------------------------------------------------
    # 2. Data Owner: 정규화 후 block 생성 (raw 좌표는 서버로 보내지 않음)
    # ---------------------------------------------------------
    print("================ Plaintext B-Model 시뮬레이션 ===================")
    pt_start_time = time()

    owner_points_raw = pts.tolist()

    owner_blocks, owner_points_norm = prepare_owner_blocks_plain(
        owner_points_raw=owner_points_raw,
        domain_mins_norm=domain_mins_norm,
        domain_maxs_norm=domain_maxs_norm,
        epsilon_norm=base_epsilon_norm,
        axis_cell_counts=axis_cell_counts,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid,
        global_min=global_min,
        global_max=global_max,
        owner_id=0
    )

    server_result = run_multiparty_point_dbscan_plain(
        owner_blocks_list=[owner_blocks],
        grid_centers_norm=grid_centers_norm,
        query_epsilon_norm=query_epsilon_norm,
        base_epsilon_norm=base_epsilon_norm,
        min_pts=min_pts_val,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid
    )

    results = reconstruct_results_plain(
        server_result=server_result,
        global_min=global_min,
        scale_factor=scale_factor
    )

    # 수정점 1: 공간(global_idx) 기준 정렬을 원본 입력(owner_id, owner_local_idx) 기준으로 변경
    ordered_results = sorted(results, key=lambda x: (x["owner_id"], x["owner_local_idx"]))

    cluster_labels_pt = []
    for row in ordered_results:
        cluster_labels_pt.append(int(row["label"]))

    cluster_labels_pt = remap_labels_to_sequential(cluster_labels_pt)

    pt_elapsed = time() - pt_start_time
    print(f"▶ Plaintext B-model 소요 시간: {pt_elapsed:.2f}초\n")

    # ---------------------------------------------------------
    # 3. 디버그 정보 수집
    # ---------------------------------------------------------
    print("================ 중간값 디버그 수집 ====================")

    neighbor_counts = server_result["neighbor_counts"]
    core_mask = server_result["core_mask"]
    adjacency_list = server_result["adjacency_list"]

    # 수정점 2: 디버그 변수들을 원래 순서(ordered_results)에 맞춰서 서버의 배열에서 매핑 추출
    debug_pt_total_neighbors = []
    debug_pt_core_mask = []
    debug_pt_adjacency_size = []

    for row in ordered_results:
        g_idx = row["global_idx"]
        debug_pt_total_neighbors.append(float(neighbor_counts[g_idx]))
        debug_pt_core_mask.append(float(core_mask[g_idx]))
        debug_pt_adjacency_size.append(len(adjacency_list[g_idx]))

    debug_pt = {
        "total_neighbors": debug_pt_total_neighbors,
        "core_mask": debug_pt_core_mask,
        "final_labels": [float(x) for x in cluster_labels_pt],
        "adjacency_size": debug_pt_adjacency_size,
    }

    timings = {
        "plaintext_bmodel_sec": pt_elapsed,
    }

    timing_filename = f"debug_plaintext_timings_eps{eps_val}_min{min_pts_val}.txt"
    save_timings_txt(timing_filename, timings)

    # ---------------------------------------------------------
    # 4. 결과 출력
    # ---------------------------------------------------------
    print("================ 결과 샘플 ==============================")
    for row in ordered_results[:min(20, len(ordered_results))]:
        print(
            f"gid={row['global_idx']:4d} | "
            f"owner={row['owner_id']} | "
            f"local_idx={row['owner_local_idx']} | "
            f"pt_recon={row['point_reconstructed']} | "
            f"pt_norm={row['point_norm']} | "
            f"grid={row['grid_idx']} | "
            f"neighbors={row['neighbor_count']} | "
            f"core={row['is_core']} | "
            f"label={row['label']}"
        )
    print()

    # ---------------------------------------------------------
    # 5. 정답 라벨과 구조 비교
    # ---------------------------------------------------------
    print("================ 검증 결과 ==============================")

    true_labels_seq = remap_labels_to_sequential(true_labels.tolist())

    pt_unique_clusters = sorted(list(set(cluster_labels_pt)))
    pt_valid_clusters = [c for c in pt_unique_clusters if c != -1]

    print("[Plaintext B-model 결과]")
    print(f"  - 최종 발견된 클러스터 목록 (고유 ID): {pt_unique_clusters}")
    print(f"  - 총 {len(pt_valid_clusters)}개의 유효 군집과 노이즈(-1)가 도출되었습니다.\n")

    ari_score = adjusted_rand_score(true_labels_seq, cluster_labels_pt)
    print("📊 [정답 레이블 vs Plaintext B-model 구조 일치도 평가 (ARI)] 📊")
    print(f"  => 데이터셋 정답 구조 vs Plaintext B-model 구조 일치율: 【 {ari_score * 100:.2f} 점 / 100점 】")

    if ari_score > 0.99:
        print("  => 결론: 🎉 대성공! 데이터셋 정답 구조와 사실상 동일합니다.")
    elif ari_score > 0.80:
        print("  => 결론: 👍 대부분 동일하게 묶였으나, 경계선에서 소수의 차이가 있습니다.")
    else:
        print("  => 결론: ❌ eps/min_pts 또는 grid/base epsilon 설정 재점검이 필요합니다.")

    # ---------------------------------------------------------
    # 6. 디버그 CSV 파일 저장
    # ---------------------------------------------------------
    debug_filename = f"debug_plaintext_bmodel_eps{eps_val}_min{min_pts_val}.csv"
    print("\n================ 디버그 CSV 저장 ========================")
    print(f"중간 단계 값을 '{debug_filename}' 파일에 저장 중...")

    try:
        with open(debug_filename, 'w', encoding='utf-8') as f:
            f.write("Point_ID,Adjacency_Size,Neighbor_Count,Core_Mask,Final_Label,True_Label\n")
            for i in range(N):
                f.write(
                    f"{i},"
                    f"{debug_pt['adjacency_size'][i]},"
                    f"{debug_pt['total_neighbors'][i]:.4f},"
                    f"{debug_pt['core_mask'][i]:.4f},"
                    f"{debug_pt['final_labels'][i]:.4f},"
                    f"{true_labels_seq[i]}\n"
                )

        print("✅ 디버깅 비교 파일 저장 완료!\n")
    except Exception as e:
        print(f"❌ 디버깅 파일 저장 실패: {e}\n")

    # ---------------------------------------------------------
    # 7. CSV 파일 저장
    # ---------------------------------------------------------
    output_filename = f"hepta_plaintext_bmodel_result_eps{eps_val}_min{min_pts_val}.csv"
    print("================ 파일 저장 ==============================")
    print(f"데이터셋의 모든 좌표와 클러스터 정보를 '{output_filename}' 에 병합 저장 중...")

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            axis_headers = [f"x{i+1}" for i in range(dimension)]
            header_str = ",".join(axis_headers) + ",PT_BModel_Cluster,True_Class\n"
            f.write(header_str)

            for i in range(N):
                coords = ",".join([f"{val:.4f}" for val in ordered_results[i]["point_reconstructed"]])
                # 수정점 3: 헤더의 선언 순서에 맞춰 예측 라벨(cluster_labels_pt)을 먼저, 정답 라벨(true_labels)을 뒤에 배치
                f.write(f"{coords},{cluster_labels_pt[i]},{true_labels[i]}\n")

        print("✅ 파일 저장 완료! (복원 좌표 기준 저장)")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")


if __name__ == '__main__':
    main()
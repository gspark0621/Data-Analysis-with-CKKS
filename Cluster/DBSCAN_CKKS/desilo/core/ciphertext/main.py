# main.py

import time
import csv
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack

# ── 모듈 임포트 ────────────────────────────────────────────────────
from core.ciphertext.client.FinalClient import (
    assign_global_indices,
    build_owner_coord_map,   # [추가]
    reconstruct_results,
)
# [수정] GridIndex_plain → core.server.GridIndex 로 통합
from core.ciphertext.client.GridIndex import (
    generate_public_grid_centers_nd,
    compute_axis_cell_counts,
)
from core.ciphertext.client.MultipartyDataOwner import prepare_and_encrypt_owner_blocks
from core.ciphertext.server.MultipartyServer import run_multiparty_point_dbscan


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
        raise ValueError("데이터를 찾을 수 없습니다.")
    return np.array(pts, dtype=np.float64), np.array(true_labels, dtype=int)


def main():
    print("==================================================")
    print("      FHE C-Model DBSCAN (전체 데이터 검증 모드)   ")
    print("==================================================\n")

    # ================================================================
    # [사용자 입력] DBSCAN 파라미터 설정
    # ================================================================
    query_epsilon = float(input("▶ eps (query_epsilon) 값을 입력하세요 (예: 0.5) > "))
    min_pts = int(float(input("▶ min_pts 값을 입력하세요 (예: 3) > ")))

    # ================================================================
    # [Step 1 & 2] 데이터 로드 및 메타데이터 추출
    # ================================================================
    t0 = time.time()
    print(f"\n▶ [FC] 데이터셋 로딩 중: {DATASET_PATH}")
    pts, _ = load_arff_to_pts_with_labels(DATASET_PATH)

    owner_raw_pts = pts.tolist()
    total_N       = len(owner_raw_pts)
    dimension     = len(owner_raw_pts[0])

    global_min   = float(np.min(pts))
    global_max   = float(np.max(pts))
    scale_factor = (global_max - global_min) if (global_max - global_min) != 0.0 else 1.0

    print(f"▶ 추출된 메타데이터: 전체 데이터 수(N)={total_N}, 차원(Dimension)={dimension}")

    base_epsilon_norm  = min((min_pts / total_N) ** (1 / dimension), 1.0)
    query_epsilon_norm = query_epsilon / scale_factor
    t_step_1_2 = time.time() - t0

    # ================================================================
    # [Step 3] FC Keygen 및 그리드 생성
    # ================================================================
    t1 = time.time()
    print("▶ [FC] FHE 키 발급(Keygen) 및 그리드 생성 중...")
    engine = Engine(use_bootstrap=True, mode="gpu")
    secret_key = engine.create_secret_key()
    keypack = KeyPack(
        public_key=engine.create_public_key(secret_key),
        rotation_key=engine.create_rotation_key(secret_key),
        relinearization_key=engine.create_relinearization_key(secret_key),
        conjugation_key=engine.create_conjugation_key(secret_key),
        bootstrap_key=engine.create_bootstrap_key(secret_key),
    )

    bucket_size          = 4
    max_blocks_per_grid  = 1

    domain_mins_norm = [0.0] * dimension
    domain_maxs_norm = [1.0] * dimension

    # [수정] GridIndex_plain → core.server.GridIndex
    grid_centers_norm = generate_public_grid_centers_nd(
        domain_mins_norm, domain_maxs_norm, base_epsilon_norm
    )
    axis_cell_counts = compute_axis_cell_counts(
        domain_mins_norm, domain_maxs_norm, base_epsilon_norm
    )
    t_step_3 = time.time() - t1

    # ================================================================
    # [Step 4] DO 암호화
    # ================================================================
    t2 = time.time()
    print("▶ [DO] 데이터 버케팅 및 동형암호화(Encryption) 진행 중...")

    # [수정] 반환값 분리: (client_blocks, server_blocks) 튜플
    client_blocks, server_blocks = prepare_and_encrypt_owner_blocks(
        engine=engine, keypack=keypack,
        owner_points_raw=owner_raw_pts,
        domain_mins_norm=domain_mins_norm,
        domain_maxs_norm=domain_maxs_norm,
        epsilon_norm=base_epsilon_norm,
        axis_cell_counts=axis_cell_counts,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid,
        global_min=global_min,
        global_max=global_max,
        owner_id=0,
    )

    # [수정] client_blocks로 인덱스 부여 + 좌표 맵 구성
    all_client_blocks_list = [client_blocks]
    all_points_plain, _   = assign_global_indices(all_client_blocks_list)
    coord_map             = build_owner_coord_map(all_client_blocks_list)  # [추가]

    total_points_upper_bound = (
        len(grid_centers_norm) * max_blocks_per_grid * bucket_size
    )
    t_step_4 = time.time() - t2

    # ================================================================
    # [Step 6] Server 연산
    # ================================================================
    t3 = time.time()
    print("▶ [Server] 수신한 암호문으로 DBSCAN 연산 시작 (시간이 오래 소요됩니다)...")

    # [수정] server_blocks만 서버에 전달 (평문 좌표 없음)
    final_result_ct = run_multiparty_point_dbscan(
        engine=engine,
        keypack=keypack,
        encrypted_server_blocks_list=[server_blocks],  # [수정]
        grid_centers_norm=grid_centers_norm,
        query_epsilon_norm=query_epsilon_norm,
        base_epsilon_norm=base_epsilon_norm,
        min_pts=min_pts,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid,
        total_points_upper_bound=total_points_upper_bound,
        adj_chunk_size=2000,   # Phase 2 청크 크기 (CT 1개 ≈ 500KB → ~1GB)
    )
    t_step_6 = time.time() - t3

    # ================================================================
    # [Step 8] FC 복호화 및 CSV 저장
    # ================================================================
    print("▶ [FC] 서버 결과 복호화(Decryption) 및 결과 맵핑 중...")

    # [수정] owner_coord_map 추가 전달
    results = reconstruct_results(
        engine=engine,
        secret_key=secret_key,
        final_ct=final_result_ct,
        plain_all_points=all_points_plain,
        owner_coord_map=coord_map,        # [수정]
        global_min=global_min,
        scale_factor=scale_factor,
    )

    print("\n================= ⏱️ 수행 시간 측정 결과 =================")
    print(f"Step 1~2 (FC 전처리): {t_step_1_2:.4f} 초")
    print(f"Step 3   (FC Keygen): {t_step_3:.4f} 초")
    print(f"Step 4   (DO 암호화) : {t_step_4:.4f} 초")
    print(f"Step 6   (S 연산)    : {t_step_6:.4f} 초")
    print("======================================================")

    csv_filename = "clustering_results.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "global_idx", "owner_id", "owner_local_idx",
            "point_reconstructed", "grid_idx", "block_idx", "label"
            # [수정] point_norm 제거 (서버에 노출 방지 위해 FinalClient 로컬 관리)
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"▶ 결과를 {csv_filename} 파일로 저장했습니다. (총 {len(results)}개 노드)")


if __name__ == '__main__':
    main()
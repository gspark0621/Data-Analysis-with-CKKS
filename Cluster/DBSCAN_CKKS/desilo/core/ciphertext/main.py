# main.py

import time
import csv
import math
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack

from core.ciphertext.client.FinalClient import (
    assign_global_indices,
    build_owner_coord_map,
    reconstruct_results,
)
from core.ciphertext.client.GridIndex import (
    generate_public_grid_centers_nd,
    compute_axis_cell_counts,          # 참고용 (현재 미사용, 필요 시 활용)
)
from core.ciphertext.client.MultipartyDataOwner import prepare_and_encrypt_owner_blocks
from core.ciphertext.server.MultipartyServer    import run_multiparty_point_dbscan


DATASET_PATH = (
    "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/"
    "DBSCAN_CKKS/desilo/dataset/Other_cluster/hepta.arff"
)


# ────────────────────────────────────────────────────────────
# 데이터 로더
# ────────────────────────────────────────────────────────────

def load_arff_to_pts_with_labels(filepath: str):
    pts, true_labels = [], []
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
                line   = line.replace('\t', ' ').replace(',', ' ')
                values = line.split()
                if len(values) < 2:
                    continue
                pts.append([float(v) for v in values[:-1]])
                true_labels.append(int(float(values[-1])))
    if not pts:
        raise ValueError("데이터를 찾을 수 없습니다.")
    return np.array(pts, dtype=np.float64), np.array(true_labels, dtype=int)


# ────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────

def main():
    print("=" * 54)
    print("   FHE Column-Major DBSCAN  (단일 DO 검증 모드)")
    print("=" * 54)

    # ── Step 0: 파라미터 입력 ────────────────────────────────
    query_epsilon = float(input("▶ eps 값 입력 (예: 0.5) > "))
    min_pts       = int(float(input("▶ min_pts 값 입력 (예: 3) > ")))

    # ── Step 1~2: 데이터 로드 & 메타데이터 추출 ──────────────
    t0 = time.time()
    print(f"\n▶ [FC] 데이터셋 로딩: {DATASET_PATH}")
    pts, _ = load_arff_to_pts_with_labels(DATASET_PATH)

    owner_raw_pts = pts.tolist()
    total_N       = len(owner_raw_pts)
    dimension     = len(owner_raw_pts[0])

    global_min   = float(np.min(pts))
    global_max   = float(np.max(pts))
    scale_factor = (global_max - global_min) or 1.0

    print(f"  N={total_N}, dim={dimension}, "
          f"min={global_min:.4f}, max={global_max:.4f}")

    # 정규화 공간에서의 epsilon 값
    query_epsilon_norm = query_epsilon / scale_factor   # e.g. 0.5/7 ≈ 0.071
    baseepsilonnorm    = query_epsilon_norm              # 이전에는 minpts/N/dim

    print(f"  baseepsilonnorm={baseepsilonnorm:.6f}, "
          f"queryepsilonnorm={query_epsilon_norm:.6f}")
    t_step_12 = time.time() - t0

    # ── Step 3: FC Keygen + 그리드 생성 ─────────────────────
    t1 = time.time()
    print("\n▶ [FC] FHE Keygen 및 그리드 생성 중...")

    engine     = Engine(use_bootstrap=True, mode="gpu")
    secret_key = engine.create_secret_key()
    keypack    = KeyPack(
        public_key          = engine.create_public_key(secret_key),
        rotation_key        = engine.create_rotation_key(secret_key),
        relinearization_key = engine.create_relinearization_key(secret_key),
        conjugation_key     = engine.create_conjugation_key(secret_key),
        bootstrap_key       = engine.create_bootstrap_key(secret_key),
    )

    domain_mins_norm = [0.0] * dimension
    domain_maxs_norm = [1.0] * dimension

    grid_centers_norm = generate_public_grid_centers_nd(
        domain_mins_norm, domain_maxs_norm, baseepsilonnorm
    )
    G = len(grid_centers_norm)

    # safety_factor:
    # 균등/랜덤 분포          → 2  (포아송 기댓값의 2배)
    # 일반 클러스터 데이터    → 3  (3σ 수준, 범용 권장)
    # hepta/밀집 구형 클러스터 → 5~8 (클러스터가 1~2개 셀에 몰릴 때)
    safety_factor = 5
    bucket_size              = max(int(math.ceil(total_N / G * safety_factor)), 10)  
    N_batch                  = bucket_size * G                                      
    total_points_upper_bound = N_batch

    print(f"  격자 수 G={G},  N_batch={N_batch}")
    t_step_3 = time.time() - t1

    # ── Step 4: DO 암호화 ────────────────────────────────────
    t2 = time.time()
    print("\n▶ [DO] 컬럼-메이저 패킹 및 동형암호화 진행 중...")

    client_pack, server_pack = prepare_and_encrypt_owner_blocks(
        engine            = engine,
        keypack           = keypack,
        owner_points_raw  = owner_raw_pts,
        grid_centers_norm = grid_centers_norm,  # ★ 변경
        epsilon_norm      = baseepsilonnorm,
        bucket_size       = bucket_size,
        global_min        = global_min,
        global_max        = global_max,
        owner_id          = 0,
    )

    # FinalClient: 전역 인덱스 부여 + 로컬 좌표 맵
    all_client_packs   = [client_pack]
    all_points, total  = assign_global_indices(all_client_packs)
    coord_map          = build_owner_coord_map(all_client_packs)

    print(f"  전역 인덱스 부여 완료: {total}개 포인트")
    t_step_4 = time.time() - t2

    # ── Step 6: 서버 DBSCAN 연산 ─────────────────────────────
    t3 = time.time()
    print("\n▶ [Server] FHE DBSCAN 연산 시작 (시간 소요)...")

    final_result_ct = run_multiparty_point_dbscan(
        engine                   = engine,
        keypack                  = keypack,
        encrypted_owner_packs    = [server_pack],  # ★ 변경
        grid_centers_norm        = grid_centers_norm,
        query_epsilon_norm       = query_epsilon_norm,
        base_epsilon_norm        = baseepsilonnorm,
        min_pts                  = min_pts,
        bucket_size              = bucket_size,
        total_points_upper_bound = total_points_upper_bound,
    )
    t_step_6 = time.time() - t3

    # ── Step 8: FC 복호화 & CSV 저장 ─────────────────────────
    print("\n▶ [FC] 복호화 및 결과 매핑 중...")

    results = reconstruct_results(
        engine           = engine,
        secret_key       = secret_key,
        final_ct         = final_result_ct,
        plain_all_points = all_points,
        owner_coord_map  = coord_map,     # ★ 변경
        global_min       = global_min,
        scale_factor     = scale_factor,
    )

    # ── 수행 시간 출력 ───────────────────────────────────────
    print("\n" + "=" * 54)
    print(f"  Step 1~2  FC 전처리 : {t_step_12:.4f} 초")
    print(f"  Step 3    FC Keygen : {t_step_3:.4f} 초")
    print(f"  Step 4    DO 암호화 : {t_step_4:.4f} 초")
    print(f"  Step 6    S  연산   : {t_step_6:.4f} 초")
    print("=" * 54)

    # ── CSV 저장 ────────────────────────────────────────────
    csv_filename = "clustering_results.csv"
    fieldnames   = [
        "global_idx", "owner_id", "owner_local_idx",
        "point_reconstructed",   # FinalClient 로컬 좌표 복원 결과
        "grid_idx", "label",
        # ★ point_norm 제거 — 서버 전달 방지 설계와 일관성 유지
    ]
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n▶ 저장 완료: {csv_filename}  (총 {len(results)}개 노드)")


if __name__ == '__main__':
    main()
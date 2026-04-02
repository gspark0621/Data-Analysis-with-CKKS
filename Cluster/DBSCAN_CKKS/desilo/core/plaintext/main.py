# core/plain/main.py
import math
from core.plaintext.GridIndex_plain import (
    generate_public_grid_centers_nd,
    compute_axis_cell_counts,
)
from core.plaintext.MultipartyOwner_plain import prepare_owner_blocks_plain
from core.plaintext.MultipartyServer_plain import run_multiparty_point_dbscan_plain
from core.plaintext.FinalClient_plain import reconstruct_results_plain


def main():
    print("========== Plaintext Multiparty B-Model DBSCAN (Ciphertext Architecture) ==========")

    epsilon = 2.0  
    min_pts = 3
    bucket_size = 4

    owner_A = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.1], [1.2, 1.2]]
    owner_B = [[2.1, 2.1], [2.2, 2.2], [2.3, 2.3], [3.0, 3.0]]
    owner_C = [[10.1, 10.1], [10.2, 10.2], [11.1, 11.1], [14.0, 14.0]]

    all_pts = owner_A + owner_B + owner_C
    N = len(all_pts)
    dimension = len(all_pts[0]) if N > 0 else 2

    global_min = min(min(row) for row in all_pts)
    global_max = max(max(row) for row in all_pts)
    scale_factor = global_max - global_min if global_max != global_min else 1.0
    
    domain_mins_norm = [0.0] * dimension
    domain_maxs_norm = [1.0] * dimension

    base_epsilon_norm = min((min_pts / N) ** (1 / dimension), 1.0)
    query_epsilon_norm = epsilon / scale_factor

    print(f"[Final Client] 초기화: N={N}, global_min={global_min}, scale_factor={scale_factor}")

    grid_centers_norm = generate_public_grid_centers_nd(domain_mins_norm, domain_maxs_norm, base_epsilon_norm)
    axis_cell_counts = compute_axis_cell_counts(domain_mins_norm, domain_maxs_norm, base_epsilon_norm)
    max_blocks_per_grid = max(1, math.ceil(N / bucket_size))

    print("\n[Data Owners] 원본 좌표 폐기 및 정규화 기반 블록화 진행 중...")
    blocks_A, _ = prepare_owner_blocks_plain(owner_A, domain_mins_norm, domain_maxs_norm, base_epsilon_norm, axis_cell_counts, bucket_size, max_blocks_per_grid, global_min, global_max, 0)
    blocks_B, _ = prepare_owner_blocks_plain(owner_B, domain_mins_norm, domain_maxs_norm, base_epsilon_norm, axis_cell_counts, bucket_size, max_blocks_per_grid, global_min, global_max, 1)
    blocks_C, _ = prepare_owner_blocks_plain(owner_C, domain_mins_norm, domain_maxs_norm, base_epsilon_norm, axis_cell_counts, bucket_size, max_blocks_per_grid, global_min, global_max, 2)

    print("\n[Server] 정규화된 점(point_norm)만을 이용하여 다자간 DBSCAN 수행 중...")
    server_result = run_multiparty_point_dbscan_plain(
        owner_blocks_list=[blocks_A, blocks_B, blocks_C],
        grid_centers_norm=grid_centers_norm,
        query_epsilon_norm=query_epsilon_norm,
        base_epsilon_norm=base_epsilon_norm,
        min_pts=min_pts,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid
    )

    print("\n[Final Client] 서버 결과 수신 및 Scale-up(역정규화) 복원 중...")
    # 파라미터 변경: original_owner_points_list 대신 global_min과 scale_factor를 넘겨 수학적 복원 수행
    results = reconstruct_results_plain(server_result, global_min, scale_factor)
    
    for res in sorted(results, key=lambda x: x["global_idx"]):
        print(f"GID: {res['global_idx']:2d} | Owner: {res['owner_id']} | "
              f"Recon_Pt: {res['point_reconstructed']} | "  # 복원된 좌표 출력
              f"Neighbors: {res['neighbor_count']:2d} | "
              f"Core: {res['is_core']} | Label: {res['label']}")

if __name__ == "__main__":
    main()
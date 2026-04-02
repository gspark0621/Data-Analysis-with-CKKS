# main_multiparty_b.py
import desilofhe
from util.keypack import KeyPack

from core.server.GridIndex import generate_public_grid_centers
from core.client.MultipartyDataOwner import encrypt_owner_blocks
from core.server.MultipartyServer import run_multiparty_point_dbscan
from core.client.FinalClient import decrypt_cluster_labels


def main():
    print("========== KDCA 초기화 ==========")
    engine = desilofhe.Engine(use_bootstrap=True, mode="gpu")
    kdca_secret_key = engine.create_secret_key()
    kdca_public_key = engine.create_public_key(kdca_secret_key)

    keypack = KeyPack(
        public_key=kdca_public_key,
        rotation_key=engine.create_rotation_key(kdca_secret_key),
        relinearization_key=engine.create_relinearization_key(kdca_secret_key),
        conjugation_key=engine.create_conjugation_key(kdca_secret_key),
        bootstrap_key=engine.create_bootstrap_key(kdca_secret_key)
    )

    domain_min_x, domain_max_x = 0.0, 16.0
    domain_min_y, domain_max_y = 0.0, 16.0
    epsilon = 2.0
    min_pts = 4
    bucket_size = 8
    max_blocks_per_grid = 2

    grid_centers = generate_public_grid_centers(
        domain_min_x, domain_max_x, domain_min_y, domain_max_y, epsilon
    )

    owner_A = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.1], [1.2, 1.2]]
    owner_B = [[2.1, 2.1], [2.2, 2.2], [2.3, 2.3], [3.0, 3.0]]
    owner_C = [[10.1, 10.1], [10.2, 10.2], [11.1, 11.1], [14.0, 14.0]]

    all_pts = owner_A + owner_B + owner_C
    global_min = min(min(row) for row in all_pts)
    global_max = max(max(row) for row in all_pts)
    scale = global_max - global_min if global_max != global_min else 1.0
    normalized_eps = epsilon / scale

    print("========== 각 Data Owner 암호화 ==========")
    enc_A, _ = encrypt_owner_blocks(
        engine, keypack, owner_A, grid_centers, epsilon,
        bucket_size, max_blocks_per_grid, global_min, global_max
    )
    enc_B, _ = encrypt_owner_blocks(
        engine, keypack, owner_B, grid_centers, epsilon,
        bucket_size, max_blocks_per_grid, global_min, global_max
    )
    enc_C, _ = encrypt_owner_blocks(
        engine, keypack, owner_C, grid_centers, epsilon,
        bucket_size, max_blocks_per_grid, global_min, global_max
    )

    total_points_upper_bound = len(grid_centers) * max_blocks_per_grid * bucket_size * 3

    print("========== 서버의 multiparty B-model DBSCAN ==========")
    encrypted_result = run_multiparty_point_dbscan(
        engine=engine,
        keypack=keypack,
        encrypted_owner_blocks_list=[enc_A, enc_B, enc_C],
        grid_centers=grid_centers,
        epsilon=epsilon,
        normalized_eps=normalized_eps,
        min_pts=min_pts,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid,
        total_points_upper_bound=total_points_upper_bound
    )

    print("========== KDCA 복호화 ==========")
    result_pts, labels = decrypt_cluster_labels(
        engine,
        kdca_secret_key,
        encrypted_result,
        total_points_upper_bound,
        all_pts
    )

    print(result_pts)
    print(labels)


if __name__ == "__main__":
    main()
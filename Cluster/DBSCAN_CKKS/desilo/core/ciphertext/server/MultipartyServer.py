# core/server/MultipartyServer.py
from core.server.Normalize import check_neighbor_closed_interval
from core.server.Core import identify_core_points_fhe_converted
from core.server.LabelPropagation import fhe_max_propagation_fhe, _refresh
from core.server.GridIndex import build_grid_adjacency


def multiply_masked_coords(engine, coord_ct, mask_ct, relin_key):
    return engine.multiply(coord_ct, mask_ct, relin_key)


def compare_two_blocks(engine,
                       keypack,
                       left_coords,
                       left_mask_pt,
                       right_coords,
                       right_mask_pt,
                       bucket_size,
                       eps,
                       dimension):
    relin_key = keypack.relinearization_key

    left_mask = engine.encode(left_mask_pt)
    right_mask = engine.encode(right_mask_pt)

    masked_left = [engine.multiply(c, left_mask) for c in left_coords]
    masked_right = [engine.multiply(c, right_mask) for c in right_coords]

    neighbor_ct_list = []
    total_neighbor_from_this_pair = None

    for k in range(bucket_size):
        dist_sq_k = None
        for d in range(dimension):
            rotated_right = engine.rotate(masked_right[d], keypack.rotation_key, k)
            diff_ct = engine.subtract(masked_left[d], rotated_right)
            sq_ct = engine.square(diff_ct, relin_key)
            dist_sq_k = sq_ct if dist_sq_k is None else engine.add(dist_sq_k, sq_ct)

        adj_k = check_neighbor_closed_interval(engine, dist_sq_k, eps ** 2, keypack, dimension)
        neighbor_ct_list.append(adj_k)
        total_neighbor_from_this_pair = adj_k if total_neighbor_from_this_pair is None else engine.add(total_neighbor_from_this_pair, adj_k)

    return total_neighbor_from_this_pair, neighbor_ct_list


def run_multiparty_point_dbscan(engine,
                                keypack,
                                encrypted_owner_blocks_list,
                                grid_centers,
                                epsilon,
                                normalized_eps,
                                min_pts,
                                bucket_size,
                                max_blocks_per_grid,
                                total_points_upper_bound):
    """
    encrypted_owner_blocks_list:
      [
        [ {enc_coords:[ct_x, ct_y], enc_mask:ct, grid_idx:g, block_idx:b}, ... ],   # owner A
        [ ... ],                                                                    # owner B
        ...
      ]
    """
    adjacency_grid = build_grid_adjacency(grid_centers, epsilon)
    num_grids = len(grid_centers)

    # 전역 point graph adjacency 누적용
    adjacency_ct_list = []
    total_neighbors_ct = None

    # 전역 point 수 상한 = owner수 * grid수 * block수 * bucket_size
    N = total_points_upper_bound
    dimension = 2

    # owner별 block lookup
    owner_lookup = []
    for owner_blocks in encrypted_owner_blocks_list:
        table = {}
        for blk in owner_blocks:
            table[(blk["grid_idx"], blk["block_idx"])] = blk
        owner_lookup.append(table)

    for g in range(num_grids):
        neighbor_grids = [j for j, val in enumerate(adjacency_grid[g]) if val == 1]

        for owner_a_idx, table_a in enumerate(owner_lookup):
            for block_a in range(max_blocks_per_grid):
                if (g, block_a) not in table_a:
                    continue
                left_blk = table_a[(g, block_a)]

                for ng in neighbor_grids:
                    for owner_b_idx, table_b in enumerate(owner_lookup):
                        for block_b in range(max_blocks_per_grid):
                            if (ng, block_b) not in table_b:
                                continue
                            right_blk = table_b[(ng, block_b)]

                            pair_neighbor_sum_ct, pair_adj_list = compare_two_blocks(
                                engine=engine,
                                keypack=keypack,
                                left_coords=left_blk["enc_coords"],
                                left_mask_pt=left_blk["selection_mask_pt"],
                                right_coords=right_blk["enc_coords"],
                                right_mask_pt=right_blk["selection_mask_pt"],
                                bucket_size=bucket_size,
                                eps=normalized_eps,
                                dimension=dimension
                            )

                            if total_neighbors_ct is None:
                                total_neighbors_ct = pair_neighbor_sum_ct
                            else:
                                total_neighbors_ct = engine.add(total_neighbors_ct, pair_neighbor_sum_ct)

                            adjacency_ct_list.extend(pair_adj_list)

    if total_neighbors_ct is None:
        zero_pt = engine.encode([0.0 for _ in range(N)])
        total_neighbors_ct = zero_pt

    ones_plaintext = engine.encode([1.0 for _ in range(N)])
    total_neighbors_ct = engine.add(total_neighbors_ct, ones_plaintext)

    core_ct = identify_core_points_fhe_converted(
        engine, total_neighbors_ct, min_pts, N, keypack=keypack
    )

    cluster_id_pt = [(i + 1) / float(N + 1) for i in range(N)]
    final_norm_ct = fhe_max_propagation_fhe(
        engine,
        keypack,
        adjacency_ct_list,
        core_ct,
        cluster_id_pt,
        N,
        max_iter=5
    )

    scale_back_pt = engine.encode([float(N + 1) for _ in range(N)])
    final_ct = engine.multiply(final_norm_ct, scale_back_pt)
    final_ct = _refresh(engine, final_ct, keypack)

    return final_ct
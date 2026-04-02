import numpy as np
from core.plaintext.Normalize import check_neighbor_closed_interval_np
from core.plaintext.Core import identify_core_points_np
from core.plaintext.Label_Propagation import fhe_max_propagation_np

def send_to_server_np(encrypted_columns, num_points, eps, min_pts, dimension):
    dim = len(encrypted_columns)
    N = num_points
    adj_k_list = []
    total_neighbors_ct = None

    for k in range(1, N):
        dist_sq_k = None
        for d in range(dim):
            base_col = encrypted_columns[d]
            rotated_col = np.roll(base_col, -k)
            diff_ct = base_col - rotated_col
            sq_ct = diff_ct ** 2

            if dist_sq_k is None:
                dist_sq_k = sq_ct
            else:
                dist_sq_k = dist_sq_k + sq_ct

        adj_k = check_neighbor_closed_interval_np(dist_sq_k, eps**2, dimension)
        adj_k_list.append(adj_k)

        if total_neighbors_ct is None:
            total_neighbors_ct = adj_k
        else:
            total_neighbors_ct = total_neighbors_ct + adj_k

    ones_plaintext = np.ones(N, dtype=np.float64)
    total_neighbors_ct = total_neighbors_ct + ones_plaintext

    core_ct = identify_core_points_np(total_neighbors_ct, min_pts, N)

    cluster_id_pt = [float(i + 1) for i in range(N)]
    final_ct, iter_count = fhe_max_propagation_np(
        adjacency_ct_list=adj_k_list, 
        core_ct=core_ct, 
        cluster_id_pt=cluster_id_pt, 
        num_points=N
    )
    
    # 🚨 수정됨: 디버깅용 중간 상태값 추출
    debug_np = {
        'total_neighbors': np.array(total_neighbors_ct).copy(),
        'core_mask': np.array(core_ct).copy(),
        'final_labels': np.array(final_ct).copy()
    }
    
    return final_ct, iter_count, debug_np # debug_np 추가 반환

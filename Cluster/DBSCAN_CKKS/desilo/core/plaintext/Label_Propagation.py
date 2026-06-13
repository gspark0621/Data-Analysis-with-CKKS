import math
import numpy as np

def fhe_sign_unit(x_ct, num_points: int, depth: int = 5):
    current_x = x_ct
    for i in range(depth):
        x_sq = current_x ** 2
        x_cub = x_sq * current_x
        term1 = current_x * 1.5
        term2 = x_cub * 0.5
        current_x = term1 - term2
    return np.round(current_x)

def fhe_fast_max_unit(A_ct, B_ct, num_points: int, depth: int = None):
    diff_ct = B_ct - A_ct
    scale_factor = 1.0 / float(num_points)
    scaled_diff_ct = diff_ct * scale_factor
    
    if depth is None:
        min_initial_val = scale_factor
        depth = math.ceil(math.log(1.0 / min_initial_val, 1.5)) + 1
        
    sign_ct = fhe_sign_unit(scaled_diff_ct, num_points, depth=depth)
    diff_sign_ct = diff_ct * sign_ct
    relu_ct = diff_ct + diff_sign_ct
    relu_ct = relu_ct * 0.5
    return A_ct + relu_ct

def fhe_hard_mask01(x_ct, num_points: int, depth: int = 4):
    centered_ct = x_ct - 0.5
    sign_ct = fhe_sign_unit(centered_ct, num_points, depth=depth)
    out_ct = (sign_ct * 0.5) + 0.5
    return out_ct

def _single_rotate_and_mask_plain(ct, k):
    """
    FHE 버전의 '2배 패킹 + 단일 회전 + valid_mask_pt 마스킹' 로직은
    길이 N인 평문 배열에서는 np.roll과 수학적/논리적으로 완벽히 동일합니다.
    """
    return np.roll(ct, -k)

def fhe_max_propagation_np(
    adjacency_ct_list: list,
    core_ct,
    cluster_id_pt: list,
    num_points: int,
    max_iter: int = None
):
    cluster_id_encoded = np.array(cluster_id_pt, dtype=np.float64)

    clean_adj_list = []
    # print("adj_ct start")
    for adj_ct in adjacency_ct_list:
        adj_ct_masked = fhe_hard_mask01(adj_ct, num_points, depth=4)
        clean_adj_list.append(adj_ct_masked)
    # print("adj_ct end")

    core_mask_ct = fhe_hard_mask01(core_ct, num_points, depth=4)
    non_core_mask_ct = 1.0 - core_mask_ct
    non_core_mask_ct = fhe_hard_mask01(non_core_mask_ct, num_points, depth=4)

    core_labels_ct = core_mask_ct * cluster_id_encoded
    zero_ct = np.zeros(num_points, dtype=np.float64)

    # ✅ [수정된 부분] for문을 지우고, while문으로 iter_count를 계산합니다.
    iter_count = 0
    tolerance = 1e-4  

    while True:
        iter_count += 1
        # print(f"core-core start (Iteration: {iter_count})")
        
        prev_labels = core_labels_ct.copy()  # 수렴 비교용 저장
        
        for idx, adj_ct in enumerate(clean_adj_list):
            k = idx + 1
            shifted_core_labels = _single_rotate_and_mask_plain(core_labels_ct, k)
            shifted_core_mask = _single_rotate_and_mask_plain(core_mask_ct, k)

            edge_mask_ct = adj_ct * core_mask_ct * shifted_core_mask
            candidate_labels_ct = edge_mask_ct * shifted_core_labels

            core_labels_ct = fhe_fast_max_unit(core_labels_ct, candidate_labels_ct, num_points)
            core_labels_ct = core_labels_ct * core_mask_ct
            
        # ✅ 최대 차이 계산 및 탈출 조건
        max_diff = np.max(np.abs(core_labels_ct - prev_labels))
        if max_diff < tolerance:
            print(f"[알림] 라벨 수렴 완료. 필요 max_iter: {iter_count}")
            break
        if iter_count >= num_points:
            break

    border_labels_ct = zero_ct.copy()
    assigned_mask_ct = zero_ct.copy()

    for idx, adj_ct in enumerate(clean_adj_list):
        # print("border assignment start")
        # (기존 Border 할당 로직 동일)
        k = idx + 1
        shifted_core_labels = _single_rotate_and_mask_plain(core_labels_ct, k)
        shifted_core_mask = _single_rotate_and_mask_plain(core_mask_ct, k)

        cand_mask_ct = adj_ct * shifted_core_mask * non_core_mask_ct
        cand_mask_ct = fhe_hard_mask01(cand_mask_ct, num_points, depth=4)

        empty_mask_ct = 1.0 - assigned_mask_ct
        empty_mask_ct = fhe_hard_mask01(empty_mask_ct, num_points, depth=4)

        accept_mask_ct = cand_mask_ct * empty_mask_ct
        accept_mask_ct = fhe_hard_mask01(accept_mask_ct, num_points, depth=4)

        accepted_labels_ct = accept_mask_ct * shifted_core_labels
        border_labels_ct = border_labels_ct + accepted_labels_ct

        assigned_mask_ct = fhe_fast_max_unit(assigned_mask_ct, accept_mask_ct, num_points)
        assigned_mask_ct = fhe_hard_mask01(assigned_mask_ct, num_points, depth=4)

    final_labels_norm_ct = core_labels_ct + border_labels_ct
    
    # ✅ [수정된 부분] iter_count를 함께 반환합니다.
    return final_labels_norm_ct, iter_count
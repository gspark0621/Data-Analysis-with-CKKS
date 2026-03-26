import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack


def _refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack) -> Ciphertext:
    return engine.bootstrap(
        engine.intt(ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key
    )


def fhe_circular_shift(engine, ct, k, num_points, keypack):
    """
    1배 패킹된 암호문을 평문의 np.roll(ct, -k)와 동일하게 만드는 함수.
    나머지 잉여 슬롯은 철저하게 0.0으로 마스킹하여 노이즈 침범을 막습니다.
    """
    slot_count = engine.slot_count
    
    # 1. 왼쪽 시프트 (앞부분 데이터 보존용)
    left_shifted = engine.rotate(ct, keypack.rotation_key, -k)
    
    # 2. 오른쪽 시프트 (꼬리로 넘어간 데이터 보존용)
    right_shifted = engine.rotate(ct, keypack.rotation_key, num_points - k)
    
    # 3. 평문 마스크 벡터 생성 (num_points 밖의 슬롯은 모조리 0.0 처리)
    # left_shifted는 [0 ~ num_points - k - 1] 구역만 살림
    mask_left_pt = [1.0 if i < (num_points - k) else 0.0 for i in range(slot_count)]
    
    # right_shifted는 [num_points - k ~ num_points - 1] 구역만 살림
    mask_right_pt = [0.0] * slot_count
    for i in range(num_points - k, num_points):
        mask_right_pt[i] = 1.0
        
    # 평문을 인코딩
    mask_left_encoded = engine.encode(mask_left_pt)
    mask_right_encoded = engine.encode(mask_right_pt)
    
    # 4. 마스킹(곱셈) 후 병합
    # 평문과의 곱셈이므로 relin_key가 필요하지 않습니다. (레벨만 1 소모)
    left_clean = engine.multiply(left_shifted, mask_left_encoded)
    right_clean = engine.multiply(right_shifted, mask_right_encoded)
    
    return engine.add(left_clean, right_clean)


def fhe_sign_unit(engine: Engine, x_ct: Ciphertext, num_points: int, keypack: KeyPack, depth: int = 5, bootstrap_interval: int = 3) -> Ciphertext:
    relin_key = keypack.relinearization_key
    c15_pt = engine.encode([1.5 for _ in range(num_points)])
    c05_pt = engine.encode([0.5 for _ in range(num_points)])
    current_x = x_ct
    for i in range(depth):
        x_sq = engine.square(current_x, relin_key)
        x_cub = engine.multiply(x_sq, current_x, relin_key)
        term1 = engine.multiply(current_x, c15_pt)
        term2 = engine.multiply(x_cub, c05_pt)
        current_x = engine.subtract(term1, term2)
        
        if (i + 1) % bootstrap_interval == 0 and (i + 1) != depth:
            current_x = _refresh(engine, current_x, keypack)
    return current_x


def fhe_fast_max_unit(engine: Engine, A_ct: Ciphertext, B_ct: Ciphertext, num_points: int, keypack: KeyPack, depth: int = 5) -> Ciphertext:
    relin_key = keypack.relinearization_key
    half_pt = engine.encode([0.5 for _ in range(num_points)])
    
    # [핵심 디버그] 라벨 값들은 이미 0~1 사이로 정규화되어 있으므로, 
    # 차이값은 자연스럽게 [-1, 1] 범위에 들어옵니다. (1/N 스케일링 삭제됨)
    diff_ct = engine.subtract(B_ct, A_ct)
    
    sign_ct = fhe_sign_unit(engine, diff_ct, num_points, keypack, depth=depth)
    
    sign_ct = _refresh(engine, sign_ct, keypack)
    diff_ct_ref = _refresh(engine, diff_ct, keypack)
    
    diff_sign_ct = engine.multiply(diff_ct_ref, sign_ct, relin_key)
    diff_sign_ct = _refresh(engine, diff_sign_ct, keypack)
    
    relu_ct = engine.add(diff_ct_ref, diff_sign_ct)
    relu_ct = engine.multiply(relu_ct, half_pt)
    
    final_ct = engine.add(A_ct, relu_ct)
    return _refresh(engine, final_ct, keypack)


def fhe_hard_mask01(engine: Engine, x_ct: Ciphertext, num_points: int, keypack: KeyPack, depth: int = 5) -> Ciphertext:
    half_pt = engine.encode([0.5 for _ in range(num_points)])
    centered_ct = engine.subtract(x_ct, half_pt)
    
    centered_ct = _refresh(engine, centered_ct, keypack)
    
    sign_ct = fhe_sign_unit(engine, centered_ct, num_points, keypack, depth=depth)
    
    out_ct = engine.add(engine.multiply(sign_ct, half_pt), half_pt)
    return _refresh(engine, out_ct, keypack)


#TODO: dataset에 대해서 iteration횟수를 계산하면 3~10임. 하지만 이를 이용해 clustering을 만든다면 이는 overfitting일 뿐, 실제 완벽한 해결책은 아님. N-1(worst case)로 고정하지 않고 새로운 방법을 생각
def fhe_max_propagation_fhe(
    engine: Engine,
    keypack: KeyPack,
    adjacency_ct_list: list,
    core_ct: Ciphertext,
    cluster_id_pt: list,
    num_points: int,
    max_iter: int = None
) -> Ciphertext:
    relin_key = keypack.relinearization_key
    cluster_id_encoded = engine.encode(cluster_id_pt)
    
    if max_iter is None:
        max_iter = num_points - 1

    clean_adj_list = []
    print("[Server] clean adj_ct start")
    for adj_ct in adjacency_ct_list:
        adj_ct = _refresh(engine, adj_ct, keypack)
        adj_ct = fhe_hard_mask01(engine, adj_ct, num_points, keypack, depth=4)
        clean_adj_list.append(adj_ct)
    print("[Server] clean adj_ct end")

    # Core 및 Non-core 마스크 생성
    core_mask_ct = _refresh(engine, core_ct, keypack)
    core_mask_ct = fhe_hard_mask01(engine, core_mask_ct, num_points, keypack, depth=4)

    non_core_mask_ct = engine.subtract(1.0, core_mask_ct)
    non_core_mask_ct = fhe_hard_mask01(engine, non_core_mask_ct, num_points, keypack, depth=4)

    core_labels_ct = engine.multiply(core_mask_ct, cluster_id_encoded)
    core_labels_ct = _refresh(engine, core_labels_ct, keypack)

    zero_ct = engine.subtract(core_labels_ct, core_labels_ct)

    # 1. Core-Core 전파 (고정 횟수 반복)
    for _ in range(max_iter):
        print("[Server] core-core propagation iteration")
        for idx, adj_ct in enumerate(clean_adj_list):
            k = idx + 1
            
            # [수정] 단일 회전 대신 완벽한 순환 시프트 회로 사용
            # 내부 마스킹이 포함되어 있으므로 외부 valid_mask_pt 곱셈 삭제
            shifted_core_labels = fhe_circular_shift(engine, core_labels_ct, k, num_points, keypack)
            shifted_core_mask = fhe_circular_shift(engine, core_mask_ct, k, num_points, keypack)

            # Circular Shift 연산 직후 레벨 확보를 위해 Refresh 수행 (안전 마진)
            shifted_core_labels = _refresh(engine, shifted_core_labels, keypack)
            shifted_core_mask = _refresh(engine, shifted_core_mask, keypack)

            edge_mask_ct = engine.multiply(adj_ct, core_mask_ct, relin_key)
            edge_mask_ct = engine.multiply(edge_mask_ct, shifted_core_mask, relin_key)
            edge_mask_ct = _refresh(engine, edge_mask_ct, keypack)

            candidate_labels_ct = engine.multiply(edge_mask_ct, shifted_core_labels, relin_key)
            candidate_labels_ct = _refresh(engine, candidate_labels_ct, keypack)

            core_labels_ct = fhe_fast_max_unit(engine, core_labels_ct, candidate_labels_ct, num_points, keypack)
            core_labels_ct = engine.multiply(core_labels_ct, core_mask_ct, relin_key)
            core_labels_ct = _refresh(engine, core_labels_ct, keypack)

    border_labels_ct = zero_ct
    assigned_mask_ct = zero_ct

    # 2. Border 할당
    for idx, adj_ct in enumerate(clean_adj_list):
        print("[Server] border assignment start")
        k = idx + 1
        
        # [수정] 순환 시프트 사용
        shifted_core_labels = fhe_circular_shift(engine, core_labels_ct, k, num_points, keypack)
        shifted_core_mask = fhe_circular_shift(engine, core_mask_ct, k, num_points, keypack)
        
        shifted_core_labels = _refresh(engine, shifted_core_labels, keypack)
        shifted_core_mask = _refresh(engine, shifted_core_mask, keypack)

        cand_mask_ct = engine.multiply(adj_ct, shifted_core_mask, relin_key)
        cand_mask_ct = engine.multiply(cand_mask_ct, non_core_mask_ct, relin_key)
        cand_mask_ct = _refresh(engine, cand_mask_ct, keypack)
        cand_mask_ct = fhe_hard_mask01(engine, cand_mask_ct, num_points, keypack, depth=4)

        empty_mask_ct = engine.subtract(1.0, assigned_mask_ct)
        empty_mask_ct = fhe_hard_mask01(engine, empty_mask_ct, num_points, keypack, depth=4)

        accept_mask_ct = engine.multiply(cand_mask_ct, empty_mask_ct, relin_key)
        accept_mask_ct = _refresh(engine, accept_mask_ct, keypack)
        accept_mask_ct = fhe_hard_mask01(engine, accept_mask_ct, num_points, keypack, depth=4)

        accepted_labels_ct = engine.multiply(accept_mask_ct, shifted_core_labels, relin_key)
        border_labels_ct = engine.add(border_labels_ct, accepted_labels_ct)
        border_labels_ct = _refresh(engine, border_labels_ct, keypack)

        assigned_mask_ct = fhe_fast_max_unit(engine, assigned_mask_ct, accept_mask_ct, num_points, keypack)
        assigned_mask_ct = fhe_hard_mask01(engine, assigned_mask_ct, num_points, keypack, depth=4)

    final_labels_norm_ct = engine.add(core_labels_ct, border_labels_ct)
    final_labels_norm_ct = _refresh(engine, final_labels_norm_ct, keypack)

    return final_labels_norm_ct


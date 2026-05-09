# core/ciphertext/server/MultipartyServer.py

from typing import Dict, List, Tuple
from core.ciphertext.server.Normalize        import check_neighbor_closed_interval
from core.ciphertext.server.Core             import identify_core_points_fhe_converted
from core.ciphertext.server.LabelPropagation import fhe_max_propagation_fhe, refresh
from core.ciphertext.client.GridIndex import (
    build_grid_adjacency,
    get_unique_grid_deltas,
    build_adjacency_mask_for_delta,
    compute_axis_cell_counts,       # [추가] maxiter 계산용
)


# ─────────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────────

def _rotate_mask_pt(mask_pt: list, k: int, N_batch: int) -> list:
    return [mask_pt[(s + k) % N_batch] for s in range(N_batch)]


def _compare_batch_with_shift(
    engine, keypack,
    left_enc_coords, left_mask_pt,
    right_enc_coords, right_mask_pt,
    k_global, adj_mask_pt,
    N_batch, eps_sq, dim,
):
    relin = keypack.relinearization_key

    rot_right = [
        engine.rotate(ct, keypack.rotation_key, k_global)
        for ct in right_enc_coords
    ]
    rot_right_mask_pt = _rotate_mask_pt(right_mask_pt, k_global, N_batch)

    combined_mask_pt = [
        left_mask_pt[s] * rot_right_mask_pt[s] * adj_mask_pt[s]
        for s in range(N_batch)
    ]

    left_mask_enc      = engine.encode(left_mask_pt)
    rot_right_mask_enc = engine.encode(rot_right_mask_pt)

    dist_sq_ct = None
    for d in range(dim):
        l_masked = engine.multiply(left_enc_coords[d], left_mask_enc)
        r_masked = engine.multiply(rot_right[d],       rot_right_mask_enc)
        diff     = engine.subtract(l_masked, r_masked)
        sq       = engine.square(diff, relin)
        dist_sq_ct = sq if dist_sq_ct is None else engine.add(dist_sq_ct, sq)

    adj_ct = check_neighbor_closed_interval(
        engine, dist_sq_ct, eps_sq, keypack, dim
    )
    adj_ct = engine.multiply(adj_ct, engine.encode(combined_mask_pt))
    return adj_ct


def _compute_maxiter_from_grid(
    domain_mins: List[float],
    domain_maxs: List[float],
    epsilon: float,
) -> int:
    """
    Grid 구조 기반으로 Label Propagation 의 maxiter 결정.

    [설계 근거]
    Column-major 패킹에서 1 iteration = 모든 직접 인접 그리드로 동시 1홉 전파.
    필요한 최대 반복 횟수 = 가장 멀리 떨어진 두 Core 그리드 간 홉 수.
    실용적 상한 = max(axis_cell_counts) (~sqrt(G) for 2D).

    [log2(G) 를 사용하지 않는 이유]
    log2(G) 는 이진 트리 탐색 가정 → 격자에서는 과소추정.
    예) G=100(10x10): log2(100)≈6.6 이지만 실제 최대 홉=18(대각선).

    안전 하한: 3 (매우 작은 grid 예외 처리).
    """
    axis_counts = compute_axis_cell_counts(domain_mins, domain_maxs, epsilon)
    maxiter = max(axis_counts)
    return max(maxiter, 3)


# ─────────────────────────────────────────────────────────────────
# 메인 서버 파이프라인
# ─────────────────────────────────────────────────────────────────

def run_multiparty_point_dbscan(
    engine,
    keypack,
    encrypted_owner_packs,
    grid_centers_norm,
    query_epsilon_norm,
    base_epsilon_norm,
    min_pts,
    bucket_size,
    total_points_upper_bound,
    domain_mins_norm: List[float] = None,   # [추가] maxiter 계산용
    domain_maxs_norm: List[float] = None,   # [추가] maxiter 계산용
):
    """
    FHE Multiparty DBSCAN 서버 파이프라인.

    변경 내역
    ---------
    1. _compute_maxiter_from_grid() 로 maxiter 를 grid 구조 기반 결정.
       기존 하드코딩 maxiter=5 제거.

    2. cluster_id_pt 초기화에 valid_mask_pt 적용.
       더미 슬롯(실제 포인트 없는 슬롯)의 cluster_id = 0.
       Core 마스크와의 곱에서 더미 슬롯이 자동 0 이 됨.

    3. domain_mins_norm / domain_maxs_norm 파라미터 추가.
       기본값 = [0.0]*dim / [1.0]*dim (정규화 도메인).
    """
    adjacency_grid = build_grid_adjacency(
        grid_centers_norm, query_epsilon_norm, base_epsilon_norm
    )
    G       = len(grid_centers_norm)
    dim     = len(grid_centers_norm[0])
    N_batch = bucket_size * G
    eps_sq  = query_epsilon_norm ** 2

    # domain 정보 기본값 설정
    if domain_mins_norm is None:
        domain_mins_norm = [0.0] * dim
    if domain_maxs_norm is None:
        domain_maxs_norm = [1.0] * dim

    unique_deltas = get_unique_grid_deltas(adjacency_grid, G)
    print(f"[Server] G={G}, N_batch={N_batch}, "
          f"unique_deltas={len(unique_deltas)}, "
          f"owners={len(encrypted_owner_packs)}")

    total_neighbors_ct = None
    adj_accum: Dict[int, object] = {}

    # ── 이웃 수 집계 & adj 누적 ──────────────────────────────
    for a_idx, pack_a in enumerate(encrypted_owner_packs):
        mask_a = pack_a["selection_mask_pt"]
        for ct in pack_a["enc_coords"]:
            engine.to_cuda(ct)

        for b_idx, pack_b in enumerate(encrypted_owner_packs):
            mask_b = pack_b["selection_mask_pt"]
            for ct in pack_b["enc_coords"]:
                engine.to_cuda(ct)

            for delta in unique_deltas:
                adj_mask_pt = build_adjacency_mask_for_delta(
                    delta, G, bucket_size, adjacency_grid
                )

                for k_local in range(bucket_size):
                    k_global = (k_local * G + delta + N_batch) % N_batch

                    if k_global == 0 and a_idx == b_idx:
                        continue

                    adj_ct = _compare_batch_with_shift(
                        engine, keypack,
                        pack_a["enc_coords"], mask_a,
                        pack_b["enc_coords"], mask_b,
                        k_global, adj_mask_pt,
                        N_batch, eps_sq, dim,
                    )

                    if total_neighbors_ct is None:
                        total_neighbors_ct = adj_ct
                    else:
                        engine.to_cuda(total_neighbors_ct)
                        total_neighbors_ct = engine.add(total_neighbors_ct, adj_ct)
                    engine.to_cpu(total_neighbors_ct)

                    if k_global not in adj_accum:
                        adj_accum[k_global] = adj_ct
                    else:
                        engine.to_cuda(adj_accum[k_global])
                        adj_accum[k_global] = engine.add(adj_accum[k_global], adj_ct)
                    engine.to_cpu(adj_accum[k_global])

                    del adj_ct

            for ct in pack_b["enc_coords"]:
                engine.to_cpu(ct)

        for ct in pack_a["enc_coords"]:
            engine.to_cpu(ct)

    # ── self-neighbor +1 ─────────────────────────────────────
    print("[Server] self-neighbor +1 및 sign_bootstrap refresh...")
    if total_neighbors_ct is None:
        zero_pt = engine.encode([0.0] * N_batch)
        total_neighbors_ct = engine.encrypt(zero_pt, keypack.public_key)
    else:
        engine.to_cuda(total_neighbors_ct)

    ones_pt = engine.encode([1.0] * N_batch)
    total_neighbors_ct = engine.add(total_neighbors_ct, ones_pt)

    # ── adj_ct_pairs 구성 ────────────────────────────────────
    adj_ct_pairs: List[Tuple[int, object]] = []
    for k_global, ct in sorted(adj_accum.items()):
        engine.to_cuda(ct)
        ct = refresh(engine, ct, keypack)   # adj_ct: 0~1 이진 -> scale=1.0
        engine.to_cpu(ct)
        adj_ct_pairs.append((k_global, ct))
    del adj_accum
    
    print(f"[Server] adj_ct_pairs: {len(adj_ct_pairs)}개 고유 k_global")

    # ── Core Point 판별 ──────────────────────────────────────
    core_ct = identify_core_points_fhe_converted(
        engine, total_neighbors_ct, min_pts, N_batch, keypack
    )

    # ── 초기 클러스터 ID: valid_mask 적용 ────────────────────
    # [변경] 각 Owner 의 selection_mask_pt OR 합산 -> valid_mask_pt 구성.
    # 유효 슬롯: cluster_id = (i+1)/(N_batch+1) in (0,1]  (슬롯별 고유값)
    # 더미 슬롯: cluster_id = 0
    valid_mask_pt = [0.0] * N_batch
    for pack in encrypted_owner_packs:
        for s, v in enumerate(pack["selection_mask_pt"]):
            if v > 0.0:
                valid_mask_pt[s] = 1.0

    cluster_id_pt = [
        ((i + 1) / float(N_batch + 1)) * valid_mask_pt[i]
        for i in range(N_batch)
    ]

    # ── maxiter: Grid 구조 기반 결정 ─────────────────────────
    # [변경] 하드코딩 maxiter=5 -> grid axis 최대 셀 수로 교체.
    maxiter = _compute_maxiter_from_grid(
        domain_mins_norm, domain_maxs_norm, base_epsilon_norm
    )
    print(f"[Server] Label Propagation maxiter={maxiter} "
          f"(axis_max 기반, base_eps_norm={base_epsilon_norm:.6f})")

    # ── Label Propagation ────────────────────────────────────
    print("[Server] Label Propagation 시작...")
    final_norm_ct = fhe_max_propagation_fhe(
        engine        = engine,
        keypack       = keypack,
        adj_ct_pairs  = adj_ct_pairs,
        core_ct       = core_ct,
        cluster_id_pt = cluster_id_pt,
        numpoints     = N_batch,
        maxiter       = maxiter,
    )

    scale_back = engine.encode([float(N_batch + 1)] * N_batch)
    final_ct   = engine.multiply(final_norm_ct, scale_back)
    return refresh(engine, final_ct, keypack, scale=float(N_batch + 1))
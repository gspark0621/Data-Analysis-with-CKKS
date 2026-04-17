# core/ciphertext/client/MultipartyDataOwner.py
import math
import random
from core.ciphertext.client.GridIndex import normalize_points_global, bucketize_owner_blocks


def prepare_and_encrypt_owner_blocks(engine, keypack,
                                     owner_points_raw,
                                     domain_mins_norm, domain_maxs_norm,
                                     epsilon_norm, axis_cell_counts,
                                     bucket_size, max_blocks_per_grid,
                                     global_min, global_max, owner_id):
    """
    [OOM 수정] 암호화 직후 engine.to_cpu() 로 GPU → CPU RAM 이동.

    bootstrap key 등 고정 키가 GPU 메모리 대부분을 점유하므로
    블록 CT를 GPU에 상주시키면 즉시 OOM 발생.
    → 암호화 직후 to_cpu() 로 CPU RAM에 보관.
    → 비교 시 서버가 to_cuda() 로 2개 블록만 순간 GPU 적재.

    GPU 상주 CT:
      비교 중: left 블록 1개 + right 블록 1개 + 중간 연산 CT (~2-3 GB)
      비교 외: running_sum 1개만
    """
    dim = len(owner_points_raw[0]) if owner_points_raw else 2

    owner_points_norm, _ = normalize_points_global(
        owner_points_raw, global_min, global_max
    )

    plain_blocks = bucketize_owner_blocks(
        points_norm=owner_points_norm,
        domain_mins_norm=domain_mins_norm,
        domain_maxs_norm=domain_maxs_norm,
        epsilon_norm=epsilon_norm,
        axis_cell_counts=axis_cell_counts,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid,
        owner_id=owner_id,
    )

    non_empty_blocks = [
        blk for blk in plain_blocks if sum(blk["selection_mask"]) > 0
    ]

    # Oblivious Padding
    total_N     = len(owner_points_raw)
    upper_bound = math.ceil(total_N / bucket_size) * max_blocks_per_grid
    num_dummies = max(0, upper_bound - len(non_empty_blocks))
    dummy_blocks = [{
        "points_norm":    [[0.0] * dim] * bucket_size,
        "point_refs":     [None] * bucket_size,
        "selection_mask": [0.0] * bucket_size,
        "grid_idx": -1, "block_idx": -1,
    } for _ in range(num_dummies)]

    all_plain = non_empty_blocks + dummy_blocks
    random.shuffle(all_plain)

    client_blocks = []
    server_blocks = []

    for blk in all_plain:
        enc_coords = []
        for d in range(dim):
            dim_values = [
                p[d] if p is not None else 0.0
                for p in blk["points_norm"]
            ]
            ct = engine.encrypt(engine.encode(dim_values), keypack.public_key)
            engine.to_cpu(ct)           # ✅ in-place, ct 여전히 유효
            enc_coords.append(ct)

        enc_mask = engine.encrypt(
            engine.encode(blk["selection_mask"]), keypack.public_key
        )
        engine.to_cpu(enc_mask)         # ✅ in-place

        ...
        server_blocks.append({
            "enc_coords":         enc_coords,
            "enc_selection_mask": enc_mask,
        })

    print(f"▶ [DO] 블록: 실제={len(non_empty_blocks)}, "
          f"더미={num_dummies}, 총={len(all_plain)} (전부 CPU RAM 보관)")

    return client_blocks, server_blocks
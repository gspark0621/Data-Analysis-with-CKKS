import math
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack


def check_neighbor_closed_interval(
    engine: Engine,
    dist_sq_ct,
    eps_sq: float,
    keypack: KeyPack,
    dimension: int,
    bootstrap_interval: int = 3
):
    relin_key = keypack.relinearization_key
    conj_key = keypack.conjugation_key
    boot_key = keypack.bootstrap_key

    slot_count = engine.slot_count
    margin_val = 0.05

    threshold_pt = engine.encode([eps_sq + margin_val for _ in range(slot_count)])
    x = engine.subtract(dist_sq_ct, threshold_pt)

    max_dist_sq = float(dimension)
    lower_abs = abs(-(eps_sq + margin_val))
    upper_abs = abs(max_dist_sq - (eps_sq + margin_val))
    bound = max(lower_abs, upper_abs) * 1.1

    scale_factor = 1.0 / bound
    scale_pt = engine.encode([scale_factor for _ in range(slot_count)])
    current_x = engine.multiply(x, scale_pt)

    min_initial_val = margin_val * scale_factor

    if min_initial_val <= 0:
        required_depth = 5
    else:
        required_depth = math.ceil(math.log(1.0 / min_initial_val, 1.5)) + 3

    c15_pt = engine.encode([1.5 for _ in range(slot_count)])
    c05_pt = engine.encode([0.5 for _ in range(slot_count)])
    m05_pt = engine.encode([-0.5 for _ in range(slot_count)])

    for i in range(required_depth):
        x_sq = engine.square(current_x, relin_key)
        x_cub = engine.multiply(x_sq, current_x, relin_key)

        term1 = engine.multiply(current_x, c15_pt)
        term2 = engine.multiply(x_cub, c05_pt)
        current_x = engine.subtract(term1, term2)

        if (i + 1) % bootstrap_interval == 0 and (i + 1) != required_depth:
            current_x = engine.intt(current_x)
            current_x = engine.bootstrap(current_x, relin_key, conj_key, boot_key)

    minus_half = engine.multiply(current_x, m05_pt)
    result = engine.add(minus_half, c05_pt)

    result = engine.intt(result)
    return engine.bootstrap(result, relin_key, conj_key, boot_key)


def check_neighbor_closed_interval_heaviside9(
    engine: Engine,
    dist_sq_ct,
    eps_sq: float,
    keypack: KeyPack,
    dimension: int,
    margin_val: float = 0.05,
    safety_factor: float = 1.1,
    bound_shrink: float = 1.0,   # 추가
    do_final_bootstrap: bool = True,
):
    relin_key = keypack.relinearization_key
    conj_key = keypack.conjugation_key
    boot_key = keypack.bootstrap_key
    slot_count = engine.slot_count

    threshold_pt = engine.encode([eps_sq + margin_val for _ in range(slot_count)])
    x = engine.subtract(dist_sq_ct, threshold_pt)

    max_dist_sq = float(dimension)
    lower_abs = abs(-(eps_sq + margin_val))
    upper_abs = abs(max_dist_sq - (eps_sq + margin_val))

    base_bound = max(lower_abs, upper_abs) * safety_factor
    bound = base_bound * bound_shrink   # 공격적으로 줄이기
    if bound <= 0:
        bound = base_bound

    scale_factor = 1.0 / bound
    scale_pt = engine.encode([scale_factor for _ in range(slot_count)])
    t = engine.multiply(x, scale_pt)

    a1 = 2.4609375
    a3 = -5.7421875
    a5 = 9.84375
    a7 = -7.3828125
    a9 = 1.8203125

    a1_pt = engine.encode([a1 for _ in range(slot_count)])
    a3_pt = engine.encode([a3 for _ in range(slot_count)])
    a5_pt = engine.encode([a5 for _ in range(slot_count)])
    a7_pt = engine.encode([a7 for _ in range(slot_count)])
    a9_pt = engine.encode([a9 for _ in range(slot_count)])

    half_pt = engine.encode([0.5 for _ in range(slot_count)])
    minus_half_pt = engine.encode([-0.5 for _ in range(slot_count)])

    t2 = engine.square(t, relin_key)
    t3 = engine.multiply(t2, t, relin_key)
    t4 = engine.square(t2, relin_key)
    t5 = engine.multiply(t4, t, relin_key)
    t7 = engine.multiply(t4, t3, relin_key)
    t9 = engine.multiply(t4, t5, relin_key)

    s1 = engine.multiply(t, a1_pt)
    s3 = engine.multiply(t3, a3_pt)
    s5 = engine.multiply(t5, a5_pt)
    s7 = engine.multiply(t7, a7_pt)
    s9 = engine.multiply(t9, a9_pt)

    sign_like = engine.add(s1, s3)
    sign_like = engine.add(sign_like, s5)
    sign_like = engine.add(sign_like, s7)
    sign_like = engine.add(sign_like, s9)

    step_part = engine.multiply(sign_like, minus_half_pt)
    result = engine.add(step_part, half_pt)

    if do_final_bootstrap:
        result = engine.intt(result)
        result = engine.bootstrap(result, relin_key, conj_key, boot_key)

    return result

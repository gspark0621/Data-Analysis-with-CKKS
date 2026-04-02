# core/server/Normalize.py
import math
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
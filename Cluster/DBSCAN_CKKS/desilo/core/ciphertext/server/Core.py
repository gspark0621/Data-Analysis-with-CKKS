# core/server/Core.py
import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack


def identify_core_points_fhe_converted(
    engine: Engine,
    neighbor_count_ct: Ciphertext,
    min_pts: float,
    N: int,
    keypack: KeyPack,
    bootstrap_interval: int = 3,
    **kwargs
) -> Ciphertext:
    relin_key = keypack.relinearization_key
    conj_key = keypack.conjugation_key
    boot_key = keypack.bootstrap_key

    margin = 0.5
    min_pts_margin = min_pts - margin
    min_pts_pt = engine.encode([min_pts_margin for _ in range(N)])
    x = engine.subtract(neighbor_count_ct, min_pts_pt)

    scale_factor = 1.0 / float(N)
    scale_pt = engine.encode([scale_factor for _ in range(N)])
    current_x = engine.multiply(x, scale_pt)

    required_depth = math.ceil(math.log(N / margin, 1.5)) + 1
    print(f"[Server] Core sign 반복 횟수 (N={N}): {required_depth}")

    c15_pt = engine.encode([1.5 for _ in range(N)])
    c05_pt = engine.encode([0.5 for _ in range(N)])

    for i in range(required_depth):
        x_sq = engine.square(current_x, relin_key)
        x_cub = engine.multiply(x_sq, current_x, relin_key)

        term1 = engine.multiply(current_x, c15_pt)
        term2 = engine.multiply(x_cub, c05_pt)
        current_x = engine.subtract(term1, term2)

        if (i + 1) % bootstrap_interval == 0 and (i + 1) != required_depth:
            current_x = engine.intt(current_x)
            current_x = engine.bootstrap(current_x, relin_key, conj_key, boot_key)

    half_pt = engine.encode([0.5 for _ in range(N)])
    half_x = engine.multiply(current_x, half_pt)
    core_indicator = engine.add(half_x, half_pt)

    core_indicator = engine.intt(core_indicator)
    return engine.bootstrap(core_indicator, relin_key, conj_key, boot_key)
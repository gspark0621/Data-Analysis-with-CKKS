# core/ciphertext_single/Normalize.py

import math
from desilofhe import Engine
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp


def _eval_mcp_fhe(engine, ct, components, slot_count, keypack):
    """
    FHE 상에서 Minimax Composite Polynomial 평가.
    (Core.py 와 동일한 구조, slot_count 기준으로 encode)
    """
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key
    current   = ct

    for step_idx, comp in enumerate(components):
        coeffs = comp["coeffs"]

        x_sq   = engine.square(current, relin_key)
        x_pow  = current
        result = engine.multiply(
            x_pow, engine.encode([coeffs[0]] * slot_count)
        )

        for k in range(1, len(coeffs)):
            x_pow  = engine.multiply(x_pow, x_sq, relin_key)
            result = engine.add(
                result,
                engine.multiply(x_pow, engine.encode([coeffs[k]] * slot_count))
            )

        current = result

        if step_idx < len(components) - 1:
            print(f"  - [Normalize MCP] p_{step_idx+1} 완료, 중간 부트스트래핑...")
            current = engine.intt(current)
            current = engine.bootstrap(current, relin_key, conj_key, boot_key)

    return current


def check_neighbor_closed_interval(
    engine, dist_sq_ct, eps_sq, keypack, dimension,
    bootstrap_interval=3,
    mcp_path="mcp_alpha8.json",
):
    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    boot_key   = keypack.bootstrap_key
    slot_count = engine.slot_count

    components = load_mcp(mcp_path)

    # [핵심 수정 1] margin_val: eps_sq의 5% 수준, 최소 1e-3
    # 기존 0.05는 eps_sq=0.01615 보다 3배 커서 경계를 크게 이동시켰음
    margin_val = max(eps_sq * 0.05, 1e-4)
    print(f"  [Normalize] eps_sq={eps_sq:.6f}, margin_val={margin_val:.6f}, "
          f"threshold={eps_sq + margin_val:.6f}")

    threshold_pt = engine.encode([eps_sq + margin_val] * slot_count)
    x = engine.subtract(dist_sq_ct, threshold_pt)

    # [핵심 수정 2] bound: max_dist_sq를 실제 정규화 범위로 계산
    # 정규화 후 각 좌표 ∈ [0,1] → dist² 최대 = dimension
    # 하지만 threshold shift 후 x의 범위:
    #   x_min = -(eps_sq + margin_val)            (dist²=0 일 때)
    #   x_max = dimension - (eps_sq + margin_val) (dist²=dim 일 때)
    max_dist_sq = float(dimension)  # 정규화 후 이론적 최대 dist²
    x_min_abs   = eps_sq + margin_val
    x_max_abs   = max_dist_sq - (eps_sq + margin_val)
    bound       = max(x_min_abs, x_max_abs) * 1.05  # 5% 여유

    scale_factor = 1.0 / bound
    print(f"  [Normalize] bound={bound:.6f}, scale_factor={scale_factor:.6f}")

    current_x = engine.multiply(x, engine.encode([scale_factor] * slot_count))

    # MCP sign 근사
    current_x = _eval_mcp_fhe(engine, current_x, components, slot_count, keypack)

    # (-sign + 1) / 2: sign=-1(이웃) → 1, sign=+1(비이웃) → 0
    m05_pt = engine.encode([-0.5] * slot_count)
    c05_pt = engine.encode([ 0.5] * slot_count)
    result = engine.add(engine.multiply(current_x, m05_pt), c05_pt)

    result = engine.intt(result)
    return engine.bootstrap(result, relin_key, conj_key, boot_key)


def check_neighbor_closed_interval_heaviside9(
    engine             : Engine,
    dist_sq_ct,
    eps_sq             : float,
    keypack            : KeyPack,
    dimension          : int,
    margin_val         : float = 0.05,
    safety_factor      : float = 1.1,
    bound_shrink       : float = 1.0,
    do_final_bootstrap : bool  = True,
):
    """
    degree-9 단일 Chebyshev 다항식 기반 Heaviside 근사.
    (기존 코드 그대로 유지 — 디버깅/비교용)
    """
    relin_key     = keypack.relinearization_key
    conj_key      = keypack.conjugation_key
    boot_key      = keypack.bootstrap_key
    slot_count    = engine.slot_count

    threshold_pt  = engine.encode([eps_sq + margin_val] * slot_count)
    x             = engine.subtract(dist_sq_ct, threshold_pt)

    max_dist_sq   = float(dimension)
    lower_abs     = abs(-(eps_sq + margin_val))
    upper_abs     = abs(max_dist_sq - (eps_sq + margin_val))
    base_bound    = max(lower_abs, upper_abs) * safety_factor
    bound         = base_bound * bound_shrink if base_bound * bound_shrink > 0 else base_bound
    scale_factor  = 1.0 / bound
    t             = engine.multiply(x, engine.encode([scale_factor] * slot_count))

    a1, a3, a5, a7, a9 = 2.4609375, -5.7421875, 9.84375, -7.3828125, 1.8203125

    t2 = engine.square(t, relin_key)
    t3 = engine.multiply(t2, t, relin_key)
    t4 = engine.square(t2, relin_key)
    t5 = engine.multiply(t4, t, relin_key)
    t7 = engine.multiply(t4, t3, relin_key)
    t9 = engine.multiply(t4, t5, relin_key)

    sign_like = engine.add(engine.multiply(t,  engine.encode([a1] * slot_count)),
                           engine.multiply(t3, engine.encode([a3] * slot_count)))
    sign_like = engine.add(sign_like, engine.multiply(t5, engine.encode([a5] * slot_count)))
    sign_like = engine.add(sign_like, engine.multiply(t7, engine.encode([a7] * slot_count)))
    sign_like = engine.add(sign_like, engine.multiply(t9, engine.encode([a9] * slot_count)))

    result = engine.add(
        engine.multiply(sign_like, engine.encode([-0.5] * slot_count)),
        engine.encode([0.5] * slot_count)
    )

    if do_final_bootstrap:
        result = engine.intt(result)
        result = engine.bootstrap(result, relin_key, conj_key, boot_key)

    return result
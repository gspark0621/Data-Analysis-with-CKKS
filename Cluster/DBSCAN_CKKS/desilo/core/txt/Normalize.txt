# core/ciphertext_single/Normalize.py
#
# 핵심:
#   DesiloFHE lazy-rescaling: 곱셈당 2레벨 소비
#   dep(15)=4 → 8레벨 소비 → bootstrap(level=10) 후 10-8=2 남음
#   sign_bootstrap 최소 입력 level=3 미충족 → RuntimeError 원인
#
#   [수정] _eval_mcp_fhe: 마지막 컴포넌트 평가 *후* regular bootstrap 추가
#          → 반환 level=10 → sign_bootstrap 입력 level=10 ≥ 3 → 성공

import math
from desilofhe import Engine
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp


def _eval_mcp_fhe(engine, ct, components, slot_count, keypack):
    """
    FHE MCP 평가. domain_b 정규화 + 마지막 후 bootstrap 포함.
    모든 step 후 bootstrap → level=10 반환.
    """
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key
    current   = ct

    for step_idx, comp in enumerate(components):
        coeffs   = comp["coeffs"]
        domain_b = comp.get("domain_b", 1.0)

        if abs(domain_b - 1.0) > 1e-9:
            inv_b   = engine.encode([1.0 / domain_b] * slot_count)
            working = engine.multiply(current, inv_b)
        else:
            working = current

        x_sq   = engine.square(working, relin_key)
        x_pow  = working
        result = engine.multiply(x_pow, engine.encode([coeffs[0]] * slot_count))

        for k in range(1, len(coeffs)):
            x_pow  = engine.multiply(x_pow, x_sq, relin_key)
            result = engine.add(
                result,
                engine.multiply(x_pow, engine.encode([coeffs[k]] * slot_count))
            )

        current = result

        # ★ 중간 step AND 마지막 step 모두 bootstrap
        print(f"  - [Normalize MCP] p_{step_idx+1} 완료 "
              f"(domain_b={domain_b:.4f}), bootstrap...")
        current = engine.intt(current)
        current = engine.bootstrap(current, relin_key, conj_key, boot_key)

    return current  # level=10


def check_neighbor_closed_interval(
    engine, dist_sq_ct, eps_sq, keypack, dimension,
    bootstrap_interval=3,
    mcp_path="mcp_normalize_alpha12.json",   # ★ alpha=12: false positive 32%→2.4%
):
    """
    FHE 이웃 판별: dist² ≤ eps² → 1, else → 0.

    sign_bootstrap 활용:
      _eval_mcp_fhe 반환 level=10 → sign_bootstrap(입력 level≥3 충족)
      α=8: delta=2^{-8}=0.00391 < margin/bound≈0.00394 ✓
    """
    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    boot_key   = keypack.bootstrap_key
    slot_count = engine.slot_count

    components = load_mcp(mcp_path)

    # ★ 핵심 수정: MCP 파일에서 실제 delta 읽기 (hardcoded 2^{-8} 제거)
    #   구 코드: mcp_delta=2^{-8} 고정 → mcp_path=alpha12여도 margin=0.012 유지
    #            → eps_eff +32.8% → false positive adj 다수 발생
    #   수정 후: alpha=12 시 delta=2^{-12}=0.000244 → margin=0.00077 → eps_eff +2.4% ✓
    mcp_delta    = components[0]["domain_a"]           # 실제 MCP delta
    max_dist_sq  = float(dimension)
    approx_bound = max_dist_sq * 1.05

    margin_val = mcp_delta * approx_bound              # 최소 필요 margin만 (eps_sq*0.1 항 제거)

    print(f"  [Normalize] dim={dimension}, eps_sq={eps_sq:.6f}, "
          f"approx_bound={approx_bound:.4f}")
    print(f"  [Normalize] margin_val={margin_val:.6f}, threshold={eps_sq+margin_val:.6f}")
    print(f"  [Normalize] eps_effective={math.sqrt(eps_sq+margin_val):.5f} "
          f"(+{(math.sqrt((eps_sq+margin_val)/eps_sq)*100-100):.1f}%)")

    threshold_pt = engine.encode([eps_sq + margin_val] * slot_count)
    x = engine.subtract(dist_sq_ct, threshold_pt)

    x_min_abs    = eps_sq + margin_val
    x_max_abs    = max_dist_sq - (eps_sq + margin_val)
    bound        = max(x_min_abs, x_max_abs) * 1.05
    scale_factor = 1.0 / bound

    print(f"  [Normalize] bound={bound:.6f}, scale={scale_factor:.6f}, "
          f"min|x_scaled|={margin_val*scale_factor:.6f} (δ={mcp_delta:.6f} 대비 "
          f"{margin_val*scale_factor/mcp_delta:.2f}배)")

    current_x = engine.multiply(x, engine.encode([scale_factor] * slot_count))

    # MCP 평가 → 마지막 후 bootstrap → level=10 반환
    current_x = _eval_mcp_fhe(engine, current_x, components, slot_count, keypack)

    # ★ sign_bootstrap: level=10 입력 → level=13 출력
    print(f"  - [Normalize] sign_bootstrap (입력 level=10, 출력 level≈13)...")
    current_x = engine.sign_bootstrap(
        engine.intt(current_x),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )

    # (-sign + 1) / 2: sign=-1(이웃) → 1, sign=+1(비이웃) → 0
    m05_pt = engine.encode([-0.5] * slot_count)
    c05_pt = engine.encode([ 0.5] * slot_count)
    result = engine.add(engine.multiply(current_x, m05_pt), c05_pt)

    result = engine.intt(result)
    return engine.bootstrap(result, relin_key, conj_key, boot_key)
# core/ciphertext_single/Normalize.py
#
# 변경: _eval_mcp_fhe → bsgs_poly.eval_mcp_full (BSGS 기반)
#   α=12 degrees=[15,15,15,15]: dep(15)=4 → 8레벨 소비 → budget=10 ✓
#   naive도 deg=15까지는 작동하지만 BSGS로 통일 (Core/LP와 동일 구조)

import math
from desilofhe import Engine
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.bsgs_poly import eval_mcp_full   # ★ BSGS 공용 모듈


def check_neighbor_closed_interval(
    engine, dist_sq_ct, eps_sq, keypack, dimension,
    bootstrap_interval=3,
    mcp_path="mcp_normalize_alpha12.json",   # α=12: false positive 2.4%
):
    """
    FHE 이웃 판별: dist² ≤ eps² → 1, else → 0.

    α=12 선택 이유:
      margin_val = mcp_delta × approx_bound  (mcp_delta = 2^{-12} = 0.000244)
      eps_effective = eps + 2.4% → false positive adj 비율 2.4% (α=8: 32%)

    Pipeline:
      1. x = (dist² - threshold) / bound  → x ∈ [-1, 1] 정규화
      2. MCP 평가 (BSGS): sign 근사 (각 컴포넌트 후 bootstrap)
      3. sign_bootstrap: level=10 → level≈13
      4. (-sign + 1) / 2 → {0:이웃, 1:비이웃} → {1:이웃, 0:비이웃}
      5. 최종 bootstrap
    """
    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    boot_key   = keypack.bootstrap_key
    slot_count = engine.slot_count

    components = load_mcp(mcp_path)

    # ★ MCP 파일에서 실제 delta 읽기 (hardcoded 제거)
    mcp_delta    = components[0]["domain_a"]
    max_dist_sq  = float(dimension)
    approx_bound = max_dist_sq * 1.05
    margin_val   = mcp_delta * approx_bound

    print(f"  [Normalize] dim={dimension}, eps_sq={eps_sq:.6f}, "
          f"approx_bound={approx_bound:.4f}")
    print(f"  [Normalize] margin_val={margin_val:.6f}, threshold={eps_sq+margin_val:.6f}")
    print(f"  [Normalize] eps_effective={math.sqrt(eps_sq+margin_val):.5f} "
          f"(+{(math.sqrt((eps_sq+margin_val)/eps_sq)*100-100):.1f}%)")

    # x = (dist² - threshold) / bound → x ∈ [-1, 1]
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

    # MCP 평가 (BSGS) → 각 컴포넌트 후 bootstrap → level=10 반환
    current_x = eval_mcp_full(engine, current_x, components, slot_count, keypack,
                               tag="Normalize ")

    # sign_bootstrap: level=10 입력 → level≈13 출력
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
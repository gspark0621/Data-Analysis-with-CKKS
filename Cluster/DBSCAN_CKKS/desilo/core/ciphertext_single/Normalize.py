# core/ciphertext_single/Normalize.py
#
# [2026-05c 작업 A] 마지막 일반 bootstrap → bit_cleaning 교체
#   adj_k가 Server에서 합산되므로 noise 누적이 N(밀도)에 비례 → 빅데이터 ARI 위험.
#   bit_cleaning으로 이웃당 결손 0.00161 → 7.8e-6 (200배↓).
#
# [2026-05 수정] Chebyshev basis BSGS (Bossuat Algorithm 1) + α=15 통일
#
# 변경 이력:
#   이전:    mcp_alpha12.json           (α=12, odd power basis)
#            from bsgs_poly import eval_mcp_full
#   2026-05a: mcp_alpha12_cheb.json     (α=12, odd Chebyshev basis)
#   2026-05b: mcp_alpha15_lp_cheb.json  (α=15, Normalize/Core/LP 공유)
#            from bsgs_chebyshev import eval_mcp_full_chebyshev
#   2026-05c 현재: + bit_cleaning (일반 bootstrap 대체)
#
# α=15 통일 이유 (옵션 B: 전부 α=15):
#   - α=12 worst case 영역 (|scaled_x| < τ = 2.4e-4) 안전성 미확인.
#     sanity check에서 x=4τ는 FAIL, x=33τ는 PASS — 9.7τ(=Core min gap)는 측정 안 됨.
#   - α=15로 올리면 worst case 영역이 τ = 3.05e-5로 8배 좁아져
#     같은 데이터에서 |dist² - eps²|의 최소값이 도메인 깊숙이 위치 → 데이터/eps 의존성 거의 제거.
#   - Normalize/Core가 LP와 동일 JSON 공유 → 로드 1회, 검증 부담 0
#     (LP는 이미 sanity check α=15 PASS 확인됨).
#   - Core bootstrap 1회 추가 비용은 전체 파이프라인의 0.06% 미만 — 무시 가능.
#
# Chebyshev basis 이유 (이전과 동일):
#   - Power basis는 high-degree coefficient 폭발 (T_27 leading ≈ 6.7e7)
#   - CKKS plaintext multiplication 시 noise 증가 → 정밀도 손실
#   - Chebyshev basis로 평가하면 계수 폭발 회피
#   - Bossuat et al. EUROCRYPT 2021 Algorithm 1 정확히 따름

import math
from desilofhe import Engine
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev
from core.ciphertext_single.cleaning import bit_cleaning   # ★ [2026-05c] 작업 A

# ★ [2026-05c 작업 A] cleaning 반복 횟수 (Core와 동일 정책).
_CLEANING_ITERS = 1


def check_neighbor_closed_interval(
    engine, dist_sq_ct, eps_sq, keypack, dimension,
    bootstrap_interval=3,
    mcp_path="mcp_alpha15_lp_cheb.json",   # ★ α=15 통일 (Normalize/Core/LP 공유)
    debug: bool = False,
):
    """
    FHE 이웃 판별: dist² ≤ eps² → 1, else → 0.

    α=15 + Chebyshev BSGS (Normalize/Core/LP α=15 통일):
      mcp_delta    = 2^{-15} ≈ 3.05e-5
      margin η     = 2^{-17} (논문 Table 3 max_depth, LP와 공유 JSON)
      degrees      = [7, 15, 15, 15, 27]
      BSGS depth   = 5 (deg=27, dep×2 = 10 레벨 = budget 10 ✓)
      worst case 안전 영역: α=12 대비 8배 좁아져 데이터/eps 의존성 거의 제거.

    Pipeline:
      1. x = (dist² - threshold) / bound  → x ∈ [-1, 1]
      2. Chebyshev BSGS MCP 평가 (sign 근사)
      3. sign_bootstrap
      4. (-sign + 1) / 2 → {1:이웃, 0:비이웃}
      5. 최종 bootstrap
    """
    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    boot_key   = keypack.bootstrap_key
    slot_count = engine.slot_count

    components = load_mcp(mcp_path)

    # ── basis 확인 (Chebyshev여야 함) ────────────────────────────────
    basis = components[0].get("basis", "power")
    if basis != "chebyshev":
        raise ValueError(
            f"[Normalize] {mcp_path} has basis='{basis}', expected 'chebyshev'. "
            f"JSON 재생성 필요: compute_mcp_for_normalize_chebyshev() 사용."
        )

    mcp_delta    = components[0]["domain_a"]
    max_dist_sq  = float(dimension)
    approx_bound = max_dist_sq * 1.05
    margin_val   = mcp_delta * approx_bound

    print(f"  [Normalize] basis=chebyshev, dim={dimension}, eps_sq={eps_sq:.6f}")
    print(f"  [Normalize] mcp_delta={mcp_delta:.6f} (= 2^{math.log2(mcp_delta):.2f})")
    print(f"  [Normalize] margin_val={margin_val:.6f}, threshold={eps_sq+margin_val:.6f}")
    print(f"  [Normalize] eps_effective={math.sqrt(eps_sq+margin_val):.5f} "
          f"(+{(math.sqrt((eps_sq+margin_val)/eps_sq)*100-100):.1f}%)")

    threshold_pt = engine.encode([eps_sq + margin_val] * slot_count)
    x = engine.subtract(dist_sq_ct, threshold_pt)

    x_min_abs    = eps_sq + margin_val
    x_max_abs    = max_dist_sq - (eps_sq + margin_val)
    bound        = max(x_min_abs, x_max_abs) * 1.05
    scale_factor = 1.0 / bound

    print(f"  [Normalize] bound={bound:.6f}, scale={scale_factor:.6f}")

    current_x = engine.multiply(x, engine.encode([scale_factor] * slot_count))

    # ── Chebyshev BSGS MCP 평가 (Bossuat Algorithm 1) ────────────────
    current_x = eval_mcp_full_chebyshev(
        engine, current_x, components, slot_count, keypack,
        tag="Normalize ", debug=debug,
    )

    # ── sign_bootstrap ────────────────────────────────────────────────
    print(f"  - [Normalize] sign_bootstrap...")
    current_x = engine.sign_bootstrap(
        engine.intt(current_x),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )

    # ── (-sign + 1) / 2 → {1:이웃, 0:비이웃} ─────────────────────────
    m05_pt = engine.encode([-0.5] * slot_count)
    c05_pt = engine.encode([ 0.5] * slot_count)
    result = engine.add(engine.multiply(current_x, m05_pt), c05_pt)

    # ── ★ [2026-05c 작업 A] 마지막 일반 bootstrap → bit_cleaning 교체 ──
    #
    #   [문제] Core와 동일. sign_bootstrap 직후 깨끗한 adj_k를
    #     일반 bootstrap이 noise 주입(0.99839)으로 악화시킴.
    #     adj_k는 Server에서 total_neighbors로 *합산*되므로, 이웃 1개당 0.00161
    #     결손이 점의 이웃수 d에 비례해 누적 (d=500이면 0.8 bias → minPts 경계 오판).
    #     N이 큰 빅데이터에서 ARI 저하 위험.
    #
    #   [해결] {0,1} 도메인 bit_cleaning. 이웃당 결손 0.00161 → 7.8e-6 (200배↓).
    #     d=500이어도 누적 0.0039 → 무시 가능.
    #
    #   [레벨 자동 처리 ★]
    #     bit_cleaning 내부가 ciphertext.level 확인 → 부족 시 _refresh.
    #     (iter 진입 전 level<2, 완료 후 level<3 일 때 자동 bootstrap.)
    #     adj_k는 Server에서 합산(add)/회전(rotate)에 쓰이고(레벨 소비 無),
    #     LP 곱셈 전 _refresh되므로 이중 안전.
    print(f"  - [Normalize] bit_cleaning (n_iters={_CLEANING_ITERS}, 일반 bootstrap 대체)...")
    result = bit_cleaning(
        engine, result, keypack,
        n_iters=_CLEANING_ITERS, slot_count=slot_count,
    )
    return result
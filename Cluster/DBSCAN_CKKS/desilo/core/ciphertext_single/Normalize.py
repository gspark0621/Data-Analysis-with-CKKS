# core/ciphertext_single/Normalize.py
#
# [2026-10] 두 가지 수정
#   (1) 스케일 bound = 1.05·dim 으로 고정 (이전: 1.05·max(threshold, dim−threshold)).
#       eps_n 이 작아 이전 값도 사실상 1.05·dim 이었고(twodiamonds 2.0992 vs 2.1),
#       이제 bound 가 threshold 와 무관해져 margin_val = δ·bound 로 바로 정의된다.
#       보장 deadzone 이 정확히 [eps², eps² + 2·margin_val] 이 된다.
#   (2) 중복 sign_bootstrap 제거. eval_mcp_full_chebyshev 가 마지막 성분 뒤에도
#       sign_bootstrap 을 하므로, 여기서 한 번 더 하던 것은 중복이었다
#       (LP fhe_sgn 은 2026-07 에 같은 이유로 제거). 이제 SB 출력 → 0/1 변환 → bit_cleaning.
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
import os
from desilofhe import Engine
from util.keypack import KeyPack
from core.ciphertext_single.minimax import (
    load_mcp, compute_mcp_with_margin_chebyshev, save_mcp,
)
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev
from core.ciphertext_single.cleaning import bit_cleaning   # ★ [2026-05c] 작업 A

# ★ [2026-05c 작업 A] cleaning 반복 횟수 (Core와 동일 정책).
_CLEANING_ITERS = 1

# ★ [2026-10] sgn 입력 스케일 계수: bound = _NORM_INSET · dim.
#   좌표가 [0,1]^dim 이라 dist² ∈ [0, dim] 이고, threshold ≤ dim 이면
#   |x| = |dist² − threshold| / bound ≤ 1/_NORM_INSET (= 0.952).
#   역할: 정의역 끝 ±1 에서 떼어 놓기. MCP 는 x 가 1 을 ~1e-6 만 넘어도
#   부호가 뒤집힌다(평문 계산: α=16 1.7e-6, α=18 4.3e-7) → 먼 쌍이 이웃으로 세어짐.
#   margin_val = δ·bound 이므로 dist² = eps² 인 진짜 이웃은 정확히 x = −δ
#   (논문 보장 구간 [δ,1] 이 닫힌구간이라 FN=0 보장 성립).
_NORM_INSET = 1.05


# ── Normalize α 자동선택 (N 기반, margin 은 그대로 유지) ─────────────────
#   [margin 유지] threshold = eps² + margin_val (한쪽) 그대로.  eps 안쪽 진짜
#     이웃은 데드존 밖 → adj=1 확실 → false negative 0 (놓치는 이웃 없음).
#     이 로직은 margin 을 건드리지 않고 데드존(δ=2^-α)만 좁혀, eps 바깥
#     false positive 밴드(eps², eps_eff²)를 얇게 만들어 과대계수를 없앤다.
#   [N 기반] margin 과대계수 ≈ (표본밀도)×2^-α, 정규화 데이터에선 밀도 ∝ N
#     → α = clamp(⌈log₂N⌉ + C, 15, 20).  Core/LP 와 같은 스케일링.
#   [C=8] Normalize 는 '연속' dist² 를 eps² 와 비교 → 최소 gap 이 데이터
#     경계밀도 의존(양자화 아님) → 정확 공식 불가.  밀도경계(twodiamonds)에서
#     과검출 0 이 되는 최소가 α=18 → C=18-⌈log₂800⌉=8 (보수적 상수).
_NORM_ALPHA_SAFETY = 8
_NORM_ALPHA_MIN    = 15
_NORM_ALPHA_MAX    = 20
_NORM_DEGREES = {   # ★ [2026-09 정정] 논문 Lee et al. Table 2 "minimize depth" 컬럼 그대로.
    #   [정정 전 상태] 16:[15,15,15,15,31] / 17,18:[7,15,15,15,15,31] / 19:[15]*5+[31]
    #     / 20:[7,15,15,15,15,31,31] — 논문과 불일치했고 17,18 은 같은 세트(복붙 흔적).
    #   [영향] 우리 eval_mcp 는 **컴포넌트마다 sign_bootstrap** 하므로 비용 = 컴포넌트 수.
    #     정정으로 α=17~20 에서 컴포넌트가 6,6,6,7 → 5,5,5,5 로 줄어든다.
    #   [레벨] 최대 degree 31 (dep=5 → 10 레벨 = budget 10 ✓). 29 도 dep=5.
    15: [7, 15, 15, 15, 27],
    16: [15, 15, 15, 15, 27],
    17: [15, 15, 15, 29, 29],
    18: [15, 15, 29, 29, 31],
    19: [15, 29, 31, 31, 31],
    20: [29, 31, 31, 31, 31],
}


def _norm_alpha(N: int) -> int:
    """Normalize α = clamp(⌈log₂N⌉ + C, 15, 20).  margin 은 별도로 유지됨."""
    a = math.ceil(math.log2(max(int(N), 2))) + _NORM_ALPHA_SAFETY
    return max(_NORM_ALPHA_MIN, min(_NORM_ALPHA_MAX, a))


def _ensure_norm_mcp(N: int, out_dir: str = ".") -> str:
    """N 의 α 에 맞는 Normalize MCP 를 없으면 생성하고 경로 반환 (Core/LP _ensure 와 동일 패턴)."""
    a    = _norm_alpha(N)
    path = os.path.join(out_dir, f"mcp_alpha{a}_norm_cheb.json")
    if not os.path.exists(path):
        degs  = _NORM_DEGREES[a]
        comps = compute_mcp_with_margin_chebyshev(degs, 2.0 ** -a, 2.0 ** -(a + 2), a, verbose=False)
        save_mcp(comps, path)
        print(f"[Normalize] N={N} → α={a} MCP 생성 (degrees={degs}, margin 유지)")
    return path


def check_neighbor_closed_interval(
    engine, dist_sq_ct, eps_sq, keypack, dimension,
    num_points=None,                       # ★ N: α 자동선택용 (⌈log₂N⌉+8)
    mcp_path=None,                         # ★ 명시하면 override; None 이면 N 기반
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
      1. x = (dist² - threshold) / bound,  bound = 1.05·dim  → |x| ≤ 1/1.05
      2. Chebyshev BSGS MCP 평가 (sign 근사, 성분마다 sign_bootstrap — 마지막 성분 포함)
      3. (-sign + 1) / 2 → {1:이웃, 0:비이웃}
      4. bit_cleaning 1회
    """
    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    # ★ [2026-07] 제거: 이 파일은 표준 bootstrap 을 직접 호출하지 않는다.
    #   (sign_bootstrap → smallbootstrap_key, bit_cleaning 내부 _refresh 는
    #    cleaning.py 가 처리). full bootstrap_key 는 생성되지 않으므로
    #    이 할당을 남겨두면 None 이 되어 혼란만 준다.
    slot_count = engine.slot_count

    # ★ N 기반 α 자동선택 (margin 은 아래 threshold 계산에서 그대로 유지)
    if mcp_path is None:
        if num_points is None:
            raise ValueError("[Normalize] num_points 또는 mcp_path 중 하나는 필요합니다.")
        mcp_path = _ensure_norm_mcp(int(num_points))
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
    # ★ [2026-10] bound = 1.05·dim (데이터 무관 상수). margin_val = δ·bound.
    #   이전의 approx_bound(=1.05·dim)와 같은 값이라 margin_val 은 변하지 않는다.
    bound        = _NORM_INSET * max_dist_sq
    margin_val   = mcp_delta * bound

    print(f"  [Normalize] basis=chebyshev, dim={dimension}, eps_sq={eps_sq:.6f}")
    print(f"  [Normalize] mcp_delta={mcp_delta:.6f} (= 2^{math.log2(mcp_delta):.2f})")
    print(f"  [Normalize] margin_val={margin_val:.6f}, threshold={eps_sq+margin_val:.6f}")
    print(f"  [Normalize] eps_effective={math.sqrt(eps_sq+margin_val):.5f} "
          f"(+{(math.sqrt((eps_sq+margin_val)/eps_sq)*100-100):.1f}%)")

    threshold_pt = engine.encode([eps_sq + margin_val] * slot_count)
    x = engine.subtract(dist_sq_ct, threshold_pt)

    # ★ [2026-10] 이전: bound = 1.05·max(threshold, dim − threshold).
    #   threshold > dim 이면 x 가 −1/1.05 아래로 내려갈 수 있다(eps 가 상자 대각선보다
    #   큰 퇴화 경우 — 모든 쌍이 이웃). 그런 입력은 정의역 밖이므로 경고만 남긴다.
    if eps_sq + margin_val > max_dist_sq:
        print(f"  [Normalize] ⚠ threshold={eps_sq+margin_val:.4f} > dim={max_dist_sq} "
              f"— eps 가 정규화 상자 대각선보다 큼. x 가 정의역 [−1,1] 밖으로 나갈 수 있음.")
    scale_factor = 1.0 / bound

    print(f"  [Normalize] bound={bound:.6f} (= {_NORM_INSET}·dim), scale={scale_factor:.6f}")

    current_x = engine.multiply(x, engine.encode([scale_factor] * slot_count))

    # ── Chebyshev BSGS MCP 평가 (Bossuat Algorithm 1) ────────────────
    #   ★ eval_mcp_full_chebyshev 는 마지막 성분 뒤에도 sign_bootstrap 을 수행한다
    #     (chebyshev_eval.py: 성분 루프마다 SB). 출력은 레벨이 복구된 ±1 값이고
    #     intt 없이 곧바로 곱셈할 수 있다(LP fhe_sgn 과 같은 사용법).
    current_x = eval_mcp_full_chebyshev(
        engine, current_x, components, slot_count, keypack,
        tag="Normalize ", debug=debug,
    )

    # ── ★ [2026-10] 중복 sign_bootstrap 제거 ─────────────────────────
    #   이전에는 여기서 engine.sign_bootstrap(engine.intt(current_x), ...) 을 한 번 더
    #   호출했다. 바로 앞 SB 출력에 SB 를 다시 거는 중복이었고, 그 정리(±1 근처 오차
    #   제곱 감소)는 아래 bit_cleaning 이 0/1 도메인에서 같은 차수로 해 준다.
    #   효과: adj_k 1개당 sign_bootstrap 1회 감소 (Normalize 는 k_max 번 호출).
    #   평문 계산: 보장 deadzone(|x|<δ) 밖 출력은 그대로이고, 실효 경계만
    #   α=16 0.045δ→0.069δ, α=18 0.043δ→0.067δ 로 약간 넓어진다.

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
    #     SB 직후 레벨은 다음 MCP 성분(최대 degree 31, 10레벨)을 돌릴 만큼 복구되므로
    #     0/1 변환 + cleaning 1회 뒤에도 3 이상이 남아 _refresh 는 발동하지 않는다
    #     (레벨 예산으로 추론. Server_main 의 "[진단] Normalize 구간 표준 bootstrap
    #      호출 수" 가 0 이면 확인된 것).
    #     adj_k는 Server에서 합산(add)/회전(rotate)에 쓰이고(레벨 소비 無),
    #     LP 곱셈 전 _refresh되므로 이중 안전.
    print(f"  - [Normalize] bit_cleaning (n_iters={_CLEANING_ITERS}, 일반 bootstrap 대체)...")
    result = bit_cleaning(
        engine, result, keypack,
        n_iters=_CLEANING_ITERS, slot_count=slot_count,
    )
    return result
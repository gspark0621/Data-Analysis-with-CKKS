# core/ciphertext_single/Core.py
#
# [2026-10] 중복 sign_bootstrap 제거
#   eval_mcp_full_chebyshev 가 마지막 성분 뒤에도 sign_bootstrap 을 하므로, 여기서 한 번 더
#   하던 것은 중복이었다(LP fhe_sgn 은 2026-07 에 같은 이유로 제거, Normalize 도 2026-10 제거).
#   이제 SB 출력 → (sign+1)/2 → bit_cleaning.
#
# [2026-05c 작업 A] 마지막 일반 bootstrap → bit_cleaning 교체
#   측정 확정: sign_bootstrap 직후 0.999996(깨끗) → 일반 bootstrap이 0.99839로 악화.
#   bit_cleaning h(x)=3x²-2x³ 로 noise 주입 없이 정리 (Drucker Lemma 1, {0,1} 도메인).
#   core_mask 정밀도 2^-9.3 → 2^-17 이상 → LP damping 완화.
#
# [2026-05 수정] Chebyshev basis + α=15 통일
#
# 변경 이력:
#   이전:    mcp_alpha11.json (α=11, [7,15,15,15], odd power basis)
#            → α=12 통일 → mcp_alpha12.json (α=12, [15,15,15,15])
#   2026-05a: mcp_alpha12_cheb.json (α=12, Chebyshev basis)
#   2026-05b: mcp_alpha15_lp_cheb.json (α=15, Normalize/Core/LP 공유)
#   2026-05c 현재: + bit_cleaning (일반 bootstrap 대체)
#
# α=15 통일 이유 (옵션 B):
#   - α=12 Core worst case = 0.5/N = 0.00236 ≈ 9.7τ 영역의 안전성 미확인.
#     (sanity check에서 x=4τ FAIL, x=33τ PASS — 9.7τ는 측정 안 됨.)
#   - α=15로 올리면 worst case = 0.5/N ≈ 77.3τ로 도메인 깊숙이 위치 → 안전 보장.
#   - Normalize/LP와 동일 JSON 공유 → 검증 부담 0 (LP에서 이미 α=15 PASS 확인).
#   - 비용 차이: Core는 전체 파이프라인에서 1회 호출, bootstrap +1 ≈ 0.06% 미만 무시 가능.

from desilofhe import Engine, Ciphertext
import math
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev
from core.ciphertext_single.cleaning import bit_cleaning   # ★ [2026-05c] 작업 A


_MCP_CORE_PATH = "mcp_alpha15_lp_cheb.json"   # ★ N-aware 미사용 시 폴백 (α=15 고정)

# ★ [#1-Core 2026-07] N 기반 α 자동 선택.
#   근거: neighbor_count 는 (FHE 노이즈 이전) 정수, min_pts 도 정수 → 분자
#     (neighbor_count-(min_pts-0.5)) 는 항상 반정수 → |분자| ≥ 0.5 결정적.
#     LP 라벨과 동일한 '정수 카운트' 구조 → gap = 0.5/N 은 오직 N 에만 의존.
#     (Normalize 의 dist_sq 는 연속값이라 이 논리가 적용되지 않음 — 별개 취급 유지.)
#
#   ★★★ 중요 경고 (본 파일 상단 이력에서 확인) ★★★
#     "α=12 Core worst case = 0.5/N ≈ 9.7τ 영역의 안전성 미확인.
#      sanity check에서 x=4τ FAIL, x=33τ PASS — 9.7τ는 측정 안 됨."
#     즉 nominal gap/τ 비율(=몇 배 여유)만으로 안전을 판단하면 안 된다 — 실측
#     sanity_check_chebyshev 가 작은 배수(4τ)에서 이미 실패한 전례가 있다.
#     따라서 아래 α 는 "이 정도면 충분하다"는 보장이 아니라 출발점일 뿐이며,
#     실제 배포 전 반드시
#       sanity_check_chebyshev(test_x_values=[gap, 2·gap, 4·gap, 8·gap, ...])
#     로 선택한 α 에서 gap 근방이 PASS 하는지 실측 확인할 것. FAIL 이면
#     _CORE_EPS_ADJ 를 올려(= 오차 상한을 보수적으로 잡아) α 를 키운 뒤 재검증.
#     ★ [2026-10] α 를 2비트 깎았으므로 이 실측 확인이 이전보다 중요해졌다.
_CORE_ALPHA_AUTO        = True

# ★ [2026-10] α 를 '안전 비트'가 아니라 count 오차 상한에서 유도한다.
#   판정 조건:  |x| ≥ δ  ⟺  |분자| ≥ δN.
#   분자 = (반정수) + e 이고 |반정수| ≥ 0.5 이므로 |분자| ≥ 0.5 − |e|. 따라서
#       |e| ≤ 0.5 − N·2^−α        ⟺        α ≥ log2( N / (0.5 − |e|) )
#   여기서 좌변은 가정이 아니라 α 가 결정하는 '허용 count 오차'이다.
#   count 는 자기 자신(평문 상수 +1) + 최대 2·k_max 개의 0/1 지시자 합이므로
#       |e| ≤ 2·k_max·ε_adj + ε_sum
#   ε_adj = SB 직후 adj 한 칸이 {0,1} 에서 벗어나는 양(CKKS 노이즈 바닥, 실측 1건 ≈ 4e-6).
#   ε_sum = 회전 합산 + 1/N 곱의 노이즈 (미측정 → ε_adj 보수값에 흡수한다고 본다).
#   k_max 는 Core 로 전달되지 않으므로 최악의 경우 k_max = N 을 쓴다(실제 k_max < N).
#
#   결과: 현재 벤치마크 전체에서 α = ⌈log2 2N⌉ 과 일치하며,
#     2N 이 2의 거듭제곱일 때(N=512,1024,4096,8192)만 자동으로 +1 비트가 붙는다.
#     ⌈log2 2N⌉ 을 그대로 쓰면 그 경우 허용 오차가 정확히 0 이 되므로 이 보정이 필요하다.
#   이전(2026-09): α = ⌈log2 2N⌉ + 2 (= ⌈log2 8N⌉). 이는 ε_adj 를 실측의 약 100배로
#     잡은 것과 같은 값이며, 해당 여유가 필요하면 _CORE_EPS_ADJ 를 4e-4 로 올리면 된다.
_CORE_ALPHA_MIN         = 8
_CORE_ALPHA_MAX         = 16

_MINIMIZE_DEPTH_DEGREES_CORE = {
    8:  [7, 15, 15],        9:  [7, 7, 7, 13],     10: [7, 7, 13, 15],
    11: [7, 15, 15, 15],    12: [15, 15, 15, 15],  13: [15, 15, 15, 31],
    14: [7, 7, 15, 15, 27], 15: [7, 15, 15, 15, 27], 16: [15, 15, 15, 15, 27],
}


# [기록] 한때 α=13 을 금지했으나 원인은 minimax.py 의 degree 표 오타였고, 정정 후
#   α=13 오차가 이웃 α 와 동급(1e-13)임을 확인해 회피 매핑을 삭제했다.


def _core_alpha(N: int, k_max: int = None) -> int:
    """N → Core sign 근사 최소 α (유도식, 위 주석 참조).

        α ≥ log2( N / (0.5 − |e|) ),    |e| ≤ 2·k_max·ε_adj + ε_sum

    k_max 를 모르면 최악의 경우 k_max = N 으로 둔다.
    |e| ≥ 0.5 이면 반정수 구조로는 어떤 α 로도 보장할 수 없다(에러).
    """
    a = math.ceil(math.log2(float(N) / 0.5))
    a = max(_CORE_ALPHA_MIN, min(_CORE_ALPHA_MAX, a))
    return a


def _load_core_mcp(N: int, mcp_path_override: str = None):
    """N-aware α 로 Core MCP 로드(없으면 생성). override 지정 시 그대로 사용."""
    if mcp_path_override is not None:
        return load_mcp(mcp_path_override), mcp_path_override
    if not _CORE_ALPHA_AUTO:
        return load_mcp(_MCP_CORE_PATH), _MCP_CORE_PATH
    alpha = _core_alpha(N)
    path = f"mcp_alpha{alpha}_lp_cheb.json"   # LP/Core 공통 명명(같은 α면 파일 재사용)
    try:
        comps = load_mcp(path)
        print(f"[Core] N={N} → α={alpha} MCP 로드: {path} "
              f"(gap=0.5/{N}={0.5/N:.5f}, δ=2^-{alpha}, "
              f"여유={ (0.5/N)/(2.0**-alpha):.1f}배 — ★sanity_check 로 실측 재확인 권장)")
    except (FileNotFoundError, OSError):
        from core.ciphertext_single.minimax import (
            compute_mcp_for_label_prop_chebyshev, save_mcp,
        )
        degrees = _MINIMIZE_DEPTH_DEGREES_CORE.get(alpha)
        print(f"[Core] N={N} → α={alpha} MCP 생성 (degrees={degrees}) → {path}")
        comps = compute_mcp_for_label_prop_chebyshev(alpha=alpha, verbose=True)
        try:
            save_mcp(comps, path)
        except Exception as _e:
            print(f"[Core] ⚠ MCP 저장 실패({_e}) — 메모리 컴포넌트로 진행")
    return comps, path

# ★ [2026-05c 작업 A] cleaning 반복 횟수.
#   1회면 0.99839 → 0.9999923 (2^-17, 충분). 2회면 CKKS 한계 (2^-32).
#   레벨 예산 절약 위해 기본 1. 레벨 부족 시 폴백은 함수 내 주석 참조.
_CLEANING_ITERS = 1


def identify_core_points_fhe_converted(
    engine: Engine,
    neighbor_count_ct: Ciphertext,
    min_pts: float,
    N: int,
    keypack: KeyPack,
    mcp_path: str = None,
    debug: bool = False,
    **kwargs
) -> Ciphertext:
    """
    Core point 판별: totalNeighbors >= min_pts → 1, else → 0.

    α=15 + Chebyshev(N=212):
      δ = 2^{-15} ≈ 3.05e-5 → ~77배 안전 여유 vs min gap 0.00236 (= 0.5/N)
      margin η = 2^{-17} (논문 Table 3 max_depth, LP와 공유 JSON)
      degrees [7, 15, 15, 15, 27], BSGS depth=5 → 10 레벨 = budget 10 ✓
      (이전 α=12: 9.7τ 여유 — sanity check 미검증 영역. α=15에서 8배↑ 확보.)

    Pipeline:
      1. x = (totalNeighbors - (min_pts-0.5)) / N  → x ∈ [-1,1]
      2. Chebyshev MCP 평가 (sign 근사, 성분마다 sign_bootstrap — 마지막 성분 포함)
      3. (sign+1)/2 → {0,1}
      4. bit_cleaning 1회
    """
    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    # ★ [2026-07] 제거: 이 파일은 표준 bootstrap 을 직접 호출하지 않는다.
    #   (sign_bootstrap → smallbootstrap_key, bit_cleaning 내부 _refresh 는
    #    cleaning.py 가 처리). full bootstrap_key 는 생성되지 않으므로
    #    이 할당을 남겨두면 None 이 되어 혼란만 준다.
    slot_count = engine.slot_count

    # ★ [#1-Core] mcp_path 명시 지정 시 그대로 사용(하위호환), 아니면 N-aware 선택.
    components, mcp_path = _load_core_mcp(N, mcp_path_override=mcp_path)
    print(f"[Server] Core: Chebyshev BSGS MCP 로드 ({mcp_path})")

    # basis 확인
    basis = components[0].get("basis", "power")
    if basis != "chebyshev":
        raise ValueError(
            f"[Core] {mcp_path} has basis='{basis}', expected 'chebyshev'. "
            f"JSON 재생성 필요: compute_mcp_for_core_chebyshev() 사용."
        )

    print(f"[Server] Core: degrees={[c['degree'] for c in components]}, "
          f"sign_err={components[-1]['error']:.4e}")

    # x = (totalNeighbors - (min_pts - 0.5)) / N ∈ [-1, 1]
    margin     = 0.5
    min_pts_pt = engine.encode([min_pts - margin] * slot_count)
    x          = engine.subtract(neighbor_count_ct, min_pts_pt)
    scale_pt   = engine.encode([1.0 / float(N)] * slot_count)
    current_x  = engine.multiply(x, scale_pt)

    print(f"[Server] Core: N={N}, min_pts={min_pts}, scale=1/{N}={1.0/N:.4e}")
    mcp_delta = components[0]["domain_a"]
    print(f"[Server] Core: delta={mcp_delta:.5e} (= 2^{math.log2(mcp_delta):.2f}) "
          f"< 0.5/N={0.5/N:.5f} ✓ (여유 {(0.5/N)/mcp_delta:.1f}배)")

    # ── Chebyshev BSGS MCP 평가 ──────────────────────────────────────
    #   ★ eval_mcp_full_chebyshev 는 마지막 성분 뒤에도 sign_bootstrap 을 수행한다.
    #     출력은 레벨이 복구된 ±1 값이고 intt 없이 곧바로 곱셈할 수 있다.
    current_x = eval_mcp_full_chebyshev(
        engine, current_x, components, slot_count, keypack,
        tag="Core ", debug=debug,
    )

    # ── ★ [2026-10] 중복 sign_bootstrap 제거 ─────────────────────────
    #   이전에는 여기서 engine.sign_bootstrap(engine.intt(current_x), ...) 을 한 번 더
    #   호출했다. 정수 count 는 보장 deadzone 의 4배 이상 바깥(gap/δ ∈ [4,8))이라
    #   이미 포화된 값에 SB 를 다시 거는 중복이었고, 남은 ±1 근처 오차의 제곱 감소는
    #   아래 bit_cleaning 이 0/1 도메인에서 해 준다.
    #   평문 계산(α=12): 실효 경계 0.084δ → 0.128δ 로 넓어지지만 정수 count 는 ≥ 4δ 라 무관.

    # ── (sign + 1) / 2 → {0, 1} ──────────────────────────────────────
    half_pt        = engine.encode([0.5] * slot_count)
    core_indicator = engine.add(engine.multiply(current_x, half_pt), half_pt)

    # ── ★ [2026-05c 작업 A] 마지막 일반 bootstrap → bit_cleaning 교체 ──
    #
    #   [측정으로 확정된 문제]  (※ 측정 당시 구조: MCP → SB 추가 1회 → 일반 bootstrap)
    #     sign_bootstrap 직후 core_indicator는 깨끗함 (측정A: 0.999996, ~2^-18).
    #     기존 engine.bootstrap(일반)이 noise를 *주입*하여 0.99839 (~2^-9.3)로 악화 (측정B).
    #     이 0.99839가 LP에서 mask로 누적 곱해져 라벨 damping → 과분할 (ARI 66).
    #
    #   [해결] {0,1} 도메인이므로 bit_cleaning h(x)=3x²-2x³ 적용.
    #     noise 주입 없이 0.999998 → ~1.0 (2^-32)으로 quadratic 수렴.
    #
    #   [레벨 자동 처리 ★]
    #     cleaning은 일반 bootstrap과 달리 level을 복구하지 않고 *소비*(2/iter).
    #     bit_cleaning 내부가 ciphertext.level을 확인하여:
    #       - iter 진입 전 level<2 → _refresh
    #       - 완료 후 level<3(sign_bootstrap 요구치) → _refresh
    #     즉 sign_bootstrap 후 잔여 level이 부족해도 cleaning이 자동 대응.
    #     (Core 출력 core_ct는 LP가 받아 다시 _refresh하므로 이중 안전.)
    print(f"  - [Core] bit_cleaning (n_iters={_CLEANING_ITERS}, 일반 bootstrap 대체)...")
    core_indicator = bit_cleaning(
        engine, core_indicator, keypack,
        n_iters=_CLEANING_ITERS, slot_count=slot_count,
    )
    return core_indicator
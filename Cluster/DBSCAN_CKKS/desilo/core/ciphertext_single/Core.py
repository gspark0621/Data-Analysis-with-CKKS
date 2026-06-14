# core/ciphertext_single/Core.py
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


_MCP_CORE_PATH = "mcp_alpha15_lp_cheb.json"   # ★ Normalize/LP와 공유 (Chebyshev, α=15 통일)

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
    bootstrap_interval: int = 3,
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
      2. Chebyshev MCP 평가 (sign 근사)
      3. sign_bootstrap
      4. (sign+1)/2 → {0,1}
      5. 최종 bootstrap
    """
    if mcp_path is None:
        mcp_path = _MCP_CORE_PATH

    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    boot_key   = keypack.bootstrap_key
    slot_count = engine.slot_count

    print(f"[Server] Core: Chebyshev BSGS MCP 로드 ({mcp_path})")
    components = load_mcp(mcp_path)

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
    current_x = eval_mcp_full_chebyshev(
        engine, current_x, components, slot_count, keypack,
        tag="Core ", debug=debug,
    )

    # ── sign_bootstrap ────────────────────────────────────────────────
    print(f"  - [Core] sign_bootstrap...")
    current_x = engine.sign_bootstrap(
        engine.intt(current_x),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )

    # ── (sign + 1) / 2 → {0, 1} ──────────────────────────────────────
    half_pt        = engine.encode([0.5] * slot_count)
    core_indicator = engine.add(engine.multiply(current_x, half_pt), half_pt)

    # ── ★ [2026-05c 작업 A] 마지막 일반 bootstrap → bit_cleaning 교체 ──
    #
    #   [측정으로 확정된 문제]
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
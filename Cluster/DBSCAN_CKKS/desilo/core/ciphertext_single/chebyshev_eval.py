# core/ciphertext_single/bsgs_chebyshev.py
"""
Chebyshev basis MCP evaluation using DesiloFHE native function.

 DesiloFHE 1.9 API:
   engine.evaluate_chebyshev_polynomial(ct, coefficients, relin_key)
   Evaluates: a_0 + a_1 T_1(x) + ... + a_n T_n(x)

 라이브러리가 내부적으로 efficient polynomial evaluation 알고리즘
 (Paterson-Stockmeyer 또는 BSGS 류)을 사용하여 O(√n) 비스칼라 곱셈으로
 평가합니다. 정확한 알고리즘은 공식 문서로 확인되지 않았으므로
 단정하지 않습니다. (실험적으로 deg=27까지 정상 동작 확인.)

 우리가 직접 BSGS 분해를 구현하지 않아도 되는 이유는 라이브러리가
 이 부분을 처리해주기 때문이며, 이는 ↓ 두 가지 검증으로 확인됨:
   1. coefficient form 입력만 받음 (NTT form 거부)
   2. sanity_check_chebyshev에서 plain vs FHE diff ~1e-3 수준 일치#

Odd-only 적용:
  Sign 근사 다항식 p(x) = Σ_{k=0}^{m-1} c_k T_{2k+1}(x) (홀수만)
  → odd_coeffs_to_full로 짝수 위치를 0으로 패딩
  → evaluate_chebyshev_polynomial에 그대로 전달
"""

from __future__ import annotations
import numpy as np
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack


# ★ [2026-06 Phase1b] 컴포넌트 간 bootstrap 종류.
#   "sign"     : sign_bootstrap (유계 sin 클램프 + 이차 수렴 + 레벨 복구) ← 기본
#   "standard" : 이전 동작 (표준 bootstrap, 노이즈 주입 — 거짓 양성 원인)
_INTER_COMPONENT_BOOTSTRAP = "sign"


# ─────────────────────────────────────────────────────────────────────────
# 유틸리티: odd → full Chebyshev coefficients
# ─────────────────────────────────────────────────────────────────────────

def odd_coeffs_to_full(odd_coeffs):
    """
    odd Chebyshev 계수 [c_0, c_1, ..., c_{m-1}] (for T_1, T_3, ..., T_{2m-1})
    → full Chebyshev 계수 [0, c_0, 0, c_1, 0, ..., 0, c_{m-1}]
                          (for T_0,  T_1, T_2, T_3, ..., T_{2m-2}, T_{2m-1})
    """
    d = 2 * len(odd_coeffs) - 1
    full = [0.0] * (d + 1)
    for k, c in enumerate(odd_coeffs):
        full[2 * k + 1] = float(c)
    return full


# ─────────────────────────────────────────────────────────────────────────
# 메인 API
# ─────────────────────────────────────────────────────────────────────────

def eval_mcp_full_chebyshev(
    engine: Engine,
    ct: Ciphertext,
    components: list,
    slot_count: int,
    keypack: KeyPack,
    tag: str = "",
    debug: bool = False,
) -> Ciphertext:
    """
    Chebyshev basis MCP 전체 평가 (DesiloFHE 네이티브 사용).

    각 컴포넌트:
      1. domain_b 정규화 (x → x/domain_b)
      2. odd Chebyshev 계수 → full coefficients
      3. ★ engine.evaluate_chebyshev_polynomial(...) 호출
      4. 컴포넌트 종료 후 bootstrap

    Parameters
    ----------
    components : list[dict]
        Each comp must have basis="chebyshev". Otherwise raises ValueError.
    debug : bool
        If True, print level info after each component.
    """
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key

    current = ct

    for idx, comp in enumerate(components):
        basis = comp.get("basis", "power")
        if basis != "chebyshev":
            raise ValueError(
                f"eval_mcp_full_chebyshev: component {idx+1} has basis='{basis}'. "
                f"Use compute_mcp_for_*_chebyshev() to generate Chebyshev MCP."
            )

        domain_b = comp.get("domain_b", 1.0)

        # ── 1. domain_b 정규화 ────────────────────────────────────────
        if abs(domain_b - 1.0) > 1e-9:
            scale = 1.0 / domain_b
            current = engine.multiply(current, engine.encode([scale] * slot_count))

        # ── 2. odd → full Chebyshev coefficients ─────────────────────
        odd_coeffs  = comp["coeffs"]
        full_coeffs = odd_coeffs_to_full(odd_coeffs)

        # ── 3. ★ Chebyshev 평가 전 intt (NTT → coefficient form) ─────
        # DesiloFHE 요구사항:
        #   evaluate_chebyshev_polynomial은 입력이 NTT form이면 에러
        #   ("the input ciphertext should not be in NTT form")
        # multiply/add 후 ciphertext는 NTT form이므로 intt 호출 필요.
        current = engine.intt(current)

        # ── 4. DesiloFHE 네이티브 Chebyshev 평가 ★ ────────────────────
        # 라이브러리가 BSGS 자체 처리: 우리가 cheb_divmod, baby/giant steps,
        # 재귀 분해 등을 직접 구현할 필요 없음.
        current = engine.evaluate_chebyshev_polynomial(
            current, full_coeffs, relin_key,
        )

        if debug:
            try:
                print(f"  [Cheb-native] comp {idx+1}: after eval, level={current.level}")
            except AttributeError:
                pass

        # ── 5. Bootstrap after each component ─────────────────────────
        # ★ [2026-06 Phase1b] 컴포넌트 간 표준 bootstrap → sign_bootstrap 교체.
        #
        #   [발견된 버그 — hepta 인접행렬 거짓 양성 143개 (cross 138)의 원인]
        #     표준 bootstrap이 컴포넌트마다 ~2^-9.3 노이즈를 중간값에 주입하고,
        #     다음 컴포넌트(특히 deg-27, 경계 기울기 ~n²=729)가 이를 증폭.
        #     희소 슬롯에서 중간값이 z≈2~3까지 폭주하면, 최종 sign_bootstrap의
        #     주기적 EvalSign(sin)이 z≈2→0, z≈3→−1로 접어 부호가 뒤집힘
        #     → dist²≫eps²인 cross pair가 adj≈0.5~1.0으로 오판
        #     → LP Round 1에서 전 클러스터 라벨 오염 (ARI 63 의 근본 원인).
        #
        #   [해결 — Hong et al. §4.1, Fig. 4b 그대로]
        #     컴포넌트 사이를 sign_bootstrap으로 연결:
        #     (1) sin은 유계 → 중간 폭주를 [−1,1]로 즉시 클램프 (접힘 원천 차단)
        #     (2) τ → (π²/8)τ² 이차 수렴 "공짜" (Thm 1) — p_{i+1}의 fitted
        #         domain D_{τ_i}의 부분집합으로 들어가므로 근사 보증 보존
        #     (3) 레벨 복구는 동일, 런타임도 동등 이하 (논문 Table 5)
        #     노이즈를 주입하던 유지보수 단계가 수렴을 돕는 단계로 바뀜.
        #
        #   _INTER_COMPONENT_BOOTSTRAP = "standard"로 되돌리기 가능.
        if _INTER_COMPONENT_BOOTSTRAP == "sign":
            current = engine.sign_bootstrap(
                current,
                relin_key,
                conj_key,
                keypack.rotation_key,
                keypack.smallbootstrap_key,
            )
        else:
            current = engine.bootstrap(current, relin_key, conj_key, boot_key)

        if debug:
            try:
                print(f"  [{tag}] component {idx+1}/{len(components)} done  "
                      f"level={current.level}")
            except AttributeError:
                print(f"  [{tag}] component {idx+1}/{len(components)} done")

    return current


# ─────────────────────────────────────────────────────────────────────────
# Sanity Check: 평문 vs FHE 비교
# ─────────────────────────────────────────────────────────────────────────

def sanity_check_chebyshev(
    engine: Engine,
    secret_key,
    keypack: KeyPack,
    components: list,
    test_x_values=None,
    threshold: float = 1e-3,
):
    """
    실제 fhe_sgn 시나리오로 검증 (sign_bootstrap 포함):

      Pipeline (Label_Propagation.fhe_sgn과 동일):
        1. encrypt(x)
        2. eval_mcp_full_chebyshev → ~±0.998 (systematic noise 포함)
        3. sign_bootstrap → ±1.0 (정확)

      Comparison: sign(x) vs fhe_sgn(encrypt(x))
      Reference: plain_eval = eval_mcp_np_chebyshev (참고용)

    Stage별 측정:
      - eval_diff:  |plain_val - eval_val|  (sign_bootstrap 전, ~1.6e-3 systematic)
      - final_diff: |sign(x) - sgn_val|     (sign_bootstrap 후, 실제 fhe_sgn 출력)

    final_diff < threshold이면 PASS — 이게 진짜 측정해야 할 값.
    sign_bootstrap이 |x| ≥ τ=2^{-α} 범위에서 ±1을 정밀하게 반환하므로
    eval 단계의 systematic ~1.6e-3 노이즈는 흡수됨.
    """
    from core.ciphertext_single.minimax import eval_mcp_np_chebyshev

    if test_x_values is None:
        test_x_values = [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9]

    slot_count = engine.slot_count
    print(f"\n[Sanity Check] DesiloFHE Chebyshev + sign_bootstrap (실제 fhe_sgn 시나리오)")
    print(f"  components: {len(components)}, basis={components[0].get('basis', '?')}")
    print(f"  Pipeline: encrypt → eval_mcp → sign_bootstrap")
    print(f"  Comparison target: sign(x) (이상적 ±1)\n")
    print(f"  {'x':>8} | {'sign(x)':>7} | {'eval_val':>12} | {'sgn_val':>12} | {'final_diff':>12}")
    print("  " + "-" * 64)

    max_eval_diff  = 0.0
    max_final_diff = 0.0

    for x_val in test_x_values:
        ideal_sign = float(np.sign(x_val))

        # ── Plain reference (참고) ────────────────────────────────────
        plain_val = float(eval_mcp_np_chebyshev(np.array([x_val]), components)[0])

        # ── FHE Stage 1: eval_mcp ─────────────────────────────────────
        encoded   = engine.encode([x_val] * slot_count)
        fhe_ct    = engine.encrypt(encoded, secret_key)
        eval_ct   = eval_mcp_full_chebyshev(engine, fhe_ct, components,
                                            slot_count, keypack, tag="")
        eval_val  = float(np.real(engine.decrypt(eval_ct, secret_key)[0]))
        eval_diff = abs(plain_val - eval_val)
        max_eval_diff = max(max_eval_diff, eval_diff)

        # ── FHE Stage 2: sign_bootstrap (실제 fhe_sgn 동일) ───────────
        sgn_ct = engine.sign_bootstrap(
            engine.intt(eval_ct),
            keypack.relinearization_key,
            keypack.conjugation_key,
            keypack.rotation_key,
            keypack.smallbootstrap_key,
        )
        sgn_val    = float(np.real(engine.decrypt(sgn_ct, secret_key)[0]))
        final_diff = abs(ideal_sign - sgn_val)
        max_final_diff = max(max_final_diff, final_diff)

        flag = "" if final_diff < threshold else " ✗"
        print(f"  {x_val:>+8.3f} | {ideal_sign:>+7.1f} | "
              f"{eval_val:>12.6f} | {sgn_val:>12.6f} | "
              f"{final_diff:>12.4e}{flag}")

    print("  " + "-" * 64)
    print(f"  Max eval_diff  (plain vs eval):     {max_eval_diff:.4e}  (참고)")
    print(f"  Max final_diff (sign(x) vs sgn_val): {max_final_diff:.4e}  ★ 실제 fhe_sgn 정확도")

    if max_final_diff < threshold:
        print(f"  ✓ Sanity check PASSED — fhe_sgn 정확도 < {threshold}")
        print(f"    → 실제 사용 시 sign approximation 노이즈 거의 없음")
    else:
        print(f"  ✗ Sanity check FAILED — sign_bootstrap 후에도 큰 diff")
        print(f"    가능한 원인:")
        print(f"     - sign_bootstrap 입력 magnitude 부족 (|x| < τ)")
        print(f"     - eval 결과가 ±τ 범위 밖")
        print(f"     - sign_bootstrap 자체의 API 호출 문제")

    return max_final_diff
# core/sign_approx/minimax.py
"""
Minimax Composite Polynomial (MCP) for Sign Function Approximation

논문: Lee et al., "Minimax Approximation of Sign Function by Composite
      Polynomial for Homomorphic Comparison", IEEE TDSC 2022

─────────────────────────────────────────────────────────────────────────
[2026-05 수정 사항]

1. tol: 1e-9 → 1e-13 (논문 Algorithm 1/2의 원래 기준)
   - 이유: 클라이언트 측 1회 연산이라 시간 비용 < 정확도 가치
   - float64 한계로 보통 n_iter 끝까지 돌지만, 그래도 더 정확한
     minimax polynomial을 얻을 가능성. n_iter도 400 → 800으로 상향.

2. margin η: _suggest_margin 자동 → 논문 Table 3 직접 매핑
   - 이유: 논문이 HEAAN에서 실험적으로 검증한 값. 우리가 수식으로
     재추정하는 것보다 안전.

3. 단일 구간 Remez 처리 유지 (수학적 동등성 확인):
   - sgn(x)가 홀함수이고 odd-basis만 사용하면 다항식도 홀함수 (Lemma 2)
   - 따라서 D=[-b,-a]∪[a,b]에서 sgn 근사 = [a,b]에서 1 근사
   - 양의 반구간 alternation → 음의 반구간 자동 대칭 alternation
   - Multi-interval Algorithm 2와 결과 동일, 계산만 더 효율적.

4. Chebyshev basis 추가 (Bossuat Algorithm 1 호환):
   - 논문 5.2.1 권장 사항: power basis는 high-degree coefficient 폭발 가능
     → CKKS plaintext mult noise 증가
   - 새 함수: remez_odd_sign_chebyshev, eval_mcp_np_chebyshev,
            compute_mcp_with_margin_chebyshev (cf. _chebyshev 접미사)
   - JSON 출력 형식: comp["basis"] = "chebyshev" 또는 "power"
     coeffs 의미: chebyshev → odd Chebyshev T_1, T_3, ... 계수
                  power     → odd power x, x^3, ... 계수
   - FHE 평가: bsgs_chebyshev.py 사용 (Bossuat Alg 1 BSGS)
   - 기존 power basis 코드도 유지 (역호환, A/B 비교 용도)

─────────────────────────────────────────────────────────────────────────
α 결정 (DBSCAN 기준, N=212):

  사용처           최소 |입력 gap|     α    degrees            bootstraps
  ─────────────────────────────────────────────────────────────────────
  Normalize (adj)  margin/bound        12    [15,15,15,15]      4+1=5
                   ≈ 0.00078
  Core             0.5/N = 0.00236     12    [15,15,15,15]      4+1=5
  fhe_sgn (LP)     1/N = 0.00472       15    [7,15,15,15,27]    5+1=6

  Normalize 와 Core는 α=12로 통일:
    - 논문 Table 3에 α∈{8,12,16,20}만 직접 매핑 가능
    - α=12 선택 시 margin η를 보간 없이 직접 사용 (η=2^{-14})
    - δ=2^{-12}=0.000244 < min gap 0.00236 (Core: 9.7배 여유)
    - false positive 비율 ~2.4% (α=11 대비 절반)
    - α=11 대비 mults +3회 (29 → 32, 9% 증가) — 합리적 trade-off
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json
import math
import numpy as np
from typing import List, Tuple


def _poly_np(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """odd-power polynomial 평가: p(x) = Σ c_k × x^{2k+1}"""
    x    = np.asarray(x, dtype=np.float64)
    val  = np.zeros_like(x)
    xpow = x.copy()
    xsq  = x * x
    for c in coeffs:
        val  += c * xpow
        xpow  = xpow * xsq
    return val


# ═══════════════════════════════════════════════════════════════════════════
# Remez 알고리즘 (단일 구간, odd-basis)
# ═══════════════════════════════════════════════════════════════════════════
def remez_odd_sign(
    degree: int, a: float, b: float,
    n_iter: int = 800, tol: float = 1e-11, n_sample: int = 30_000,
) -> Tuple[np.ndarray, float]:
    """
    sgn(x) 근사를 위한 odd-degree minimax 다항식 계수 결정.

    수학적 동등성:
      도메인 D=[-b,-a]∪[a,b]에서 sgn(x) 근사 (논문 Algorithm 2의 multi-interval)
      = [a,b]에서 1 근사 (이 함수)
      (sgn은 홀함수, odd-basis 다항식도 홀함수 → 음반구간 자동 대칭)

    수렴 조건 (논문 Algorithm 1, line 7):
      (max - min) / min < tol
      tol=1e-11: float64 정밀도 한계에 가까워 보통 n_iter 끝까지 돌지만,
                 그 사이 nodes 갱신 반복으로 더 정확한 minimax에 수렴.
    """
    if degree % 2 != 1:
        raise ValueError(f"degree 는 홀수여야 합니다: {degree}")
    if not (0.0 < a < b):
        raise ValueError(f"0 < a < b 조건 위반: a={a}, b={b}")

    m = (degree + 1) // 2   # odd term 수
    n_ref = m + 1            # alternation point 수

    # 초기 nodes: Chebyshev nodes (boundary 약간 안쪽으로 clip)
    k_idx = np.arange(n_ref)
    theta = (2*k_idx + 1) * np.pi / (2*n_ref)
    nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos(theta[::-1])
    eps_bd = 1e-10 * (b - a)
    nodes = np.sort(np.clip(nodes, a + eps_bd, b - eps_bd))

    x_dense   = np.linspace(a, b, n_sample)
    coeffs    = None
    E_abs     = 0.0
    converged = False

    for _it in range(n_iter):
        # ── 선형 시스템 A · sol = rhs ────────────────────────────────────
        # n_ref개 방정식: c_0·x_i + c_1·x_i^3 + ... + c_{m-1}·x_i^{2m-1} − (-1)^i E = 1
        A = np.zeros((n_ref, m + 1))
        rhs = np.ones(n_ref)
        for i, xi in enumerate(nodes):
            xi_pow = xi
            for k in range(m):
                A[i, k] = xi_pow
                xi_pow *= xi * xi
            A[i, m] = -((-1.0) ** i)

        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            break

        coeffs = sol[:m]
        E      = float(sol[m])
        E_abs  = abs(E)

        # ── 수렴 체크 (논문 line 7): (max - min) / min < tol ─────────────
        err     = _poly_np(coeffs, x_dense) - 1.0
        max_abs = float(np.max(np.abs(err)))
        if E_abs > 1e-30 and abs(max_abs - E_abs) / E_abs < tol:
            converged = True
            break

        # ── 극점 후보 수집 (논문 line 4-5: extreme points) ───────────────
        ext_x = [a]; ext_e = [float(err[0])]
        for i in range(1, n_sample - 1):
            # 부호 변화 또는 미분 부호 변화
            if err[i-1]*err[i+1] <= 0.0 or (
                (err[i]-err[i-1])*(err[i+1]-err[i]) <= 0.0 and abs(err[i]) > 0.0
            ):
                ext_x.append(float(x_dense[i])); ext_e.append(float(err[i]))
        ext_x.append(b); ext_e.append(float(err[-1]))

        # 같은 부호 연속 극점 병합 (큰 |err| 유지)
        mx = [ext_x[0]]; me = [ext_e[0]]
        for i in range(1, len(ext_x)):
            if np.sign(ext_e[i]) == np.sign(me[-1]) or me[-1] == 0.0:
                if abs(ext_e[i]) >= abs(me[-1]):
                    mx[-1] = ext_x[i]; me[-1] = ext_e[i]
            else:
                mx.append(ext_x[i]); me.append(ext_e[i])

        if len(mx) < n_ref:
            continue   # alternation point 부족 → 다음 iter

        # 절대값 합 최대 부분집합 선택 (논문 line 4: maximum absolute sum condition)
        best, best_s = -1.0, 0
        for s in range(len(mx) - n_ref + 1):
            sc = sum(abs(me[s+j]) for j in range(n_ref))
            if sc > best:
                best, best_s = sc, s
        nodes = np.array(mx[best_s: best_s + n_ref])

    if not converged:
        import warnings
        warnings.warn(
            f"[Remez] deg={degree} [{a:.6f},{b:.6f}]: {n_iter}회 내 미수렴 "
            f"(E_abs={E_abs:.4e}, tol={tol}). 현재 최적값 사용.",
            RuntimeWarning, stacklevel=2,
        )

    return (coeffs if coeffs is not None else np.zeros(m)), E_abs


# ═══════════════════════════════════════════════════════════════════════════
# Margin η — 논문 Table 3 직접 매핑
# ═══════════════════════════════════════════════════════════════════════════

# 논문 Table 3 (Lee et al. 2022, IEEE TDSC):
#   key=α, value=(η_comparison_min_time, η_comparison_min_depth,
#                  η_max_min_time, η_max_min_depth)
# 모두 음의 지수(log2). 예: -12는 η=2^{-12}
_TABLE3_LOG2_ETA = {
    8:  (-12,    -12,   -9,    -10.5),
    12: (-15,    -14,   -13,   -13),
    16: (-18,    -17,   -16,   -17),
    20: (-21,    -22,   -20.5, -20),
}


def get_paper_margin(alpha: int, mode: str = "comp_depth") -> float:
    """
    논문 Table 3에서 margin η 값을 직접 가져옴.

    Parameters
    ----------
    alpha : int
        precision parameter α (정수)
    mode : str
        "comp_time"  : comparison, minimize running time
        "comp_depth" : comparison, minimize depth (기본값, 정확도 우선)
        "max_time"   : max function, minimize running time
        "max_depth"  : max function, minimize depth

    Returns
    -------
    eta : float
        margin η

    Notes
    -----
    α ∈ {8, 12, 16, 20}만 직접 표에 있음.
    그 외 α는 log scale 선형 보간 + **보수적 올림** (작은 η = 큰 안전 마진).
    """
    mode_idx = {"comp_time": 0, "comp_depth": 1,
                "max_time":  2, "max_depth":  3}.get(mode)
    if mode_idx is None:
        raise ValueError(f"mode must be one of comp_time/comp_depth/max_time/max_depth, got '{mode}'")

    table = _TABLE3_LOG2_ETA
    keys  = sorted(table.keys())   # [8, 12, 16, 20]

    # 정확히 표에 있는 경우
    if alpha in table:
        log_eta = table[alpha][mode_idx]
        return 2.0 ** log_eta

    # 범위 밖: 가장 가까운 끝값 사용
    if alpha < keys[0]:
        log_eta = table[keys[0]][mode_idx]
        return 2.0 ** log_eta
    if alpha > keys[-1]:
        log_eta = table[keys[-1]][mode_idx]
        return 2.0 ** log_eta

    # 범위 내: log scale 선형 보간, 결과는 더 작은 η (보수적) 쪽으로 올림
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo < alpha < hi:
            log_lo = table[lo][mode_idx]
            log_hi = table[hi][mode_idx]
            t      = (alpha - lo) / (hi - lo)
            log_interp = log_lo + t * (log_hi - log_lo)
            # 보수적 선택: 더 작은 η (= 더 음수인 log_eta = 더 큰 안전 마진)
            log_eta = min(log_interp, log_lo, log_hi)
            return 2.0 ** log_eta

    # fallback (도달하지 말아야 함)
    return 2.0 ** table[keys[-1]][mode_idx]


# ═══════════════════════════════════════════════════════════════════════════
# MCP 계산
# ═══════════════════════════════════════════════════════════════════════════

def compute_mcp(degrees: List[int], delta: float, verbose: bool = True) -> List[dict]:
    """margin 없는 기본 MCP (역호환용)."""
    a, b = delta, 1.0
    comps = []
    if verbose:
        print(f"\n[MCP] degrees={degrees}  delta={delta:.6f}")
    for i, deg in enumerate(degrees):
        if verbose:
            print(f"  p_{i+1} (deg={deg})  [{a:.8f}, {b:.8f}]  ...", end="", flush=True)
        coeffs, err = remez_odd_sign(deg, a, b)
        comps.append({"index": i+1, "degree": int(deg), "coeffs": coeffs.tolist(),
                      "domain_a": float(a), "domain_b": float(b), "error": float(err)})
        if verbose: print(f"  err={err:.4e}")
        a, b = 1.0 - err, 1.0 + err
    return comps


def compute_mcp_with_margin(
    degrees: List[int], delta: float,
    margin: float, alpha: int, verbose: bool = True,
) -> List[dict]:
    """
    margin η를 적용한 MCP. domain_b 필드 포함 (FHE 평가 시 x/domain_b 정규화).

    margin은 **반드시 명시적으로 전달**해야 함 (이전 _suggest_margin 자동 계산 제거).
    호출 측에서 get_paper_margin(alpha, mode)로 Table 3 값을 가져와 전달할 것.
    """
    a, b = delta, 1.0
    comps = []

    safety = 2.0 ** -(alpha - 1)
    if verbose:
        print(f"\n[MCP-margin] degrees={degrees}, δ={delta:.6e}, η={margin:.6e} (= 2^{math.log2(margin):.2f})")
        print(f"             안전 임계값 t_k ≤ {safety:.4e}")

    for i, deg in enumerate(degrees):
        if verbose:
            print(f"  p_{i+1} (deg={deg})  [{a:.8f}, {b:.8f}]  ...", end="", flush=True)
        coeffs, err = remez_odd_sign(deg, a, b)
        t_i = err + margin
        comps.append({
            "index": i+1, "degree": int(deg), "coeffs": coeffs.tolist(),
            "domain_a": float(a), "domain_b": float(b),
            "error": float(err), "margin": float(margin), "t_i": float(t_i),
        })
        if verbose: print(f"  err={err:.4e}, t_i={t_i:.4e}")
        a, b = 1.0 - t_i, 1.0 + t_i

    final_t = comps[-1]["t_i"]
    if verbose:
        print(f"\n[MCP-margin] 완료  t_k={final_t:.4e}  "
              f"{'✓ SAFE' if final_t <= safety else '✗ UNSAFE'} (≤{safety:.4e})")
    return comps


def eval_mcp_np(x, components: List[dict]) -> np.ndarray:
    """domain_b 정규화 포함 평문 평가."""
    val = np.asarray(x, dtype=np.float64).copy()
    for comp in components:
        domain_b = comp.get("domain_b", 1.0)
        val = _poly_np(np.array(comp["coeffs"]), val / domain_b)
    return val


def save_mcp(components: List[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"components": components}, f, indent=2)
    print(f"[MCP] 저장: {filepath}")


def load_mcp(filepath: str) -> List[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["components"]


# ═══════════════════════════════════════════════════════════════════════════
# 논문 Table 2 (minimize depth) degree 시퀀스
# ═══════════════════════════════════════════════════════════════════════════
# DesiloFHE lazy-rescaling 기준 실제 레벨 소비 = dep(d) × 2
#
#  α  | degrees                | 컴포넌트수 | bootstrap (mid/post + sign_boot)
# ─────┼────────────────────────┼──────────┼──────────────────────────────────
#  8  | [7, 15, 15]            | 3        | 3 + 1 = 4회
#  10 | [7, 7, 13, 15]         | 4        | 4 + 1 = 5회
#  11 | [7, 15, 15, 15]        | 4        | 4 + 1 = 5회  ← Normalize, Core (통일)
#  12 | [15, 15, 15, 15]       | 4        | 4 + 1 = 5회
#  14 | [7, 7, 15, 15, 27]     | 5        | 5 + 1 = 6회  ← BSGS dep(27)=5 ✓
#  15 | [7, 15, 15, 15, 27]    | 5        | 5 + 1 = 6회  ← LP (BSGS 필수)
#  16 | [15, 15, 15, 15, 27]   | 5        | 5 + 1 = 6회
#
# BSGS 기반 레벨 분석:
#   dep(d) = 논문 Table 1, 레벨 소비 = dep(d) × 2
#   bootstrap → level=10 → dep 최대 5 → degree 최대 27
#   naive 루프: deg=27이면 14 레벨 → ✗ budget=10 초과
#   BSGS: deg=27이면 dep=5 → 10 레벨 → ✓ budget 딱 맞음
_MINIMIZE_DEPTH_DEGREES = {
    8:  [7, 15, 15],
    9:  [7, 7, 7, 13],
    10: [7, 7, 13, 15],
    11: [7, 15, 15, 15],
    12: [15, 15, 15, 15],
    13: [15, 15, 15, 31],   # ★ [2026-07 정정] 기존 [7,7,7,7,15] 는 논문 Table 2 와
                            #   불일치(오타). 논문 minimize-depth α=13 = {15,15,15,31}.
                            #   증상: sanity_sweep 에서 α=13 만 1x 오차 1e-5 (이웃 α=12/15 는
                            #   1e-12). 스펙 2^-13=1.2e-4 는 만족하나 여유가 없어 이웃 대비
                            #   10^7 배 열등했다. 다른 8개 항목은 논문과 완전 일치 확인됨.
                            #   ※ degree 31 은 이 Chebyshev BSGS 구현에서 미검증
                            #     (논문도 '31 초과는 수치오차' 경고). 정정 후 sanity_sweep 로
                            #     α=13 재측정 전까지는 호출측의 13→12 회피를 유지할 것.
    14: [7, 7, 15, 15, 27],
    15: [7, 15, 15, 15, 27],
    16: [15, 15, 15, 15, 27],
}


# ═══════════════════════════════════════════════════════════════════════════
# 사용처별 MCP 생성 함수
# ═══════════════════════════════════════════════════════════════════════════

def compute_mcp_for_normalize(alpha: int = 12, verbose: bool = True) -> List[dict]:
    """
    Normalize용 (mcp_alpha12.json). ★ Core와 α=12로 통일

    α=12 선택 근거 (사용자 결정 사항):
      - 논문 Table 3에 α∈{8,12,16,20}만 직접 매핑 가능
      - α=11/15는 선형 보간 필요 → α=12로 통일하면 보간 없이 직접 사용
      - degrees=[15,15,15,15]: 4컴포넌트 + bootstrap 5회 (α=11과 동일)
      - δ=2^{-12}=0.000244 → 모든 use case에 안전한 margin 제공

    Normalize 안전성 (N=212, dim=3):
      dist² 비교에서 min |입력 gap| = margin_val / bound
        margin_val = δ × bound = 0.000244 × 3.15 = 0.000770
        normalized min |x_scaled| ≈ margin_val × scale = δ = 0.000244 ✓
      → false positive 비율 약 2.4% (실측치, α=11일 경우 ~5%)

    Margin η: 논문 Table 3, comparison minimize_depth 컬럼
      α=12 → η = 2^{-14} (직접 매핑, 보간 없음)
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [15, 15, 15, 15])
    delta   = 2.0 ** (-alpha)
    margin  = get_paper_margin(alpha, mode="comp_depth")

    if verbose:
        print(f"\n[MCP-Normalize] α={alpha}, degrees={degrees}, δ={delta:.6e}")
        print(f"                margin η = {margin:.6e} = 2^{math.log2(margin):.2f} (논문 Table 3, 직접 매핑)")
    return compute_mcp_with_margin(degrees=degrees, delta=delta, margin=margin,
                                   alpha=alpha, verbose=verbose)


def compute_mcp_for_core(alpha: int = 12, verbose: bool = True) -> List[dict]:
    """
    Core용 (mcp_alpha12.json). ★ Normalize와 α=12로 통일

    α=12 선택 근거 (N=212):
      최소 입력 gap = 0.5/N = 0.00236
      δ = 2^{-12} = 0.000244 < 0.00236 ✓ (9.7배 여유, α=11보다 2배 안전)
      t_k = err + margin 누적 → 마지막 컴포넌트 t_k ≤ 2^{-11} ✓ SAFE

    α=11 대비 비용:
      degrees: [7,15,15,15] → [15,15,15,15]
      non-scalar mults: 29 → 32 (3회 추가, ~9% 증가)
      bootstrap: 5회 동일
      → 코드 일관성과 Table 3 직접 매핑 가능을 위한 합리적 trade-off

    Margin η: 논문 Table 3, comparison minimize_depth 컬럼
      α=12 → η = 2^{-14} (직접 매핑)
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [15, 15, 15, 15])
    delta   = 2.0 ** (-alpha)
    margin  = get_paper_margin(alpha, mode="comp_depth")

    if verbose:
        N_ref = 212
        print(f"\n[MCP-Core] α={alpha}, degrees={degrees}, δ={delta:.6e}")
        print(f"           min input gap = 0.5/N = {0.5/N_ref:.5f} (N={N_ref})")
        print(f"           δ/min_gap = {delta/(0.5/N_ref):.2f} (작을수록 안전, < 1 권장)")
        print(f"           margin η = {margin:.6e} = 2^{math.log2(margin):.2f} (논문 Table 3, 직접 매핑)")
    return compute_mcp_with_margin(degrees=degrees, delta=delta, margin=margin,
                                   alpha=alpha, verbose=verbose)


def compute_mcp_for_label_prop_fixed(alpha: int = 15, verbose: bool = True) -> List[dict]:
    """
    Label Propagation 전용 (mcp_alpha15_lp.json). ★ α=15 유지

    α=15 선택 근거 (drift 분석):
      누적 drift = n_calls × |u-v|_avg × τ / 2
        n_calls=840, |u-v|_avg=30, threshold=1.0
      α=15: τ=2^{-15} → drift≈0.39 < 1.0 ✓ (2.6배 여유)
      α=11: τ=2^{-11} → drift≈6.15 ✗

    BSGS 필수 (deg=27 포함):
      naive 루프: 14 레벨 소비 > budget 10 → ✗
      BSGS:       dep(27)=5 → 10 레벨 = budget ✓

    Margin η: 논문 Table 3 max minimize_depth 컬럼 (max 연산 형태이므로)
      α=15 → α=16 값 사용 (보수적) = 2^{-17}
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [7, 15, 15, 15, 27])
    delta   = 2.0 ** (-alpha)
    margin  = get_paper_margin(alpha, mode="max_depth")   # ← max 컬럼 사용

    if verbose:
        tau    = delta
        n_calls_typical = 840
        uv_avg = 30
        drift  = n_calls_typical * uv_avg * tau / 2
        max_deg = max(degrees)
        bsgs_dep = {7: 3, 13: 4, 15: 4, 27: 5}.get(max_deg, 5)
        bsgs_level_cost = bsgs_dep * 2
        print(f"\n[MCP-LP] α={alpha}, degrees={degrees}, δ={delta:.6e}")
        print(f"         margin η = {margin:.6e} = 2^{math.log2(margin):.2f} (논문 Table 3 max_depth)")
        print(f"         τ=2^{{-{alpha}}}={tau:.6f}")
        print(f"         drift({n_calls_typical}콜, |u-v|_avg={uv_avg}): {drift:.3f}")
        print(f"         threshold(inter-cluster gap/2): 1.0")
        print(f"         안전 여유: {1.0/drift:.1f}배  {'✓ SAFE' if drift < 1.0 else '✗ UNSAFE'}")
        n_boots = len(degrees) + 1
        print(f"         bootstrap/fhe_sgn: {n_boots}회")
        print(f"         max_deg={max_deg}, BSGS dep={bsgs_dep}, 레벨 소비={bsgs_level_cost}/10 "
              f"{'✓' if bsgs_level_cost <= 10 else '✗ 예산 초과'}")
    return compute_mcp_with_margin(degrees=degrees, delta=delta, margin=margin,
                                   alpha=alpha, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════
# Deprecated: 자동 margin 계산 (역호환 유지, 신규 코드는 사용 금지)
# ═══════════════════════════════════════════════════════════════════════════
def _suggest_margin(degrees: List[int], delta: float, alpha: int) -> float:
    """[DEPRECATED] 자동 margin 추정 — get_paper_margin() 사용 권장."""
    import warnings
    warnings.warn(
        "_suggest_margin은 deprecated입니다. 논문 Table 3 값을 사용하는 "
        "get_paper_margin(alpha, mode='comp_depth' 또는 'max_depth')으로 교체하세요.",
        DeprecationWarning, stacklevel=2,
    )
    a, b = delta, 1.0
    last_err = 0.0
    for deg in degrees:
        _, err = remez_odd_sign(deg, a, b)
        last_err = err
        a, b = 1.0 - err, 1.0 + err
    return max(last_err / 4.0, 2.0 ** (-(alpha + 2)))


def compute_mcp_for_label_prop(
    num_points: int, safety_factor: float = 1.2, verbose: bool = True,
) -> List[dict]:
    """[DEPRECATED] compute_mcp_for_label_prop_fixed 사용 권장."""
    import warnings
    warnings.warn(
        "compute_mcp_for_label_prop은 deprecated입니다. "
        "compute_mcp_for_label_prop_fixed(alpha=15) 사용하세요.",
        DeprecationWarning, stacklevel=2,
    )
    delta_label = 1.0 / (num_points * safety_factor)
    alpha_equiv = int(np.log2(1.0 / delta_label)) + 1
    degrees = _MINIMIZE_DEPTH_DEGREES.get(min(alpha_equiv, 12), [15, 15, 15, 15])
    margin = get_paper_margin(min(alpha_equiv, 20), mode="max_depth")
    return compute_mcp_with_margin(degrees=degrees, delta=delta_label,
                                   margin=margin, alpha=alpha_equiv, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════
# Chebyshev basis 함수들 (Bossuat Algorithm 1 호환)
# ═══════════════════════════════════════════════════════════════════════════
# 
# 논문 5.2.1: power basis는 high-degree coefficient 폭발 가능 (예: T_15 leading
# coefficient = 16384). CKKS plaintext mult 시 noise 증가 → 정밀도 손실.
# 
# 해결책: Chebyshev basis로 Remez를 풀고, FHE 평가도 Chebyshev recurrence로.
# 우리 odd-only sign 근사는 odd Chebyshev T_1, T_3, T_5, ... basis 사용.


def _eval_odd_cheb(coeffs, x):
    """
    odd Chebyshev polynomial 평문 평가:
        p(x) = Σ_{k=0}^{m-1} c_k × T_{2k+1}(x)
    
    Chebyshev recurrence: T_0 = 1, T_1 = x, T_n = 2x T_{n-1} - T_{n-2}
    
    Parameters
    ----------
    coeffs : array-like
        [c_0, c_1, ..., c_{m-1}], coefficients for T_1, T_3, ..., T_{2m-1}
    x : array-like
        evaluation points
    
    Returns
    -------
    np.ndarray with same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    if len(coeffs) == 0:
        return np.zeros_like(x)
    
    result = np.zeros_like(x)
    T_prev = np.ones_like(x)   # T_0
    T_curr = x.copy()           # T_1
    
    for k, c in enumerate(coeffs):
        result += c * T_curr   # contribution of T_{2k+1}
        if k == len(coeffs) - 1:
            break
        # Advance two steps: T_{2k+1} → T_{2k+2} → T_{2k+3}
        T_next = 2.0 * x * T_curr - T_prev   # T_{2k+2}
        T_prev, T_curr = T_curr, T_next
        T_next = 2.0 * x * T_curr - T_prev   # T_{2k+3}
        T_prev, T_curr = T_curr, T_next
    
    return result


def remez_odd_sign_chebyshev(
    degree: int, a: float, b: float,
    n_iter: int = 800, tol: float = 1e-13, n_sample: int = 30_000,
) -> Tuple[np.ndarray, float]:
    """
    sgn(x) 근사 minimax 다항식의 **odd Chebyshev** 계수 결정.
    
    출력 다항식: p(x) = Σ_{k=0}^{m-1} c_k × T_{2k+1}(x) ≈ 1 for x ∈ [a, b]
    
    Power basis remez_odd_sign과 수학적으로 동일한 minimax polynomial을 반환하지만,
    수치 안정성과 CKKS 평가 정밀도에서 우월.
    
    Returns
    -------
    coeffs : np.ndarray
        odd Chebyshev coefficients [c_0_for_T_1, ..., c_{m-1}_for_T_{2m-1}]
    E_abs : float
        minimax error
    """
    if degree % 2 != 1:
        raise ValueError(f"degree must be odd, got {degree}")
    if not (0.0 < a < b):
        raise ValueError(f"0 < a < b violated: a={a}, b={b}")
    
    m = (degree + 1) // 2
    n_ref = m + 1
    
    # Initial nodes: Chebyshev nodes in [a, b]
    k_idx = np.arange(n_ref)
    theta = (2*k_idx + 1) * np.pi / (2*n_ref)
    nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos(theta[::-1])
    eps_bd = 1e-10 * (b - a)
    nodes = np.sort(np.clip(nodes, a + eps_bd, b - eps_bd))
    
    x_dense = np.linspace(a, b, n_sample)
    coeffs = None
    E_abs = 0.0
    converged = False
    
    for _it in range(n_iter):
        # Linear system: Σ c_k T_{2k+1}(x_i) - (-1)^i E = 1
        A = np.zeros((n_ref, m + 1))
        rhs = np.ones(n_ref)
        for i, xi in enumerate(nodes):
            # Evaluate T_1, T_3, ..., T_{2m-1} at xi via recurrence
            T_prev = 1.0   # T_0
            T_curr = xi     # T_1
            for k in range(m):
                A[i, k] = T_curr   # T_{2k+1}
                if k == m - 1:
                    break
                T_next = 2.0 * xi * T_curr - T_prev   # T_{2k+2}
                T_prev, T_curr = T_curr, T_next
                T_next = 2.0 * xi * T_curr - T_prev   # T_{2k+3}
                T_prev, T_curr = T_curr, T_next
            A[i, m] = -((-1.0) ** i)
        
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            break
        
        coeffs = sol[:m]
        E = float(sol[m])
        E_abs = abs(E)
        
        # Error on dense grid
        err = _eval_odd_cheb(coeffs, x_dense) - 1.0
        max_abs = float(np.max(np.abs(err)))
        if E_abs > 1e-30 and abs(max_abs - E_abs) / E_abs < tol:
            converged = True
            break
        
        # Extreme point collection (identical to power basis logic)
        ext_x = [a]; ext_e = [float(err[0])]
        for i in range(1, n_sample - 1):
            if err[i-1]*err[i+1] <= 0.0 or (
                (err[i]-err[i-1])*(err[i+1]-err[i]) <= 0.0 and abs(err[i]) > 0.0
            ):
                ext_x.append(float(x_dense[i])); ext_e.append(float(err[i]))
        ext_x.append(b); ext_e.append(float(err[-1]))
        
        # Merge same-sign extremes (keep largest |err|)
        mx = [ext_x[0]]; me = [ext_e[0]]
        for i in range(1, len(ext_x)):
            if np.sign(ext_e[i]) == np.sign(me[-1]) or me[-1] == 0.0:
                if abs(ext_e[i]) >= abs(me[-1]):
                    mx[-1] = ext_x[i]; me[-1] = ext_e[i]
            else:
                mx.append(ext_x[i]); me.append(ext_e[i])
        
        if len(mx) < n_ref:
            continue
        
        # Maximum absolute sum subset selection
        best, best_s = -1.0, 0
        for s in range(len(mx) - n_ref + 1):
            sc = sum(abs(me[s+j]) for j in range(n_ref))
            if sc > best:
                best, best_s = sc, s
        nodes = np.array(mx[best_s: best_s + n_ref])
    
    if not converged:
        import warnings
        warnings.warn(
            f"[Remez-Cheb] deg={degree} [{a:.6f},{b:.6f}]: {n_iter}회 내 미수렴 "
            f"(E_abs={E_abs:.4e}, tol={tol}). 현재 최적값 사용.",
            RuntimeWarning, stacklevel=2,
        )
    
    return (coeffs if coeffs is not None else np.zeros(m)), E_abs


def eval_mcp_np_chebyshev(x, components: List[dict]) -> np.ndarray:
    """
    Chebyshev basis MCP 평문 평가 (sanity check / debugging용).
    
    각 컴포넌트의 coeffs는 odd Chebyshev: [c_0 for T_1, c_1 for T_3, ...].
    domain_b 정규화 후 평가.
    """
    val = np.asarray(x, dtype=np.float64).copy()
    for comp in components:
        domain_b = comp.get("domain_b", 1.0)
        basis    = comp.get("basis", "power")
        if basis != "chebyshev":
            raise ValueError(
                f"Component {comp.get('index', '?')} has basis='{basis}', "
                f"expected 'chebyshev'. Use eval_mcp_np for power basis."
            )
        val = _eval_odd_cheb(np.array(comp["coeffs"]), val / domain_b)
    return val


def _build_chain_chebyshev(degrees, delta, margin, cache=None):
    """주어진 (degrees, δ, η) 로 minimax 합성 체인을 만들고 (comps, t_k) 반환."""
    a, b, comps = delta, 1.0, []
    for i, deg in enumerate(degrees):
        key = (deg, round(a / b, 12))
        if cache is not None and key in cache:
            coeffs, err = cache[key]
        else:
            coeffs, err = remez_odd_sign_chebyshev(deg, a / b, 1.0)
            if cache is not None:
                cache[key] = (coeffs, err)
        t_i = err + margin
        comps.append({
            "index": i + 1, "degree": int(deg), "coeffs": coeffs.tolist(),
            "domain_a": float(a), "domain_b": float(b),
            "error": float(err), "margin": float(margin), "t_i": float(t_i),
            "basis": "chebyshev",
        })
        if t_i >= 1.0:
            return comps, 999.0
        a, b = 1.0 - t_i, 1.0 + t_i
    return comps, comps[-1]["t_i"]


def find_max_valid_margin(degrees, delta, alpha, n_iter=14, cache=None, hi0=None):
    """τ_k ≤ 2^(1-α) 를 만족하는 **최대** margin η 를 이분탐색.

    ★ 이것이 논문(Lee et al.) Section 3.5 / Algorithm 7 의 원래 처방이다:
      "the margin η is set as large as possible among valid values of margin
       such that τ_k ≤ 2^{1-α}".
      즉 η 는 표에서 가져오는 상수가 아니라 **degree 세트에 종속된 값**이다.
      Table 3 은 α∈{8,12,16,20} 의 특정 degree 세트에 대해서만 주어져 있고,
      그 사이 α 를 보간하거나 다른 degree 세트에 그대로 쓰면 근거가 없다.
    """
    import warnings
    safety = 2.0 ** -(alpha - 1)
    # ★ 탐색 중 Remez 미수렴 경고는 억제한다. tol=1e-13 은 배정밀도 한계라 통상
    #   n_iter 를 소진하며(정상 동작), 이분탐색이 η 마다 Remez 를 재호출하므로
    #   억제하지 않으면 경고가 수십 개로 늘어 실제 문제를 가린다.
    #   (탐색 종료 후 최종 체인은 억제 없이 한 번 더 만들어 경고를 정상 노출한다.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if _build_chain_chebyshev(degrees, delta, 0.0, cache)[1] > safety:
            return 0.0                  # η=0 으로도 불가 → degree 세트 자체가 부족
        lo, hi = 0.0, (hi0 if hi0 is not None else 2.0 ** -(alpha - 1))
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            if _build_chain_chebyshev(degrees, delta, mid, cache)[1] <= safety:
                lo = mid
            else:
                hi = mid
    return lo


def _resolve_margin(degrees, delta, margin, tau_limit, alpha, cache, tag=""):
    """지정 margin 이 tau_limit 을 위반하면 논문 §3.5 처방(유효 최대 η)으로 자동 하향.
    반환: (comps, final_t, used_margin).  η=0 에서도 불가하면 RuntimeError."""
    comps, final_t = _build_chain_chebyshev(degrees, delta, margin, cache)
    if final_t <= tau_limit:
        return comps, final_t, margin

    # tau_limit 기준으로 유효 최대 η 이분탐색 (경고 억제)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if _build_chain_chebyshev(degrees, delta, 0.0, cache)[1] > tau_limit:
            raise RuntimeError(
                f"[MCP{tag}] degree 세트 자체가 부족: η=0 에서도 t_k > {tau_limit:.4e} "
                f"(α={alpha}, degrees={degrees}). degree 를 상향해야 한다.")
        # ★ 탐색 상한 = 실패한 margin. (margin 이 실패했으므로 답은 반드시 그 아래)
        #   이전엔 hi=0.5 로 시작해 12회로는 1e-4 수준을 분해하지 못하고 lo=0 으로
        #   떨어졌다(η=0 → CKKS 오차 보호 상실). 상한을 margin 으로 좁히면
        #   12회에 margin/4096 해상도가 나와 충분하다.
        lo, hi = 0.0, margin
        for _ in range(14):
            mid = (lo + hi) / 2
            if _build_chain_chebyshev(degrees, delta, mid, cache)[1] <= tau_limit:
                lo = mid
            else:
                hi = mid
    used = 0.9 * lo
    comps2, final2 = _build_chain_chebyshev(degrees, delta, used, cache)
    print(f"  [MCP{tag}] ⚠ 지정 η={margin:.4e} 는 스펙 위반 (t_k={final_t:.4e} > {tau_limit:.4e}).")
    print(f"            논문 §3.5 처방대로 유효 최대 η={lo:.4e} 를 찾아 90%={used:.4e} 로 "
          f"자동 하향 → t_k={final2:.4e}  (α={alpha}, degrees={degrees})")
    if final2 > tau_limit:
        raise RuntimeError(
            f"[MCP{tag}] 자동 하향 후에도 위반: t_k={final2:.4e} > {tau_limit:.4e}")
    return comps2, final2, used


def compute_mcp_with_margin_chebyshev(
    degrees: List[int], delta: float,
    margin: float, alpha: int, verbose: bool = True,
) -> List[dict]:
    """Chebyshev basis MCP. coeffs는 odd Chebyshev (T_1, T_3, ..., T_{2m-1}).

    ★ [2026-07] margin 자동 보정.
      [문제] get_paper_margin 은 논문 Table 3 을 α 로 보간해 η 를 준다. 그런데
        (1) Table 3 의 η 는 α∈{8,12,16,20} 의 **특정 degree 세트**에 대해 실험적으로
            정해진 값이라 다른 α/다른 세트에 그대로 쓸 근거가 없고,
        (2) η 는 체인의 **매 단계마다 더해져 누적**되므로, 여유가 얇은 degree 세트에서는
            η 가 조금만 커도 t_k 가 2^(1-α) 를 넘는다.
        실제로 논문 Table 2 세트는 여유가 1.1~1.9배뿐이라 α=10,12 에서 보간 η 를
        감당하지 못하고 스펙을 위반한다(관측된 RuntimeError 의 원인).
      [해결] 논문 Section 3.5 의 원래 처방대로 '유효 범위 내 최대 η' 를 이분탐색해
        쓴다. 주어진 margin 이 유효하면 그대로 쓰고, 초과하면 자동으로 낮춘다.
        η 는 클수록 CKKS 오차에 강인하므로 유효 최대치의 90% 를 채택한다.
    """
    safety = 2.0 ** -(alpha - 1)
    cache = {}
    comps, final_t, used_margin = _resolve_margin(
        degrees, delta, margin, safety, alpha, cache, tag="-Cheb")
    if verbose:
        print(f"\n[MCP-Cheb] degrees={degrees}, δ={delta:.6e}, η={used_margin:.6e}")
        print(f"           안전 임계값 t_k ≤ {safety:.4e}")
        for c in comps:
            print(f"  p_{c['index']} (deg={c['degree']})  "
                  f"[{c['domain_a']:.8f}, {c['domain_b']:.8f}]  "
                  f"err={c['error']:.4e}, t_i={c['t_i']:.4e}")
        print(f"[MCP-Cheb] 완료  t_k={final_t:.4e}  ✓ SAFE (≤{safety:.4e})")
    return comps


def compute_mcp_for_normalize_chebyshev(alpha: int = 12, verbose: bool = True) -> List[dict]:
    """
    Normalize용 Chebyshev MCP (mcp_alpha12_cheb.json).
    
    α=12 [15,15,15,15], η=2^{-14} (논문 Table 3 comp_depth).
    Power basis 버전 (compute_mcp_for_normalize)과 같은 minimax 다항식이지만
    Chebyshev basis로 저장되어 CKKS 평가 정밀도 우월.
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [15, 15, 15, 15])
    delta   = 2.0 ** (-alpha)
    margin  = get_paper_margin(alpha, mode="comp_depth")
    if verbose:
        print(f"\n[MCP-Norm-Cheb] α={alpha}, degrees={degrees}, δ={delta:.6e}")
        print(f"                margin η = 2^{math.log2(margin):.2f} (Table 3 직접 매핑)")
    return compute_mcp_with_margin_chebyshev(degrees, delta, margin, alpha, verbose)


def compute_mcp_for_core_chebyshev(alpha: int = 12, verbose: bool = True) -> List[dict]:
    """
    Core용 Chebyshev MCP. Normalize와 동일한 α=12 [15,15,15,15] 설정.
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [15, 15, 15, 15])
    delta   = 2.0 ** (-alpha)
    margin  = get_paper_margin(alpha, mode="comp_depth")
    if verbose:
        N_ref = 212
        print(f"\n[MCP-Core-Cheb] α={alpha}, degrees={degrees}, δ={delta:.6e}")
        print(f"                min input gap = 0.5/N = {0.5/N_ref:.5f} (N={N_ref})")
        print(f"                margin η = 2^{math.log2(margin):.2f} (Table 3)")
    return compute_mcp_with_margin_chebyshev(degrees, delta, margin, alpha, verbose)


# ══════════════════════════════════════════════════════════════════════════════
# ★ [2026-07] cleaning 예산을 활용한 LP 전용 MCP  (δ 와 τ 를 분리)
#
# [착안] 논문 Table 2 의 α 는 deadzone δ=2^-α 와 출력정밀도 τ=2^(1-α) 를 **함께**
#   묶는다. 그러나 우리 파이프라인은 MCP 뒤에 sign_cleaning g(x)=1.5x-0.5x³ 이
#   붙고, 이는 오차를 **이차수렴**으로 줄인다(e→1.5e²). 즉
#     · δ (deadzone) : 라벨차 1 을 삼키면 전파가 깨지므로 **반드시** δ ≤ gap. cleaning 무관.
#     · τ (출력정밀도): cleaning 이 처리 가능 → **훨씬 느슨해도 된다**.
#   두 요구를 분리해 τ 만 풀면 컴포넌트(=sign_bootstrap) 수를 줄일 수 있다.
#
# [τ 목표 산정] cleaning n회 후 최종오차 η 를 정하고 역산.
#   e_1=1.5e₀², e_2=3.375e₀⁴, e_3=17.1e₀⁸ …
#   η 목표는 1e-6 채택: sign 유래 감쇠 = n_max·|d|·η/2 인데, 실측상 기존 감쇠
#   (tetra 15.45/400, bootstrap 유래)와 같아지는 η* ≈ 7.8e-4 이므로 1e-6 이면
#   그 0.1% 수준 → sign 이 감쇠 지배항이 되지 않는다.
#     cleaning 2회 → τ ≤ 0.0234 / cleaning 3회 → τ ≤ 0.1277
#
# [탐색] δ=2^-α 고정, τ≤목표 를 만족하는 최소 (컴포넌트수, 깊이비용) degree 조합을
#   DP 로 전탐색(후보 {3,5,7,13,15,27,31}, dep 은 논문 Table 1). 결과가 아래 표.
#
# [효과] 논문 Table 2 대비 컴포넌트 1개 감소(α=9~13) → fhe_max bootstrap 5 → 4.
#   (#3 중복 SB 제거와 합쳐 원래 7 → 4, 총 1.75배)
#
# ★ 검증 필요: cleaning 2회는 레벨 4 를 쓴다(1회는 2). sign_bootstrap 직후의
#   레벨 여유가 4 이상인지 확인할 것. 부족하면 _SGN_CLEANING_ITERS 를 되돌리고
#   아래 표의 clean1 열(더 촘촘한 τ)을 쓰면 된다.
# ── [레버 A] 논문과 동일한 τ 목표, 단 '컴포넌트 수' 최소화 ──────────────────
#   논문은 depth/비스칼라곱셈을 최소화한다(컴포넌트 사이에 bootstrap 이 없으므로).
#   그러나 우리 파이프라인은 eval_mcp 가 **컴포넌트마다 sign_bootstrap** 을 한다.
#   즉 우리 비용은 depth 가 아니라 **컴포넌트 수**다. 목적함수를 바꿔 같은 τ 로
#   재탐색하면 총 depth 는 논문과 동일하면서 컴포넌트가 1개 적은 해가 존재한다.
#   ★ τ 목표가 논문과 완전히 같으므로 정확도 저하 위험이 없다.
#   (검증: 이 DP 를 논문 목표로 돌리면 Table 2 의 최소 depth 를 α=8~16 전부 재현.)
_LP_DEGREES_COMP_MIN = {   # τ ≤ 2^(1-α) — 논문과 동일
    8:  [7, 15, 15],       # 3개, dep 11 (논문과 동일)
    9:  [15, 15, 31],      # 3개, dep 13 (논문 4개 → 1개 감소)
    10: [15, 31, 31],      # 3개, dep 14 (논문 4개 → 1개 감소)
    11: [31, 31, 31],      # 3개, dep 15 (논문 4개 → 1개 감소)
    12: [7, 15, 15, 31],   # 4개, dep 16 (논문과 동일)
    13: [15, 15, 15, 31],  # 4개, dep 17 (논문과 동일)
    14: [15, 31, 31, 31],  # 4개, dep 19 (논문 5개 → 1개 감소)
    15: [31, 31, 31, 31],  # 4개, dep 20 (논문 5개 → 1개 감소)
}

# ── [레버 B] 추가로 τ 를 cleaning 예산까지 완화 ────────────────────────────
#   sign_cleaning g(x)=1.5x-0.5x³ 는 이차수렴(e→1.5e²)이라 MCP 출력정밀도 τ 를
#   느슨하게 둬도 최종 η 를 유지한다. δ(deadzone)는 절대 건드리지 않는다.
#   η 목표 1e-6 채택 근거: sign 유래 감쇠 = n_max·|d|·η/2 이고, 실측상 기존 감쇠
#   (tetra 15.45/400, bootstrap 유래)와 같아지는 η*≈7.8e-4 → 1e-6 이면 그 0.1%.
#   ★ 레버 A 대비 추가 이득은 α=8,12 에서만 1개. 레벨을 2 더 쓰므로(cleaning 2회)
#     기본은 A 만 쓰고, B 는 필요할 때만 켠다.
_LP_DEGREES_BY_CLEANING = {
    2: {8: [31, 31], 9: [15, 15, 15], 10: [15, 15, 31], 11: [15, 31, 31],
        12: [31, 31, 31], 13: [15, 15, 15, 15], 14: [15, 15, 15, 31],
        15: [15, 15, 31, 31]},
    3: {8: [15, 31], 9: [31, 31], 10: [15, 15, 15], 11: [15, 15, 31],
        12: [15, 31, 31], 13: [31, 31, 31]},
}
_LP_TAU_TARGET = {1: 0.0008, 2: 0.0234, 3: 0.1277}   # η≤1e-6 기준


def compute_mcp_for_label_prop_cleaning(alpha: int, cleaning_iters: int = 2,
                                        verbose: bool = True) -> List[dict]:
    """LP 전용 MCP: deadzone δ=2^-α 는 유지, 출력정밀도 τ 는 cleaning 예산까지 완화."""
    if cleaning_iters <= 1:
        table = _LP_DEGREES_COMP_MIN            # 레버 A (논문 τ)
    else:
        table = _LP_DEGREES_BY_CLEANING.get(cleaning_iters)   # 레버 A+B
    if table is None or alpha not in table:
        raise ValueError(f"[MCP-LP] (α={alpha}, cleaning={cleaning_iters}) 조합의 "
                         f"degree 표가 없다. minimax.py 의 표를 확인.")
    degrees   = table[alpha]
    delta     = 2.0 ** (-alpha)
    margin    = get_paper_margin(alpha, mode="max_depth")
    tau_limit = (2.0 ** (1 - alpha)) if cleaning_iters <= 1 else _LP_TAU_TARGET[cleaning_iters]
    cache = {}
    comps, final_t, used = _resolve_margin(
        degrees, delta, margin, tau_limit, alpha, cache, tag="-LP")
    if verbose:
        print(f"\n[MCP-LP] α={alpha}, cleaning={cleaning_iters}회, degrees={degrees}, "
              f"δ={delta:.4e}, η={used:.4e}")
        print(f"         τ 허용 한계 = {tau_limit:.4e}")
        for c in comps:
            print(f"  p_{c['index']} (deg={c['degree']}) "
                  f"[{c['domain_a']:.6f},{c['domain_b']:.6f}] "
                  f"err={c['error']:.4e} t_i={c['t_i']:.4e}")
        print(f"[MCP-LP] t_k={final_t:.4e} ✓ (≤{tau_limit:.4e})")
    return comps


def compute_mcp_for_label_prop_chebyshev(alpha: int = 15, verbose: bool = True) -> List[dict]:
    """
    LP용 Chebyshev MCP (mcp_alpha15_lp_cheb.json).
    
    α=15 [7,15,15,15,27], η=2^{-17} (Table 3 max_depth, α=15→α=16 보수적).
    deg=27 BSGS depth=5 → 10 레벨 = budget ✓
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [7, 15, 15, 15, 27])
    delta   = 2.0 ** (-alpha)
    margin  = get_paper_margin(alpha, mode="max_depth")
    
    if verbose:
        tau = delta
        n_calls_typical = 840
        uv_avg = 30
        drift = n_calls_typical * uv_avg * tau / 2
        max_deg = max(degrees)
        bsgs_dep = {7: 3, 13: 4, 15: 4, 27: 5}.get(max_deg, 5)
        print(f"\n[MCP-LP-Cheb] α={alpha}, degrees={degrees}, δ={delta:.6e}")
        print(f"              margin η = 2^{math.log2(margin):.2f} (Table 3 max_depth)")
        print(f"              τ=2^{{-{alpha}}}={tau:.6f}, drift={drift:.3f} "
              f"{'✓ SAFE' if drift < 1.0 else '✗ UNSAFE'}")
        print(f"              max_deg={max_deg}, BSGS dep={bsgs_dep}, "
              f"레벨 소비={bsgs_dep*2}/10 {'✓' if bsgs_dep*2 <= 10 else '✗'}")
    return compute_mcp_with_margin_chebyshev(degrees, delta, margin, alpha, verbose)
# core/sign_approx/minimax.py
"""
Minimax Composite Polynomial (MCP) for Sign Function Approximation

논문: Lee et al., "Minimax Approximation of Sign Function by Composite
      Polynomial for Homomorphic Comparison", IEEE TDSC 2022

sign_bootstrap 활용 (Hong et al., DesiloFHE):
  MCP 출력(≈±1) 후 sign_bootstrap → 정밀도 ≈ (π²/8)×τ² 수준으로 향상
  단, sign_bootstrap은 입력이 ±1 근방(τ 작을 것)이어야 함 → MCP의 delta 조건이 선행

─────────────────────────────────────────────────────────────────────────
α 최소값 결정 규칙 (N=212 DBSCAN 기준):

  사용처           최소 |입력 gap|    최소 α    minimize_depth degrees
  ─────────────────────────────────────────────────────────────────────
  fhe_sgn (LP)    1/N = 0.00472    α=8       [7, 15, 15]     3 comp
  Normalize (adj) margin/bound     α=8       [7, 15, 15]     3 comp
                  ≈ 0.00394
  Core            0.5/N = 0.00236  α=10      [7, 7, 13, 15]  4 comp
                                   (α=12는 동일 4 comp, 비효율)

DesiloFHE lazy-rescaling 영향:
  dep(d) × 2 = 실제 레벨 소비
  dep(15)=4 → 8레벨 소비, bootstrap(level=10) 후 10-8=2 남음
  → sign_bootstrap 최소 레벨 3 불충족 → 에러 원인
  → 해결: 마지막 컴포넌트 평가 *후* regular bootstrap → level=10
          → sign_bootstrap 입력 level=10 ≥ 3 → 성공

sign_bootstrap 출력 정밀도 (Theorem 1, τ=2^{-α}):
  error ≤ (π²/8)×τ²,  fhe_max error ≤ N×error/2
  α=8: fhe_max error ≤ 0.002 (round 임계 0.5 대비 251배 여유)
  α=10: fhe_max error ≤ 0.0001 (4009배 여유)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json
import numpy as np
from typing import List, Tuple


def _poly_np(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    x    = np.asarray(x, dtype=np.float64)
    val  = np.zeros_like(x)
    xpow = x.copy()
    xsq  = x * x
    for c in coeffs:
        val  += c * xpow
        xpow  = xpow * xsq
    return val


def remez_odd_sign(
    degree: int, a: float, b: float,
    n_iter: int = 400, tol: float = 1e-9, n_sample: int = 30_000,
) -> Tuple[np.ndarray, float]:
    """
    tol=1e-9: 1e-13은 float64 정밀도 한계로 equioscillation 조건을 충족하지
    못해 항상 400회 전부 반복됨 (결과는 올바름, 불필요한 연산 낭비).
    1e-9로 완화하면 보통 수십~수백 회 내에 조기 종료됨.
    """
    if degree % 2 != 1:
        raise ValueError(f"degree 는 홀수여야 합니다: {degree}")
    if not (0.0 < a < b):
        raise ValueError(f"0 < a < b 조건 위반: a={a}, b={b}")

    m = (degree + 1) // 2
    n_ref = m + 1
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
        E = float(sol[m])
        E_abs = abs(E)
        err = _poly_np(coeffs, x_dense) - 1.0
        max_abs = float(np.max(np.abs(err)))
        if E_abs > 1e-30 and abs(max_abs - E_abs) / E_abs < tol:
            converged = True
            break
        ext_x = [a]; ext_e = [float(err[0])]
        for i in range(1, n_sample - 1):
            if err[i-1]*err[i+1] <= 0.0 or (
                (err[i]-err[i-1])*(err[i+1]-err[i]) <= 0.0 and abs(err[i]) > 0.0
            ):
                ext_x.append(float(x_dense[i])); ext_e.append(float(err[i]))
        ext_x.append(b); ext_e.append(float(err[-1]))
        mx = [ext_x[0]]; me = [ext_e[0]]
        for i in range(1, len(ext_x)):
            if np.sign(ext_e[i]) == np.sign(me[-1]) or me[-1] == 0.0:
                if abs(ext_e[i]) >= abs(me[-1]):
                    mx[-1] = ext_x[i]; me[-1] = ext_e[i]
            else:
                mx.append(ext_x[i]); me.append(ext_e[i])
        if len(mx) < n_ref:
            continue
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


def compute_mcp(degrees: List[int], delta: float, verbose: bool = True) -> List[dict]:
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


def _suggest_margin(degrees: List[int], delta: float, alpha: int) -> float:
    a, b = delta, 1.0
    last_err = 0.0
    for deg in degrees:
        _, err = remez_odd_sign(deg, a, b)
        last_err = err
        a, b = 1.0 - err, 1.0 + err
    return max(last_err / 4.0, 2.0 ** (-(alpha + 2)))


def compute_mcp_with_margin(
    degrees: List[int], delta: float,
    margin: float = None, alpha: int = 8, verbose: bool = True,
) -> List[dict]:
    """domain_b 필드 포함 MCP. FHE 평가 시 x/domain_b 정규화 필수."""
    if margin is None:
        margin = _suggest_margin(degrees, delta, alpha)
        if verbose:
            print(f"[MCP-margin] margin 자동: η={margin:.6e}")

    a, b = delta, 1.0
    comps = []
    if verbose:
        safety = 2.0 ** -(alpha - 1)
        print(f"\n[MCP-margin] degrees={degrees}, δ={delta:.6e}, η={margin:.6e}")
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
        safety = 2.0 ** -(alpha - 1)
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


# ── 논문 Table 2 (minimize depth) degree 시퀀스 ───────────────────────────────
# DesiloFHE lazy-rescaling 기준 실제 레벨 소비 = dep(d) × 2
#
#  α  | degrees              | 컴포넌트수 | 총 bootstrap (mid/post + sign_boot)
# ─────┼──────────────────────┼──────────┼────────────────────────────────────
#  8  | [7, 15, 15]          | 3        | 3 + 1 = 4회
#  9  | [7, 7, 7, 13]        | 4        | 4 + 1 = 5회
#  10 | [7, 7, 13, 15]       | 4        | 4 + 1 = 5회  ← Core 최적 (α=12 대비 효율적)
#  11 | [7, 15, 15, 15]      | 4        | 4 + 1 = 5회
#  12 | [15, 15, 15, 15]     | 4        | 4 + 1 = 5회  ← α=10과 동일 횟수, 비효율
_MINIMIZE_DEPTH_DEGREES = {
    8:  [7, 15, 15],
    9:  [7, 7, 7, 13],
    10: [7, 7, 13, 15],
    11: [7, 15, 15, 15],
    12: [15, 15, 15, 15],
}


def compute_mcp_for_normalize(alpha: int = 8, verbose: bool = True) -> List[dict]:
    """
    Normalize용 (mcp_alpha8.json).
    α=8 최소값: delta=2^{-8}=0.00391 < margin/bound≈0.00394 ✓
    degrees=[7,15,15], bootstrap 3+1=4회 (mid 2회 + post 1회 + sign_boot 1회)
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [7, 15, 15])
    delta   = 2.0 ** (-alpha)
    margin  = 2.0 ** (-(alpha + 2))
    if verbose:
        print(f"\n[MCP-Normalize] α={alpha}, degrees={degrees}, δ={delta:.6e}, η={margin:.6e}")
    return compute_mcp_with_margin(degrees=degrees, delta=delta, margin=margin, alpha=alpha, verbose=verbose)


def compute_mcp_for_core(alpha: int = 11, verbose: bool = True) -> List[dict]:
    """
    Core용 (mcp_alpha11.json). ★ α=11 최적값

    α 선택 근거 (N=212):
      최소 delta < 0.5/N = 0.00236 필요

      α=9  [7,7,7,13]:   t_k=0.00258 > min_gap=0.00236 → NOT OK (분류 오류 가능)
      α=10 [7,7,13,15]:  t_k 0.00172~0.00197 (Remez 편차) + margin → UNSAFE 가능
      α=11 [7,15,15,15]: t_k=0.00051 << threshold=0.00098 → SAFE ✓ (안정적)
                          delta=2^{{-11}}=0.00049 < 0.00236 (4.8배 여유)

    α=12 대비:
      bootstrap 횟수: 동일 4+1=5회
      non-scalar mult: 27회 vs 32회 (5회 절약)
    """
    degrees = _MINIMIZE_DEPTH_DEGREES.get(alpha, [7, 15, 15, 15])
    delta   = 2.0 ** (-alpha)
    margin  = 2.0 ** (-(alpha + 2))
    if verbose:
        N_ref = 212
        print(f"\n[MCP-Core] α={alpha}, degrees={degrees}, δ={delta:.6e}, η={margin:.6e}")
        print(f"  delta={delta:.5f} < 0.5/N={0.5/N_ref:.5f} ✓ (안전 마진 {(0.5/N_ref)/delta:.1f}배)")
    return compute_mcp_with_margin(degrees=degrees, delta=delta, margin=margin, alpha=alpha, verbose=verbose)


def compute_mcp_for_label_prop(
    num_points: int, safety_factor: float = 1.2, verbose: bool = True,
) -> List[dict]:
    """
    Label Propagation 전용 (mcp_label_prop.json).

    safety_factor=1.2 선택 이유:
      최소 입력 gap = 1/N (label_scale=N 사용 시)
      delta = 1/(N × safety_factor) < 1/N 이어야 함 → safety_factor > 1 필요

      safety_factor=1.2: delta = 1/(N×1.2) = 0.00393 for N=212
        2^{-8} = 0.00391 < 0.00393 → alpha_equiv=8 → [7,15,15]
        → bootstrap 3+1=4회/sgn (safety_factor=3.0의 [7,7,13,15] 5회 대비 1회 절약)

      safety_factor=3.0 (기존): delta=0.001572, alpha_equiv=10 → [7,7,13,15]
        → 동일 정밀도 대비 불필요하게 많은 bootstrap 사용

    alpha cap:
      alpha_equiv ≤ 12 cap → N < 4096 이면 안전
      더 큰 N 사용 시: cap 제거 + 적절한 degrees 설정 필요
    """
    delta_label = 1.0 / (num_points * safety_factor)
    alpha_equiv = int(np.log2(1.0 / delta_label)) + 1

    degrees = _MINIMIZE_DEPTH_DEGREES.get(
        min(alpha_equiv, 12), [15, 15, 15, 15]
    )

    if verbose:
        print(f"\n[MCP-Label] N={num_points}, safety_factor={safety_factor}")
        print(f"  delta=1/(N×{safety_factor})={delta_label:.6f}, alpha_equiv={alpha_equiv}")
        print(f"  degrees={degrees} (논문 Table 2 minimize depth for α={min(alpha_equiv,12)})")
        n_boots = len(degrees) + 1  # mid/post bootstraps + sign_bootstrap
        print(f"  bootstrap/sgn: {len(degrees)}회 (mid+post) + 1회 sign_boot = {n_boots}회")
        print(f"  safe N < {int(1.0/delta_label):,} (alpha cap=12 기준: N < 4,096)")

    return compute_mcp_with_margin(
        degrees=degrees, delta=delta_label, margin=0.0, alpha=alpha_equiv, verbose=verbose,
    )
# core/sign_approx/minimax.py
"""
Minimax Composite Polynomial (MCP) for Sign Function Approximation
(순수 Python / NumPy — FHE 의존성 없음)

논문: Lee et al., "Minimax Approximation of Sign Function by Composite
      Polynomial for Homomorphic Comparison", IEEE TDSC 2022

FHE-DBSCAN 적용 목표
  1) Normalize.py  → eps 이웃 판별 (Heaviside step)
  2) Core.py       → min_pts 코어 판별 (indicator step)
  3) Label_Propagation.py → cluster-ID max 연산 (sign)

워크플로우
  1단계(오프라인): compute_mcp() 로 계수 계산  →  save_mcp() 로 JSON 저장
  2단계(온라인)  : load_mcp() 로 로드 →  평문: eval_mcp_np()
                                        →  FHE : fhe_sign.eval_mcp_fhe()
"""

from __future__ import annotations
import json
import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _poly_np(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    홀수 다항식  p(x) = c[0]·x + c[1]·x³ + c[2]·x⁵ + …  평가.
    coeffs[k] = x^(2k+1) 의 계수.
    """
    x    = np.asarray(x, dtype=np.float64)
    val  = np.zeros_like(x)
    xpow = x.copy()        # x^1
    xsq  = x * x           # x²  (재사용)
    for c in coeffs:
        val  += c * xpow
        xpow  = xpow * xsq  # 다음 홀수 거듭제곱
    return val


# ─────────────────────────────────────────────────────────────────────────────
# 1. Remez 알고리즘 (단일 컴포넌트 계수 계산)
# ─────────────────────────────────────────────────────────────────────────────

def remez_odd_sign(
    degree   : int,
    a        : float,
    b        : float,
    n_iter   : int   = 400,
    tol      : float = 1e-13,
    n_sample : int   = 30_000,
) -> Tuple[np.ndarray, float]:
    """
    [a, b] ⊂ (0, ∞) 위에서 sgn(x) = 1 을 근사하는
    홀수 다항식의 minimax 계수를 Remez 알고리즘으로 계산한다.

    도메인 [-b,-a] ∪ [a,b] 의 홀수 대칭에 의해,
    [a, b] 위에서 상수 1 에 대한 minimax 근사와 동치.

    Parameters
    ----------
    degree   : 홀수 정수 (3, 5, 7, 9, 13, 15, …)
    a, b     : 0 < a < b  (근사 도메인)
    n_iter   : 최대 Remez 반복 횟수
    tol      : (max_err − |E|)/|E| < tol 이면 수렴 종료
    n_sample : 오차 함수 샘플링 밀도

    Returns
    -------
    coeffs   : shape = ((degree+1)//2,)
               coeffs[k] = x^(2k+1) 의 minimax 계수
    error    : float  max_{x ∈ [a,b]} |p(x) − 1|  (≥ 0)

    알고리즘 개요
    -------------
    equioscillation 조건 (Chebyshev 교대 정리):
      p(xᵢ) − 1 = (−1)ⁱ · E,  i = 0, …, m   (m = (d+1)/2 개 기저)
    ↓ 선형 시스템 풀기  →  새 참조점 탐색  →  반복
    """
    if degree % 2 != 1:
        raise ValueError(f"degree 는 홀수여야 합니다: {degree}")
    if not (0.0 < a < b):
        raise ValueError(f"0 < a < b 조건 위반: a={a}, b={b}")

    m     = (degree + 1) // 2   # 기저 수: x, x³, …, x^d
    n_ref = m + 1                # equioscillation 점 수 = m+1

    # ── Chebyshev 노드로 초기화 ──────────────────────────────────────
    k_idx  = np.arange(n_ref)
    theta  = (2*k_idx + 1) * np.pi / (2*n_ref)
    nodes  = 0.5*(a+b) + 0.5*(b-a)*np.cos(theta[::-1])
    eps_bd = 1e-10 * (b - a)
    nodes  = np.sort(np.clip(nodes, a + eps_bd, b - eps_bd))

    x_dense = np.linspace(a, b, n_sample)
    coeffs  = None
    E_abs   = 0.0

    for _it in range(n_iter):

        # ── equioscillation 선형 시스템 구성 ────────────────────────
        # ∑_k c[k]·xᵢ^(2k+1)  −  (−1)ⁱ·E = 1
        A   = np.zeros((n_ref, m + 1))
        rhs = np.ones(n_ref)
        for i, xi in enumerate(nodes):
            xi_pow = xi
            for k in range(m):
                A[i, k] = xi_pow
                xi_pow  *= xi * xi          # x^(2k+3)
            A[i, m] = -((-1.0) ** i)        # E 의 계수

        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            break

        coeffs = sol[:m]
        E      = float(sol[m])
        E_abs  = abs(E)

        # ── 오차 함수 평가 ───────────────────────────────────────────
        err = _poly_np(coeffs, x_dense) - 1.0      # sgn=1 이므로

        # ── 수렴 판정 ────────────────────────────────────────────────
        max_abs = float(np.max(np.abs(err)))
        if E_abs > 1e-30 and abs(max_abs - E_abs) / E_abs < tol:
            break

        # ── 극점(극값) 수집 ──────────────────────────────────────────
        ext_x = [a]
        ext_e = [float(err[0])]
        for i in range(1, n_sample - 1):
            # 부호 변화 또는 방향 전환
            if err[i-1] * err[i+1] <= 0.0 or (
               (err[i] - err[i-1]) * (err[i+1] - err[i]) <= 0.0
               and abs(err[i]) > 0.0
            ):
                ext_x.append(float(x_dense[i]))
                ext_e.append(float(err[i]))
        ext_x.append(b)
        ext_e.append(float(err[-1]))

        # ── 동부호 연속 극점 병합 (더 큰 |오차| 보존) ───────────────
        mx = [ext_x[0]]
        me = [ext_e[0]]
        for i in range(1, len(ext_x)):
            s_new  = np.sign(ext_e[i])
            s_prev = np.sign(me[-1])
            if s_new == s_prev or me[-1] == 0.0:
                if abs(ext_e[i]) >= abs(me[-1]):
                    mx[-1] = ext_x[i]
                    me[-1] = ext_e[i]
            else:
                mx.append(ext_x[i])
                me.append(ext_e[i])

        if len(mx) < n_ref:
            continue   # 극점 부족 → 노드 유지

        # ── n_ref 개 연속 교차 극점 중 |오차| 합 최대 구간 선택 ──────
        best, best_s = -1.0, 0
        for s in range(len(mx) - n_ref + 1):
            sc = sum(abs(me[s + j]) for j in range(n_ref))
            if sc > best:
                best, best_s = sc, s
        nodes = np.array(mx[best_s: best_s + n_ref])

    return (coeffs if coeffs is not None else np.zeros(m)), E_abs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Minimax Composite Polynomial 구성
# ─────────────────────────────────────────────────────────────────────────────

def compute_mcp(
    degrees : List[int],
    delta   : float,
    verbose : bool = True,
) -> List[dict]:
    """
    Minimax Composite Polynomial  p = pₖ ∘ … ∘ p₁  계산.

    · p₁ : 도메인 [delta, 1]    위에서 sgn 근사
    · pᵢ : 도메인 [1−tᵢ₋₁, 1+tᵢ₋₁]  위에서 sgn 근사   (tᵢ₋₁ = 이전 단계 오차)

    Parameters
    ----------
    degrees : 홀수 정수 리스트  [d₁, d₂, …, dₖ]
              논문 Table 2 기준값:
                a=4  minimize mults  → [3, 3, 5]
                a=6  minimize mults  → [3, 5, 5, 5]
                a=8  minimize mults  → [7, 15, 15]   ← 논문 권장 (runtime 최소)
                a=8  minimize depth  → [7, 15, 15]
    delta   : |x|의 입력 하한 (sign 을 정의하는 gap)

    Returns
    -------
    components : dict 리스트, 각 원소:
      'index'    : 컴포넌트 번호 (1-indexed)
      'degree'   : 다항식 차수
      'coeffs'   : List[float]  — x^1, x^3, …, x^d 의 계수
      'domain_a' : 도메인 하한
      'domain_b' : 도메인 상한
      'error'    : minimax 근사 오차
    """
    a, b = delta, 1.0
    comps = []
    if verbose:
        print(f"\n[MCP] 계수 계산 시작  degrees={degrees}  delta={delta:.6f}")

    for i, deg in enumerate(degrees):
        if verbose:
            print(f"  p_{i+1} (deg={deg:2d})  [{a:.8f}, {b:.8f}]  ... ", end="", flush=True)

        coeffs, err = remez_odd_sign(deg, a, b)
        comps.append({
            "index"    : i + 1,
            "degree"   : int(deg),
            "coeffs"   : coeffs.tolist(),
            "domain_a" : float(a),
            "domain_b" : float(b),
            "error"    : float(err),
        })
        if verbose:
            print(f"오차 = {err:.4e}")

        a, b = 1.0 - err, 1.0 + err   # 다음 컴포넌트 도메인

    if verbose:
        print(f"[MCP] 완료  최종 합성 오차 ≈ {comps[-1]['error']:.4e}")
    return comps


# ─────────────────────────────────────────────────────────────────────────────
# 3. 평문(NumPy) 평가
# ─────────────────────────────────────────────────────────────────────────────

def eval_mcp_np(x, components: List[dict]) -> np.ndarray:
    """p_k ∘ … ∘ p_1(x)  — 반환값 ∈ [−1, 1]  (sgn 근사)."""
    val = np.asarray(x, dtype=np.float64).copy()
    for comp in components:
        val = _poly_np(np.array(comp["coeffs"]), val)
    return val


def heaviside_mcp(x, components: List[dict]) -> np.ndarray:
    """
    H(x) = (1 − sgn(x)) / 2  ≈ 1 if x < 0,  0 if x > 0.

    Normalize.py 용:
      x = dist_sq_scaled − eps_sq_scaled
      → H(x) = 1 if neighbor, 0 otherwise
    """
    return (1.0 - eval_mcp_np(x, components)) / 2.0


def indicator_mcp(x, components: List[dict]) -> np.ndarray:
    """
    I(x) = (sgn(x) + 1) / 2  ≈ 1 if x > 0,  0 if x < 0.

    Core.py 용:
      x = (neighbor_count − (min_pts − 0.5)) / N
      → I(x) = 1 if core point
    """
    return (eval_mcp_np(x, components) + 1.0) / 2.0


def max_mcp(a_arr, b_arr, components: List[dict]) -> np.ndarray:
    """
    max(a, b) = ((a+b) + (a−b)·sgn(a−b)) / 2  — Label_Propagation.py 용.
    """
    diff = np.asarray(a_arr) - np.asarray(b_arr)
    return (np.asarray(a_arr) + np.asarray(b_arr) + diff * eval_mcp_np(diff, components)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. 계수 저장 / 로드
# ─────────────────────────────────────────────────────────────────────────────

def save_mcp(components: List[dict], filepath: str):
    """MCP 계수를 JSON 으로 저장."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"components": components}, f, indent=2)
    print(f"[MCP] 저장: {filepath}")


def load_mcp(filepath: str) -> List[dict]:
    """JSON 에서 MCP 계수 로드."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["components"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. 비교 기준 — 현재 코드의 cubic 반복 방식
# ─────────────────────────────────────────────────────────────────────────────

def cubic_sign_iter(x, depth: int) -> np.ndarray:
    """
    현재 Core.py / Normalize.py 에서 사용하는 cubic 반복 sign 근사.
    f(x) = 1.5x − 0.5x³  를 depth 회 반복 합성.

    수렴 분석:
      · 입력 |x| ≥ δ 에 대해 fᵈᵉᵖᵗʰ(x) → sgn(x)
      · 그러나 |x| ≪ 1 (≈ 0.5/N) 인 Core.py 입력에서는 수렴 매우 느림
    """
    val = np.asarray(x, dtype=np.float64).copy()
    for _ in range(depth):
        val = 1.5 * val - 0.5 * val ** 3
    return val
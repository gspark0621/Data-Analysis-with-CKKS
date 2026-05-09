"""
=======================================================================
Minimax Composite Polynomial for Comparison in HE
논문: Lee et al., "Minimax Approximation of Sign Function by
      Composite Polynomial for Homomorphic Comparison"
      IEEE TDSC, Vol.19, No.6, 2022

목표:
  - Multi-interval Remez 알고리즘으로 각 step 의 minimax 계수 추출
  - Composite polynomial p_k ∘ ... ∘ p_1 구성
  - Gaussian 분포(0 근처) 데이터에서 plaintext comparison 과의 오차 측정

핵심 수정사항 (NaN 제거):
  1. Chebyshev T_{2i+1} 인덱스 버그 수정 (T_even → T_odd 두 단계 점프)
  2. 데이터를 u,v ∈ [0,1] 로 생성 → diff ∈ [-1,1] 보장 (T_k 발산 방지)
  3. Composite 단계마다 domain 범위 soft-clip 적용
=======================================================================
"""

import numpy as np
from scipy.linalg import solve as scipy_solve
import warnings
warnings.filterwarnings("ignore")


# ───────────────────────────────────────────────────────────────────────
# 논문 Table 1: dep(d), mult(d)
# ───────────────────────────────────────────────────────────────────────
DEP  = {3:2, 5:3, 7:3, 9:4, 11:4, 13:4, 15:4,
        17:5, 19:5, 21:5, 23:5, 25:5, 27:5, 29:5, 31:5}
MULT = {3:2, 5:3, 7:5, 9:5, 11:6, 13:7, 15:8,
        17:8, 19:8, 21:9, 23:9, 25:10, 27:10, 29:11, 31:12}

# 논문 Table 2: alpha → (minimize_mult degrees, minimize_depth degrees)
TABLE2 = {
     4: ([3, 3, 5],                   [27]),
     5: ([5, 5, 5],                   [7, 13]),
     6: ([3, 5, 5, 5],                [15, 15]),
     7: ([3, 3, 5, 5, 5],             [7, 7, 13]),
     8: ([3, 3, 5, 5, 9],             [7, 15, 15]),
     9: ([5, 5, 5, 5, 9],             [7, 7, 7, 13]),
    10: ([5, 5, 5, 5, 5, 5],          [7, 7, 13, 15]),
    11: ([3, 5, 5, 5, 5, 5, 5],       [7, 15, 15, 15]),
    12: ([3, 5, 5, 5, 5, 5, 9],       [15, 15, 15, 15]),
    13: ([3, 5, 5, 5, 5, 5, 5, 5],    [15, 15, 15, 31]),
    14: ([3, 3, 5, 5, 5, 5, 5, 5, 5], [7, 7, 15, 15, 27]),
    15: ([3, 3, 5, 5, 5, 5, 5, 5, 9], [7, 15, 15, 15, 27]),
    16: ([3, 3, 5, 5, 5, 5, 5, 5, 5, 5], [15, 15, 15, 15, 27]),
    20: ([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9], [29, 31, 31, 31, 31]),
}


# ───────────────────────────────────────────────────────────────────────
# 핵심 함수 1: 안정적인 Chebyshev 홀수 기저 평가
# p(x) = sum_{i=0}^{n-1} c_i * T_{2i+1}(x/w)
# ───────────────────────────────────────────────────────────────────────
def eval_cheb_odd(coeffs, x, w=1.0, clip=True):
    """
    T_{2i+1}(x/w) Chebyshev 홀수 기저 평가.
    
    Parameters
    ----------
    coeffs : array-like, shape (n,)  — 홀수 기저 계수 [c_0, c_1, ..., c_{n-1}]
    x      : float or array          — 평가 지점
    w      : float                   — Chebyshev 스케일링 인자 (= domain 상한 b)
    clip   : bool                    — True 이면 x 를 [-w, w] 로 클리핑 (발산 방지)
    
    수식:
        t = x / w
        T_0(t) = 1,  T_1(t) = t,  T_k(t) = 2t*T_{k-1}(t) - T_{k-2}(t)
        T_{even}(t) = 2t * T_{curr}(t)  - T_{prev}(t)   # T_{2i}
        T_{odd}(t)  = 2t * T_{even}(t)  - T_{curr}(t)   # T_{2i+1}  ← 핵심
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = (x.ndim == 0)
    x = np.atleast_1d(x.copy())

    # clip: domain 밖의 값은 ±w 로 고정 → T_k(t) 지수 발산 방지
    if clip:
        x = np.clip(x, -w * 1.0001, w * 1.0001)

    t = x / w                          # [-1, 1] 범위
    n = len(coeffs)

    if n == 0:
        return np.zeros(1 if scalar else len(x))

    # 초기화: T_0=1, T_1=t
    T_prev = np.ones_like(t)           # T_{2i-2}  (초기: T_0)
    T_curr = t.copy()                  # T_{2i-1}  (초기: T_1)
    result  = coeffs[0] * T_curr       # c_0 * T_1

    for i in range(1, n):
        # T_{2i}   = 2t * T_{2i-1} - T_{2i-2}
        T_even  = 2.0 * t * T_curr - T_prev
        # T_{2i+1} = 2t * T_{2i}   - T_{2i-1}   ← 인덱스 두 단계 점프
        T_odd   = 2.0 * t * T_even - T_curr
        result  += coeffs[i] * T_odd
        T_prev  = T_even
        T_curr  = T_odd

    return float(result[0]) if scalar else result


# ───────────────────────────────────────────────────────────────────────
# 핵심 함수 2: Multi-Interval Remez (논문 Algorithm 2)
# domain [a,b] 위에서 sgn(x) 의 홀수 차수 Chebyshev minimax 근사
# ───────────────────────────────────────────────────────────────────────
def remez_minimax(a, b, degree, max_iter=600, tol=1e-13):
    """
    [-b,-a] ∪ [a,b] 위에서 sgn(x) 의 minimax 근사 (논문 Algorithm 2).
    홀수 함수이므로 [a,b] 만 고려. 기저: T_{2i+1}(x/w), w=b.
    
    Parameters
    ----------
    a, b    : float  — domain [a, b], 0 < a < b
    degree  : int    — 홀수 다항식 최고 차수
    
    Returns
    -------
    coeffs  : ndarray, shape ((degree+1)//2,)  — Chebyshev 기저 계수
    error   : float  — minimax 오차 (ME_{a,b}^{degree})
    w       : float  — 스케일링 인자 (= b)
    """
    assert degree % 2 == 1 and degree >= 1, "degree 는 홀수 양의 정수"
    n_basis = (degree + 1) // 2   # T_1, T_3, ..., T_{degree} → n_basis 개
    w = float(b)

    def eval_poly(c, x_arr):
        return eval_cheb_odd(c, x_arr, w=w, clip=False)

    def build_and_solve(pts):
        """
        Equioscillation 조건의 선형 시스템:
          sum_j c_j * T_{2j+1}(x_i/w)  - sgn(x_i)  =  (-1)^i * E
        미지수: c_0, ..., c_{n-1}, E   (총 n_basis+1 개)
        """
        N = n_basis + 1
        A   = np.zeros((N, N))
        rhs = np.zeros(N)
        for i, xi in enumerate(pts):
            t = xi / w
            # T_0 ~ T_{degree} 재귀 계산
            T = [1.0, t]
            for k in range(2, degree + 1):
                T.append(2.0 * t * T[-1] - T[-2])
            for j in range(n_basis):
                A[i, j] = T[2*j + 1]    # T_{2j+1}(xi/w)
            A[i, n_basis] = (-1.0) ** i  # E 의 부호
            rhs[i] = float(np.sign(xi))
        try:
            sol = scipy_solve(A, rhs)
        except Exception:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return sol[:n_basis], float(sol[n_basis])

    # 초기 reference points: Chebyshev node (안정적 수렴)
    k   = np.arange(1, n_basis + 2)
    ref = np.sort(0.5*(a+b) + 0.5*(b-a) *
                  np.cos(np.pi * (2*k - 1) / (2*(n_basis + 1))))
    ref = np.clip(ref, a + 1e-12, b - 1e-12)

    # 오차 계산용 조밀한 격자
    x_dense = np.linspace(a, b, max(3000, 200 * n_basis))

    best_coeffs, best_err = None, np.inf

    for _ in range(max_iter):
        coeffs, E = build_and_solve(ref)
        p_vals    = eval_poly(coeffs, x_dense)
        err_fn    = p_vals - np.sign(x_dense)
        abs_err   = np.abs(err_fn)
        eps_max   = float(np.max(abs_err))

        if eps_max < best_err:
            best_err    = eps_max
            best_coeffs = coeffs.copy()

        # 수렴 판단: |eps_max - |E|| / |E| < tol
        if abs(E) > 1e-15 and (eps_max - abs(E)) / abs(E) < tol:
            break

        # 새 reference points: 오차 함수의 극점(extreme points) 탐색
        flips = np.where(np.diff(np.sign(err_fn)) != 0)[0]
        bps   = [0] + list(flips + 1) + [len(x_dense)]
        ext_x, ext_e = [], []
        for s, e in zip(bps[:-1], bps[1:]):
            if s >= e:
                continue
            idx = int(np.argmax(abs_err[s:e]))
            ext_x.append(x_dense[s + idx])
            ext_e.append(abs_err[s + idx])

        if len(ext_x) < n_basis + 1:
            # 극점 부족 → Chebyshev node 재초기화
            ref = np.sort(0.5*(a+b) + 0.5*(b-a) *
                          np.cos(np.pi * np.arange(1, n_basis+2) / (n_basis+1)))
            ref = np.clip(ref, a + 1e-12, b - 1e-12)
            continue

        ext_x = np.array(ext_x)
        ext_e = np.array(ext_e)

        # 교번(alternating) 조건 유지하며 n_basis+1 개 선택
        order  = np.argsort(ext_e)[::-1]
        chosen, last_s = [], None
        for idx in order:
            xi = ext_x[int(idx)]
            si = int(np.sign(err_fn[int(np.searchsorted(x_dense, xi))]))
            if last_s is None or si != last_s:
                chosen.append(int(idx))
                last_s = si
            if len(chosen) == n_basis + 1:
                break

        if len(chosen) < n_basis + 1:
            sel = np.round(np.linspace(0, len(ext_x)-1, n_basis+1)).astype(int)
            ref = np.sort(ext_x[sel])
        else:
            ref = np.sort(ext_x[chosen])

        ref = np.clip(ref, a + 1e-12, b - 1e-12)

    return best_coeffs, float(best_err), float(w)


# ───────────────────────────────────────────────────────────────────────
# 핵심 함수 3: Minimax Composite Polynomial 구성 (논문 Algorithm 6/7)
# ───────────────────────────────────────────────────────────────────────
def build_composite(epsilon, degrees, verbose=True):
    """
    논문 Algorithm 6 (MinimaxComp) 구현.
    
    각 step i:
      - p_1 : MP_{R_{ε,1}}^{d_1}     (domain [ε, 1])
      - p_i : MP_{R_{1-τ_{i-1}, 1+τ_{i-1}}}^{d_i}  (i ≥ 2)
      - τ_i : ME_{a_i, b_i}^{d_i}    (minimax 오차)
    
    Parameters
    ----------
    epsilon  : float  — 입력 정밀도 파라미터 (= 2^{-alpha})
    degrees  : list   — 각 step 의 홀수 차수 [d_1, d_2, ..., d_k]
    
    Returns
    -------
    component_list : list of (coeffs, w, a, b)  — 각 step 의 다항식 정보
    tau_list       : list of float               — 각 step 의 minimax 오차
    """
    component_list, tau_list = [], []
    a, b = float(epsilon), 1.0

    for step, d in enumerate(degrees):
        if verbose:
            print(f"  [Step {step+1}] degree={d:2d}, "
                  f"domain=[{a:.6f}, {b:.6f}]", end=" ... ")
        coeffs, err, w = remez_minimax(a, b, d)
        component_list.append((coeffs.copy(), w, a, b))
        tau_list.append(float(err))
        if verbose:
            print(f"ME = {err:.4e}")
        # 다음 step 의 domain: [1 - τ_i, 1 + τ_i]
        a = max(1.0 - err, 1e-12)
        b = 1.0 + err

    return component_list, tau_list


# ───────────────────────────────────────────────────────────────────────
# 핵심 함수 4: Composite Polynomial 안전 평가
# ───────────────────────────────────────────────────────────────────────
def eval_composite(x_val, component_list):
    """
    p_k ∘ ... ∘ p_1(x) 안전하게 평가.
    각 단계마다 다음 domain 범위로 soft-clip → 수치 발산 방지.
    
    Parameters
    ----------
    x_val          : float or array  — 입력값 (diff = u - v)
    component_list : list            — build_composite 의 반환값
    
    Returns
    -------
    ndarray  — 각 입력에 대한 sgn(x) 근사값 (≈ ±1)
    """
    x = np.atleast_1d(np.asarray(x_val, dtype=np.float64)).copy()

    for coeffs, w, a, b in component_list:
        # 다음 step domain 이탈 방지 (2% 마진)
        margin = (b - a) * 0.02
        x_in   = np.clip(x, -(b + margin), (b + margin))
        x      = eval_cheb_odd(coeffs, x_in, w=w, clip=True)
        # nan/inf 방어
        bad = ~np.isfinite(x)
        if np.any(bad):
            x[bad] = 0.0

    return x


# ───────────────────────────────────────────────────────────────────────
# 실험 함수
# ───────────────────────────────────────────────────────────────────────
def run_experiment(alpha, strategy, degrees, n_samples=2000):
    """
    alpha, strategy 에 따른 오차 실험.
    
    데이터 설계:
      - u, v ~ N(0.5, sigma=0.15) clipped to [0, 1]
      - → diff = u-v ∈ [-1, 1] 보장 (T_k 발산 방지)
      - → diff 가 0 근처에 집중 (비교가 어려운 케이스)
      - |u-v| >= epsilon 인 유효 샘플만 사용 (논문 조건)
    
    오차:
      comp_plain  = comp(u, v)         ∈ {0, 0.5, 1}  (exact)
      comp_approx = (p(u-v) + 1) / 2  ∈ ℝ             (approximate)
      error       = |comp_approx - comp_plain|
    """
    epsilon = float(2 ** (-alpha))
    target  = float(2 ** (1 - alpha))   # 논문 오차 보장: 2^{1-alpha}
    sigma   = 0.15

    print(f"\n{'='*65}")
    print(f"  alpha = {alpha},  strategy = {strategy}")
    print(f"  epsilon  = 2^(-{alpha}) = {epsilon:.4e}")
    print(f"  degrees  = {degrees}")
    print(f"  total depth = {sum(DEP[d] for d in degrees)}, "
          f"total mult = {sum(MULT[d] for d in degrees)}")
    print(f"  목표 오차  = 2^(1-{alpha}) = {target:.4e}")
    print(f"{'='*65}")

    # ── 1. Composite Polynomial 구성 ─────────────────────────────────
    print("\n[Composite Polynomial 구성]")
    component_list, tau_list = build_composite(epsilon, degrees, verbose=True)
    final_tau = tau_list[-1]
    ok = "✓ 달성" if final_tau <= target else "✗ 미달"
    print(f"\n  최종 tau_k = {final_tau:.6e}  (목표 {target:.4e})  → {ok}")

    # ── 2. 데이터 생성 ────────────────────────────────────────────────
    # u, v ∈ [0,1]: 논문 조건 (u,v ∈ [0,1] s.t. |u-v| ≥ ε)
    # N(0.5, 0.15) → diff = u-v 가 0 근처에 집중 (어려운 비교 케이스)
    np.random.seed(42)
    u_all = np.clip(np.random.normal(0.5, sigma, n_samples), 0.0, 1.0)
    v_all = np.clip(np.random.normal(0.5, sigma, n_samples), 0.0, 1.0)

    diff_all = np.abs(u_all - v_all)
    valid    = diff_all >= epsilon
    u, v     = u_all[valid], v_all[valid]
    n_valid  = len(u)

    print(f"\n[데이터 생성]")
    print(f"  분포     : N(0.5, sigma={sigma}) clipped to [0, 1]")
    print(f"  총 샘플  : {n_samples}")
    print(f"  유효 샘플: {n_valid} / {n_samples} "
          f"({100*n_valid/n_samples:.1f}%)  (|u-v| ≥ ε)")
    print(f"  |diff| 통계: "
          f"min={diff_all[valid].min():.4f}, "
          f"mean={diff_all[valid].mean():.4f}, "
          f"max={diff_all[valid].max():.4f}")

    # ── 3. 오차 계산 ─────────────────────────────────────────────────
    comp_plain  = np.where(u > v, 1.0, np.where(u < v, 0.0, 0.5))
    sgn_approx  = eval_composite(u - v, component_list)
    comp_approx = (sgn_approx + 1.0) / 2.0

    # nan/inf 제거
    fin = np.isfinite(comp_approx)
    if not np.all(fin):
        print(f"  [경고] nan/inf {int(np.sum(~fin))}개 제외")
        comp_plain  = comp_plain[fin]
        comp_approx = comp_approx[fin]
        u, v        = u[fin], v[fin]

    errors = np.abs(comp_approx - comp_plain)

    # ── 4. 결과 출력 ─────────────────────────────────────────────────
    print(f"\n[오차 통계]")
    print(f"  max    오차 : {errors.max():.6e}")
    print(f"  mean   오차 : {errors.mean():.6e}")
    print(f"  median 오차 : {np.median(errors):.6e}")
    print(f"  std    오차 : {errors.std():.6e}")

    n_exceed = int(np.sum(errors > epsilon))
    print(f"\n  2^(-{alpha}) = {epsilon:.2e} 초과 샘플: "
          f"{n_exceed} / {len(errors)} ({100*n_exceed/len(errors):.1f}%)")

    print(f"\n[오차 Percentile]")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  {p:3d}th : {np.percentile(errors, p):.4e}")

    print(f"\n[상세 — 처음 10개 유효 샘플]")
    print(f"  {'u':>8}  {'v':>8}  {'u-v':>9}  "
          f"{'plain':>5}  {'approx':>9}  {'error':>11}")
    print(f"  {'-'*56}")
    for i in range(min(10, len(u))):
        print(f"  {u[i]:>8.5f}  {v[i]:>8.5f}  {u[i]-v[i]:>9.5f}  "
              f"{comp_plain[i]:>5.1f}  {comp_approx[i]:>9.5f}  "
              f"{errors[i]:>11.4e}")

    return errors


# ───────────────────────────────────────────────────────────────────────
# 메인: alpha = 8, 12, 16  ×  (minimize_mult, minimize_depth)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ALPHAS = [8, 12, 16]

    for alpha in ALPHAS:
        degs_mm = TABLE2[alpha][0]   # minimize multiplications
        degs_md = TABLE2[alpha][1]   # minimize depth

        for strategy, degrees in [("minimize_mult",  degs_mm),
                                   ("minimize_depth", degs_md)]:
            print(f"\n{'#'*65}")
            print(f"# alpha={alpha},  {strategy}")
            print(f"# degrees = {degrees}")
            print(f"{'#'*65}")

            run_experiment(
                alpha     = alpha,
                strategy  = strategy,
                degrees   = degrees,
                n_samples = 2000,
            )
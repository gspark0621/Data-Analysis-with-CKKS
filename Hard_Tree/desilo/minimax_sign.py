# soft_tree/desilo/minimax_sign.py
"""
Minimax composite polynomial 로 sign(x) 근사.

 근거:  Lee, Lee, No, Kim,
        "Minimax Approximation of Sign Function by Composite Polynomial
         for Homomorphic Comparison"
        (Cheon et al. Asiacrypt'20 의 후속. 동일 α 에서 depth 최대 48% 감소)

 [핵심]  p = p_k ∘ ... ∘ p_1,  각 p_i 는 직전 출력 구간에서의 minimax 근사.
         p_1 : [ε, 1]           → [1−τ_1, 1+τ_1]
         p_i : [1−τ_{i−1}, 1+τ_{i−1}] → [1−τ_i, 1+τ_i]
         τ_k ≤ 2^{−(α−1)} 이면 (α−1, ε)-close.

 [우리 상황이 유리한 이유]
         입력을 bin 으로 이산화하고 θ 를 bin 경계에 두면
              |x − θ| ≥ 1/(2·n_max)      →  ε 가 크게 잡힌다.
         일반적인 ε=2^-20 이면 차수 {29,31,31,31,31} (depth 25) 인데,
         ε≈2^-5 이면 {15,15} (depth 8) 로 충분하다.

 [스케일 값 w]  논문 §2.4·§5.2.1 — 파워 기저는 계수가 불안정하므로 체비셰프 기저.
         도메인이 [-1,1] 을 벗어나면 T̃_i(x)=T_i(x/w) 를 쓴다.
         §5.2.2 의 요령: 1/w_{i+1} 을 p_i 의 계수에 미리 곱해 두면
         추가 깊이 소모가 없다.  ★ 이 파일은 그 방식을 쓴다.

 [마진 η]  Algorithm 7 — CKKS 오차로 중간 구간을 벗어나는 것을 막기 위해
         각 단계의 τ 에 η 를 더해 다음 성분의 도메인을 넓게 잡는다.
"""
from __future__ import annotations
import math
import numpy as np
from numpy.polynomial import chebyshev as C



try:
    from desilofhe import Engine, Ciphertext
except Exception:
    Engine = Ciphertext = object


# ══════════════════════════════════════════════════════════════════
# Appendix E — α 별 (depth, #mults, degrees).  ε = 2^-α 기준
# ══════════════════════════════════════════════════════════════════
APPENDIX_E = {
    4:  [(5, 10, [27]), (6, 8, [5, 7]), (7, 7, [3, 3, 5])],
    5:  [(7, 12, [7, 13]), (8, 10, [3, 5, 7]), (9, 9, [5, 5, 5])],
    6:  [(8, 16, [15, 15]), (9, 14, [3, 7, 13]), (10, 12, [5, 5, 11]),
         (11, 11, [3, 5, 5, 5])],
    7:  [(10, 17, [7, 7, 13]), (11, 15, [3, 5, 7, 7]), (12, 14, [5, 5, 5, 7]),
         (13, 13, [3, 3, 5, 5, 5])],
    8:  [(11, 21, [7, 15, 15]), (12, 19, [3, 7, 7, 13]), (13, 17, [5, 5, 7, 11]),
         (14, 15, [3, 3, 5, 5, 9])],
}

# Table 1 — 홀수차 다항식 평가 비용 (참고용)
PAPER_DEP = {d: (2 if d <= 3 else 3 if d <= 7 else 4 if d <= 15 else 5)
             for d in range(3, 33, 2)}


# DesiloFHE 실측: 성분 다항식 하나당 논문 dep + 1
DESILO_OVERHEAD = 1

# ── 실현 가능한 α 의 상한 ──
#   성분 사이 표준 bootstrap(small_bootstrap_key) 이 레벨을 회복하므로
#   **한 번에 필요한 깊이 = 가장 깊은 성분 하나** (deg 15 이면 6).
#   → α 에 레벨 천장이 없다.  α ≥ 14 는 Remez 수렴 안 함.
ALPHA_MAX_FRESH = 12
ALPHA_MAX_AFTER_REFRESH = 4


def max_alpha(start_level, l_fresh=26):
    """시작 레벨에서 감당 가능한 최대 α (경험적 — build_composite 로 확인 권장)."""
    for a in range(12, 3, -1):
        try:
            spec = build_composite(a, 2.0 ** (-a), "depth")
            if sign_depth(spec) <= start_level - 2:
                return a
        except Exception:
            continue
    return 4


BOOT_MIN_IN = 3     # sign_bootstrap 입력 레벨 하한 (= 최소 stage_count)


def sign_depth(spec, to_01: bool = True, boot_stage: int = None) -> int:
    """
    **입력에 필요한** 레벨.

    성분 사이 표준 bootstrap 이 레벨을 회복하므로 한 번에 필요한 건
    가장 깊은 성분 하나뿐이다.  다만 마지막 성분 뒤에 sign_bootstrap 이
    오므로 그 입력 레벨(stage_count)도 남아 있어야 한다.

      필요 = max(성분 깊이) + BOOT_MIN_IN
    ★ stage_count 는 남은 레벨에 맞춰 자동으로 낮춘다(5→4→3)므로
      하한 BOOT_MIN_IN(=3) 만 확보하면 된다.
    ★ to_01 은 마지막 부트 **이후**라 입력 요구에 안 들어간다
      (출력 레벨은 sign_out_level 참고).
    """
    if spec.get("inter") == "sign":                 # build_composite_sb: 첫 부트까지의 깊이 + 부트 입력
        need = 0
        for d, tr in zip(spec["degs"], spec.get("trans", ["S%d" % s for s in spec["stages"]])):
            need += cheb_dep(d)
            if tr == "R":
                return need
            if tr[0] == "S":
                return need + int(tr[1])
        return need
    per = [PAPER_DEP[d] + DESILO_OVERHEAD for d in spec["degs"]]
    return max(per) + (boot_stage if boot_stage else BOOT_MIN_IN)


def alpha_for_margin(n_max: int) -> int:
    """bin 경계 임계값이 주는 마진 ε = 1/(2·n_max) → 필요한 α."""
    return max(4, min(8, math.ceil(-math.log2(1.0 / (2.0 * n_max)))))


def pick_degrees(alpha: int, mode: str = "depth"):
    """
    mode='depth' 깊이 최소 / 'mult' 곱셈 최소.

    ★ Appendix E 는 α ≤ 8 까지만 있다.  더 큰 α 는 성분을 덧붙여 확장한다.
      각 성분이 τ 를 크게 줄이므로, deg 15 를 필요한 만큼 추가하면 된다.
      (분할 선택 argmax 는 공개 타이브레이커를 써도 α≈12 가 필요하다 —
       추론 비교(α=4~8)와 달리 마진이 binning 으로 보장되지 않기 때문)
    """
    a = max(4, alpha)
    if a <= 8:
        rows = APPENDIX_E[a]
        return min(rows, key=lambda r: (r[0], r[1])) if mode == "depth" \
            else min(rows, key=lambda r: (r[1], r[0]))
    base = pick_degrees(8, mode)
    extra = -(-(a - 8) // 3)                      # 성분 하나가 α 를 ~3 올린다
    degs = list(base[2]) + [15] * extra
    dep = sum(PAPER_DEP[g] for g in degs)
    mul = base[1] + 8 * extra
    return (dep, mul, degs)


# ══════════════════════════════════════════════════════════════════
# 성분 다항식 — 다구간 Remez (홀수차, 체비셰프 기저)
# ══════════════════════════════════════════════════════════════════
def minimax_odd(a: float, b: float, deg: int, w: float = 1.0,
                n_grid: int = 8000, iters: int = 60, tol: float = 1e-10):
    """
    sgn(x) 를 [-b,-a] ∪ [a,b] 에서 근사하는 홀수차 다항식 (차수 ≤ deg).

    sgn 이 기함수이고 성분도 홀수차만 쓰므로, [a,b] 에서 상수 1 을
    근사하는 문제로 축약된다 (논문 §3.1).

    기저:  T_{2j+1}(x/w)      j = 0 … m,   m = (deg−1)//2
    반환:  (coeffs, tau)
           coeffs : 길이 deg+1 의 체비셰프 계수 (T_i(x/w) 기준, 짝수차는 0)
           tau    : minimax 근사 오차
    """
    assert deg % 2 == 1, "홀수차만"
    m = (deg - 1) // 2
    npts = m + 2                                   # 등진동 점 개수

    # 초기 참조점 — 체비셰프 노드
    k = np.arange(npts)
    x = a + (b - a) * (1 - np.cos(np.pi * k / (npts - 1))) / 2.0
    grid = np.linspace(a, b, n_grid)

    def basis(xs):
        """(len(xs), m+1) 행렬,  열 j = T_{2j+1}(xs/w)"""
        t = np.asarray(xs, float) / w
        B = np.empty((len(t), m + 1))
        for j in range(m + 1):
            cc = np.zeros(2 * j + 2); cc[2 * j + 1] = 1.0
            B[:, j] = C.chebval(t, cc)
        return B

    c = None
    tau = 1.0
    for _ in range(iters):
        A = np.zeros((npts, npts))
        A[:, :m + 1] = basis(x)
        A[:, m + 1] = -np.power(-1.0, np.arange(npts))
        try:
            sol = np.linalg.solve(A, np.ones(npts))
        except np.linalg.LinAlgError:
            break
        c, E = sol[:m + 1], sol[m + 1]

        err = basis(grid) @ c - 1.0
        # 극점 후보 = 부호가 바뀌는 구간마다의 최대 |err| 위치 + 양 끝
        s = np.sign(err)
        cuts = [0] + list(np.where(np.diff(s) != 0)[0] + 1) + [n_grid]
        cand = []
        for i in range(len(cuts) - 1):
            seg = slice(cuts[i], cuts[i + 1])
            if cuts[i + 1] - cuts[i] < 1:
                continue
            loc = cuts[i] + int(np.argmax(np.abs(err[seg])))
            cand.append(loc)
        cand = sorted(set(cand + [0, n_grid - 1]))
        if len(cand) < npts:
            break
        # 부호가 교대하는 npts 개를 |err| 큰 순으로
        cand = sorted(cand, key=lambda i: -abs(err[i]))[:npts]
        cand = sorted(cand)
        newx = grid[cand]
        if np.max(np.abs(newx - x)) < tol:
            x = newx
            break
        x = newx

    err = basis(grid) @ c - 1.0
    tau = float(np.max(np.abs(err)))
    full = np.zeros(deg + 1)
    for j in range(m + 1):
        full[2 * j + 1] = c[j]
    return full, tau


# ══════════════════════════════════════════════════════════════════
# Improved Remez (Lee et al., Eurocrypt'21 Alg.2·3 — Minimax sign 논문 §2.4 가 쓰는 알고리즘)
# ══════════════════════════════════════════════════════════════════
#   minimax_odd 와 같은 문제 (홀수 기저 T_{2j+1} 로 [a,b] 에서 1 근사 ≡ [−b,−a]∪[a,b] 에서 sgn) 를
#   논문 방식으로 푼다.  minimax_odd 의 문제:
#     · 극점을 균등 8000점 격자에서 찾음  → a 가 작으면 a 근처 극점을 놓침
#     · 부호 교대를 확인하지 않고 |err| 큰 순으로 고름  → 발산·특이 행렬 (a ≤ 2^-24 에서 예외)
#   여기서는
#     ① 기하 + 균등 혼합 격자로 극점 후보를 잡고, 각 극점을 이분 탐색으로 정밀화 (Alg.2 극점 탐색)
#     ② 국소 극값 조건 |r| ≥ |E| + 교대 조건 + 최대 절대합 조건으로 n+1 개 선택 (Alg.3)
#     ③ 수렴 판정 (max−min)/min < γ,  반환 오차는 정밀화된 극점에서의 최대 |r| (= 참 sup)
_ODD_BASIS_CACHE = {}


def _odd_basis(xs, m):
    """(len(xs), m+1) 행렬,  열 j = T_{2j+1}(xs).  삼각함수 항등식 대신 점화식 (|x|≤1)."""
    x = np.asarray(xs, float)
    B = np.empty((len(x), m + 1))
    t_prev, t_cur = np.ones_like(x), x.copy()          # T0, T1
    B[:, 0] = t_cur
    for k in range(2, 2 * m + 2):
        t_prev, t_cur = t_cur, 2.0 * x * t_cur - t_prev
        if k % 2 == 1:
            B[:, (k - 1) // 2] = t_cur
    return B


def _select_alternating(xs, rs, n_pts):
    """
    Alg.3 — 후보 극점 (정렬됨) 에서 교대 조건 + 최대 절대합 조건을 만족하는 n_pts 개.
      1) 같은 부호가 연속이면 |r| 작은 쪽 제거
      2) 남은 수가 n_pts 보다 많으면:
           n_pts+1 개  → 양 끝 중 |r| 작은 쪽 제거
           그 이상     → 인접 쌍 합 |r_i|+|r_{i+1}| 이 최소인 쌍 제거
                          (최소 쌍이 끝점을 포함하면 그 끝점만 제거)
    """
    xs, rs = list(xs), list(rs)
    i = 0
    while i < len(xs) - 1:
        if np.sign(rs[i]) == np.sign(rs[i + 1]):
            j = i if abs(rs[i]) < abs(rs[i + 1]) else i + 1
            del xs[j], rs[j]
            i = max(i - 1, 0)
        else:
            i += 1
    while len(xs) > n_pts:
        if len(xs) == n_pts + 1:
            j = 0 if abs(rs[0]) < abs(rs[-1]) else len(xs) - 1
            del xs[j], rs[j]
            continue
        sums = [abs(rs[k]) + abs(rs[k + 1]) for k in range(len(xs) - 1)]
        k = int(np.argmin(sums))
        if k == 0:
            del xs[0], rs[0]
        elif k == len(xs) - 2:
            del xs[-1], rs[-1]
        else:
            del xs[k:k + 2], rs[k:k + 2]
    return np.array(xs), np.array(rs)


def remez_odd(a: float, b: float, deg: int, gamma: float = 1e-7, iters: int = 80,
              n_geo: int = 6000, n_lin: int = 6000, refine: int = 40):
    """
    [a, b] (0<a<b≤1) 에서 1 을 근사하는 홀수차 minimax (≡ [−b,−a]∪[a,b] 에서 sgn).
    반환: (coeffs, tau)   coeffs 는 길이 deg+1 의 체비셰프 계수 (짝수차 0), tau 는 참 minimax 오차.
    실패하면 RuntimeError.
    """
    assert deg % 2 == 1 and 0 < a < b <= 1.0
    m = (deg - 1) // 2
    n_pts = m + 2
    # 극점 후보 격자 — 작은 a 근처는 기하 간격, 나머지는 균등 간격
    n_lin = max(n_lin, 80 * deg)                         # 1 근처 극점 간격 ~ (π/deg)²/2
    grid = np.unique(np.concatenate([np.geomspace(a, b, n_geo), np.linspace(a, b, n_lin)]))
    Bg = _odd_basis(grid, m)
    # 초기 참조점 — x = a + (b−a)·sin(πs/2)  (s 균등).
    #   ★ 홀수 기저 T_{2j+1} 는 [−1,1] 전체의 기저라 [a,b] 양 끝에 몰린 체비셰프 노드는
    #     cond 1e11~1e17 (deg 63/127) 로 첫 해가 발산한다.  기하 간격은 0 근처에서 기저가
    #     거의 선형 종속 (모두 ≈ (2j+1)x).  sin 배치는 1 쪽에만 몰려 cond ≈ 10 (실측).
    k = np.arange(n_pts)
    x = a + (b - a) * np.sin(np.pi * k / (2.0 * (n_pts - 1)))
    sgn = np.power(-1.0, np.arange(n_pts))
    c = None
    for _ in range(iters):
        A = np.empty((n_pts, n_pts))
        A[:, :m + 1] = _odd_basis(x, m)
        A[:, m + 1] = -sgn                                # p(x_i) − 1 = s_i · E
        try:
            sol = np.linalg.solve(A, np.ones(n_pts))
        except np.linalg.LinAlgError:
            raise RuntimeError(f"remez_odd: 특이 행렬 (a={a:.3e}, deg={deg})")
        c, E = sol[:m + 1], abs(sol[m + 1])
        r = Bg @ c - 1.0
        # 극점 후보: 격자 국소 극값 + 양 끝
        ar = np.abs(r)
        inner = np.where((ar[1:-1] >= ar[:-2]) & (ar[1:-1] >= ar[2:]) &
                         (ar[1:-1] >= 0.5 * E))[0] + 1        # |r| 가 E 에 한참 못 미치면 극점 후보 아님
        # 내부 극점 정밀화 — 3분 탐색을 모든 후보에 대해 벡터로 (Alg.2 극점 탐색)
        lo, hi = grid[inner - 1].copy(), grid[inner + 1].copy()
        for _ in range(refine):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            f1 = np.abs(_odd_basis(m1, m) @ c - 1.0)
            f2 = np.abs(_odd_basis(m2, m) @ c - 1.0)
            left = f1 < f2
            lo = np.where(left, m1, lo)
            hi = np.where(left, hi, m2)
        cx = np.concatenate([[grid[0]], 0.5 * (lo + hi), [grid[-1]]])
        cr = _odd_basis(cx, m) @ c - 1.0
        keep = np.abs(cr) >= E * (1 - 1e-9)                 # 국소 극값 조건
        cx, cr = cx[keep], cr[keep]
        order = np.argsort(cx)
        cx, cr = cx[order], cr[order]
        if len(cx) < n_pts:
            raise RuntimeError(f"remez_odd: 극점 부족 {len(cx)}<{n_pts} (a={a:.3e}, deg={deg})")
        x, rx = _select_alternating(cx, cr, n_pts)
        if len(x) < n_pts:
            raise RuntimeError(f"remez_odd: 교대 극점 부족 (a={a:.3e}, deg={deg})")
        sgn = np.sign(rx)                                # 다음 단계는 이 부호 패턴으로 등진동
        emax, emin = np.max(np.abs(rx)), np.min(np.abs(rx))
        if emax < 1e-13 or (emax - emin) < gamma * emin:   # 오차가 double 한계면 그대로 종료
            break
    tau = float(np.max(np.abs(cr)))                     # 정밀화된 모든 극점 중 최대 = 참 sup
    full = np.zeros(deg + 1)
    full[1::2] = c
    return full, tau


_DP_CACHE = {}


def build_composite(alpha: int, epsilon: float, mode: str = "depth",
                    eta: float = None, verbose: bool = False):
    """
    성분 다항식 집합 생성.

    반환: dict(
        comps  = [계수 배열, ...]   ← 이대로 순차 평가하면 됨
        degs, taus, scales, depth_paper, mult_paper )

    ★ 스케일 요령(§5.2.2): 다음 성분의 1/w 를 현재 성분의 계수에 미리 곱한다.
      마지막 성분에는 아무것도 곱하지 않는다 (출력이 ±1 이어야 하므로).

    ★ 성분 사이에는 **표준 bootstrap(값 보존)** 만 들어가므로 이 폴딩이 그대로
      성립한다.  sign_bootstrap 을 끼우면 z 가 ±1 로 바뀌어 폴딩 전제가 깨진다
      (증상: 성분 수가 짝수일 때만 실패) — 그래서 마지막에 한 번만 건다.
    """
    dep_p, mul_p, degs = pick_degrees(alpha, mode)
    if eta is None:
        eta = 2.0 ** (-(alpha + 4))                # 논문 Table 3 수준

    comps, taus, scales, raw, boots = [], [], [], [], []
    a, b = float(epsilon), 1.0
    prev_boot = False
    for i, deg in enumerate(degs):
        # ★ 직전이 부트였으면 입력이 이미 ±1 근처다 → w=1 (스케일 불필요).
        #   여기서 w=2 를 쓰면 chebval 이 T_j(z) 를 쓰는데 계수는 T_j(z/2) 기준이라
        #   완전히 다른 다항식이 된다 (검증 오차 87~670 의 원인).
        w = 1.0 if (prev_boot or b <= 1.0) else 2.0
        cf, tau = minimax_odd(a, b, deg, w=w)
        tau = min(0.999, tau + eta)                # 마진
        raw.append((cf, w)); taus.append(tau); scales.append(w)
        # ★ 부트는 τ 를 **줄일 때만** 건다.  (π²/8)τ² < τ  ⟺  τ < 8/π² = 0.811
        do_boot = False    # 성분 사이 부트 없음 — 표준 bootstrap 사용
        boots.append(do_boot)
        if do_boot:
            tau = BOOT_SQ * tau * tau              # Hong et al. Thm 1
        prev_boot = do_boot
        a, b = 1.0 - tau, 1.0 + tau

    # 스케일 폴딩: 부트를 **거는** 구간에서는 하지 않는다 (부트 출력이 ±1 근처라
    # 다음 성분의 w 로 나눌 필요가 없다).  거는 곳이 아니면 기존대로 접어넣는다.
    for i, (cf, w) in enumerate(raw):
        if boots[i]:
            comps.append(cf)
        else:
            nxt_w = scales[i + 1] if i + 1 < len(raw) else 1.0
            comps.append(cf / nxt_w)

    if verbose:
        print(f"  α={alpha}  ε={epsilon:.5f}  degs={degs}  "
              f"paper depth={dep_p}, mult={mul_p}")
        for i, (d, t, w) in enumerate(zip(degs, taus, scales)):
            print(f"    p{i+1}: deg={d:>2}  w={w}  τ={t:.3e}")
    return dict(comps=comps, degs=degs, taus=taus, scales=scales,
                depth_paper=dep_p, mult_paper=mul_p, alpha=alpha, epsilon=epsilon)


# ══════════════════════════════════════════════════════════════════
# 평문 평가 (검증용)
# ══════════════════════════════════════════════════════════════════
def eval_composite_plain(x, spec):
    """
    CKKS 없이 합성 다항식을 평가.
    성분 사이는 표준 bootstrap(값 보존)이라 평문에서는 그냥 이어 평가하면 된다.
    """
    # ★ 1/w 는 이미 직전 성분의 계수에 접혀 있으므로 여기서 다시 나누지 않는다.
    #   (w_1 = 1 이라 첫 성분도 그대로)
    z = np.asarray(x, float)
    for cf in spec["comps"]:
        z = C.chebval(z, cf)
    return z


def verify_plain(spec, n=200000, verbose=True):
    """|x| ≥ ε 에서 |p(x) − sgn(x)| ≤ 2^{-(α-1)} 인지 확인."""
    eps, alpha = spec["epsilon"], spec["alpha"]
    x = np.concatenate([np.linspace(-1.0, -eps, n // 2),
                        np.linspace(eps, 1.0, n // 2)])
    p = eval_composite_plain(x, spec)
    err = np.abs(p - np.sign(x)).max()
    bound = 2.0 ** (-(alpha - 1))
    ok = err <= bound
    if verbose:
        print(f"    최대 오차 {err:.3e}   허용 2^-(α-1) = {bound:.3e}   "
              f"→ {'OK' if ok else 'FAIL'}")
    return err, bound, ok


# ══════════════════════════════════════════════════════════════════
# DP — 성분 뒤 전이(없음 / refresh / sign_bootstrap)까지 함께 고른다
# ══════════════════════════════════════════════════════════════════
#   논문 DP(Lee et al.) 는 성분 사이 bootstrap 이 없다고 가정하고 깊이·곱셈을 최소화한다.
#   우리는 Desilo 의 bootstrap 비용이 지배적이므로 **시간(bootstrap 합 + 평가)** 을 최소화하고,
#   각 성분 뒤에 올 전이를 DP 가 같이 고른다 (DBSCAN minimax.py "레버 A" 의 확장).
#
#   상태:  [a, 1] 위에서 1 을 근사할 차례 (홀수 다항식이므로 [−1,−a] 도 대칭) + 남은 레벨 L
#   성분 d:  minimax_unit(d, a) → 출력 ∈ [1−t, 1+t],  dep(d) 레벨 소모
#   전이 ─ '-'   부트 없이 다음 성분.   스케일 폴딩(계수 × 1/(1+t), Lee Lemma 3)
#                → [ (1−t)/(1+t), 1 ],  L ← L − dep
#        ─ 'R'   refresh (값 보존 bootstrap).  스케일 폴딩 후 refresh
#                → [ (1−t)/(1+t) − REFRESH_FLOOR, 1 ],  L ← 10,  +BOOT_MS
#        ─ 'Ss'  sign_bootstrap stage s = sin(πz/2).  t < TAU_SB_MAX (z<2 라야 부호 유지)
#                → [ cos(πt/2) − SB_FLOOR, 1 ],  L ← 16 − s,  +SIGNBOOT_MS[s]
#                ★ 출력 ≤ 1 이라 폴딩 불필요.  오차 t → (π²/8)t² 공짜 수축
#   목표:  마지막 전이가 sign_bootstrap 이고 1 − a ≤ τ_target
#   δ (dead zone) 와 τ (출력 정밀도) 를 분리해서 받는다 (DBSCAN 레버 B).
#
#   ★ 정밀도 한계 (A6000 실측, 2026-11-08):
#       fresh 2^-50,  곱셈 몇 번 후 2^-35,  refresh 1회 후 2^-18 (값 크기와 무관한 절대 오차)
#     → refresh 를 거친 입력에 δ < 2^-16 은 의미가 없다.  큰 α 는 refresh 전 높은 레벨에서,
#       sign_bootstrap 전용 계획(allow_refresh=False)으로 처리한다.
#   ★ FHE 검증 (test/test_dp_sign.py): sign_bootstrap ≡ sin(πz/2) on |z|≤1.98 (오차 ≤ 8.1e-7),
#     cheb_dep = 실측, 구판 계획 5개 τ 이내 · 시간 ±4%.  '-' 전이를 쓰는 계획은 test_dp_sign C2 로 확인.
SIGNBOOT_MS = {3: 2280.3, 4: 1461.4, 5: 1256.5}     # 실측 (README)
REFRESH_MS = 2814.4                                # refresh 1회 실측 (hard_core.BOOT_MS)
REFRESH_LEVEL = 10                                 # small-key refresh 회복 레벨
SB_FLOOR = 2.0 ** -20                              # sign_bootstrap 실측 최대오차 7.7e-7
REFRESH_FLOOR = 2.0 ** -17                         # refresh 실측 절대오차 3.6e-6 ~ 5e-6
TAU_SB_MAX = 0.98                                  # sign_bootstrap 입력 τ 상한 (z = 1+τ < 2)
#   첫 성분은 τ 가 0.9 를 넘는 게 보통이다 (Appendix E 도 동일).
#   sin(πu/2) ≥ u (u∈[0,1]) 이므로 sign_bootstrap 은 τ 가 커도 하한 a 를 손해 보지 않는다.
MUL_MS = 9.84                                      # ct·ct 곱 1회 (Model_process 실측)
DP_DEGREES = (3, 5, 7, 15, 31, 63, 127)              # 127 은 improved Remez 로 안정화 후 추가
MIN_LEVEL_AFTER = 2                                # '-' 전이 뒤 다음 성분이 쓸 최소 레벨


def cheb_dep(d: int) -> int:
    """odd 차수 d 의 체비셰프 평가 레벨 소모 (논문 Table 1 + Desilo 1, deg 3~63 실측 일치)."""
    return int(math.ceil(math.log2(d + 1))) + DESILO_OVERHEAD


CHEB_MS = {3: 12.4, 5: 23.6, 7: 38.7, 15: 76.3, 31: 156.9, 63: 302.2, 127: 730.0}
#   A6000 실측 (test_dp_sign B).  127 은 test_dp_sign2 의 계획 시간 차이로 역산
#   (곱셈 수 추정식은 127 을 290 ms 로 잡아 α20 L20 계획을 1.3 s 과소평가했다).


def cheb_eval_ms(d: int) -> float:
    """평가 시간 — 실측 표, 없으면 비스칼라 곱 ≈ 2√d + log2 d 추정."""
    if d in CHEB_MS:
        return CHEB_MS[d]
    return (2.0 * math.sqrt(d) + math.log2(d + 1)) * MUL_MS


def _snap(a: float) -> float:
    """
    a 를 **보수적으로**(더 어려운 쪽으로) 격자에 맞춘다 — Remez 캐시 적중률을 높이기 위함.
      a < 1/2 :  log2 a 를 1/64 bit 단위로 내림
      a ≥ 1/2 :  log2(1−a) 를 1/64 bit 단위로 올림
    DP 와 계수 생성이 같은 격자를 쓰므로 계획과 실제 계수가 어긋나지 않는다.
    """
    if a < 0.5:
        return 2.0 ** (math.floor(math.log2(a) * 64) / 64)
    return 1.0 - 2.0 ** (math.ceil(math.log2(1.0 - a) * 64) / 64)


_REMEZ_CACHE = {}


def _remez_unit(deg: int, a: float):
    """
    [a, 1] 에서 1 을 근사하는 홀수차 minimax (w=1).  (coeffs, err)

    ★ remez_odd (improved Remez) 사용.  구판 minimax_odd 는 작은 a 에서 극점을 놓쳐
      참 1−τ 가 **음수** (a 근처에서 다항식이 0 아래 → 부호 반전) 였다
      (예: deg 31, a=2^-24 에서 보고 3.1e-6 / 참 −3.7e-5).  remez_odd 는 deg 7~127,
      a=2^-4~2^-30 에서 보고 = 참 오차.  독립 격자 확인을 통과하지 못하면 inf (후보 제외).
    """
    key = (deg, a)
    if key not in _REMEZ_CACHE:
        try:
            cf, err = remez_odd(a, 1.0, deg)            # 참 minimax 오차 (정밀화된 극점 기준)
            # 안전 확인: 독립 격자에서 보고 오차를 넘지 않아야 한다
            grid = np.concatenate([np.geomspace(a, 1.0, 20000), np.linspace(a, 1.0, 20000)])
            with np.errstate(all="ignore"):
                chk = float(np.max(np.abs(C.chebval(grid, cf) - 1.0)))
            if not (np.isfinite(err) and chk <= err * (1 + 1e-6) + 1e-12):
                cf, err = None, float("inf")
        except Exception:
            cf, err = None, float("inf")
        _REMEZ_CACHE[key] = (cf, err)
    return _REMEZ_CACHE[key]


def _after_sb(t: float) -> float:
    """sign_bootstrap 통과 후 하한 a' = cos(πt/2) − floor."""
    return math.cos(math.pi * t / 2.0) - SB_FLOOR


def _dp_key(L: int, a: float, last_sb: bool):
    la = math.log2(a) if a < 0.5 else 1000 + math.log2(1.0 - a)
    return (L, round(la * 16), last_sb)


def dp_degrees(delta: float, tau_target: float, level_in: int = 10,
               eta: float = None, max_comps: int = 8, degrees=DP_DEGREES,
               stages=(5, 4, 3), allow_refresh: bool = True):
    """
    δ (dead zone), τ_target (최종 출력 오차) 를 만족하는 최소 시간 계획.

    best-first 탐색 (비용이 단조 증가하므로 처음 목표에 닿은 해가 최적).
    (L, a 격자, 마지막이 sign_bootstrap 인지) 가 같은 상태는 먼저 도달한 것만 펼친다.
    반환: dict(degs, trans, stages, cost_ms, tau_final, taus, a_list, ...) 또는 None.
      trans[i] ∈ {'-', 'R', 'S3','S4','S5'}
    """
    import heapq
    if eta is None:
        eta = 2.0 ** -(int(round(-math.log2(delta))) + 4)   # CKKS 잡음 마진 (Lee Alg.7 η)
    a0 = _snap(float(delta))
    heap = [(0.0, 0, (), (), (), (a0,), int(level_in), False)]
    seen = set()
    uid = 1
    while heap:
        cost, _, degs, trans, taus, a_list, L, last_sb = heapq.heappop(heap)
        a = a_list[-1]
        if last_sb and 1.0 - a <= tau_target:
            sts = [int(x[1]) for x in trans if x[0] == "S"]
            return dict(degs=list(degs), trans=list(trans), stages=sts, cost_ms=cost,
                        tau_final=1.0 - a, taus=list(taus), a_list=list(a_list), eta=eta,
                        delta=delta, tau_target=tau_target, level_in=level_in,
                        allow_refresh=allow_refresh)
        k = _dp_key(L, a, last_sb)
        if k in seen:
            continue
        seen.add(k)
        if len(degs) >= max_comps:
            continue
        for d in degrees:
            dep = cheb_dep(d)
            if dep > L:
                continue
            _, err = _remez_unit(d, a)
            t = err + eta
            if not math.isfinite(t) or t >= 1.0:
                continue
            c0 = cost + cheb_eval_ms(d)
            a_fold = (1.0 - t) / (1.0 + t)
            nxt = []
            if L - dep >= MIN_LEVEL_AFTER:
                nxt.append((c0, "-", _snap(a_fold), L - dep, False))
            if allow_refresh and L - dep >= 1:
                af = a_fold - REFRESH_FLOOR
                if af > 0:
                    nxt.append((c0 + REFRESH_MS, "R", _snap(af), REFRESH_LEVEL, False))
            if t < TAU_SB_MAX:
                for s in stages:
                    if dep <= L - s:
                        nxt.append((c0 + SIGNBOOT_MS[s], f"S{s}", _snap(_after_sb(t)), 16 - s, True))
            for c2, tr, a2, L2, sb in nxt:
                heapq.heappush(heap, (c2, uid, degs + (d,), trans + (tr,), taus + (t,),
                                      a_list + (a2,), L2, sb))
                uid += 1
    return None


def build_composite_sb(delta: float, tau_target: float, level_in: int = 10,
                       verbose: bool = False, **kw):
    """
    dp_degrees 결과로 성분 계수를 만든다.  spec["inter"] = "sign".
      comps[i] 는 [a_i, 1] 위의 체비셰프 계수 (w=1).
      trans[i] 가 '-' 또는 'R' 이면 계수에 1/(1+t_i) 를 미리 곱해 둔다 (스케일 폴딩).
    """
    plan = dp_degrees(delta, tau_target, level_in=level_in, **kw)
    if plan is None:
        raise RuntimeError(f"DP 해 없음: δ={delta:.3e}, τ={tau_target:.3e}, L_in={level_in}")
    comps = []
    for d, tr, t, a in zip(plan["degs"], plan["trans"], plan["taus"], plan["a_list"]):
        cf, _ = _remez_unit(d, a)
        cf = np.asarray(cf, float)
        if tr in ("-", "R"):
            cf = cf / (1.0 + t)
        comps.append(cf)
    alpha = int(round(-math.log2(delta)))
    spec = dict(comps=comps, degs=plan["degs"], trans=plan["trans"], stages=plan["stages"],
                taus=plan["taus"], scales=[1.0] * len(comps), inter="sign",
                alpha=alpha, epsilon=delta, tau_target=tau_target,
                cost_ms=plan["cost_ms"], tau_final=plan["tau_final"],
                level_in=level_in, eta=plan["eta"])
    if verbose:
        print(f"  δ=2^{math.log2(delta):.1f}  τ≤2^{math.log2(tau_target):.1f}  L_in={level_in}"
              f"  → {plan_str(spec)}  {plan['cost_ms']/1000:.2f}s"
              f"  τ_final=2^{math.log2(max(plan['tau_final'], 1e-300)):.1f}")
    return spec


def plan_str(spec) -> str:
    """'15- 15R 31S4 63S5' 형식 (차수 + 뒤 전이)."""
    return " ".join(f"{d}{t}" for d, t in zip(spec["degs"], spec["trans"]))


def eval_composite_plain_sb(x, spec):
    """성분 → 전이 ('-'/'R': 그대로, 'S': sin(πz/2)) 를 차례로 평가 (FHE 평문 모델)."""
    z = np.asarray(x, float)
    for cf, tr in zip(spec["comps"], spec.get("trans", ["S"] * len(spec["comps"]))):
        z = C.chebval(z, cf)
        if tr[0] == "S":
            z = np.sin(np.pi * z / 2.0)
    return z


def verify_plain_sb(spec, n=200000, verbose=True):
    """|x| ≥ δ 에서 |p(x) − sgn(x)| ≤ τ_target 인지 (sign_bootstrap 모델 포함)."""
    eps = spec["epsilon"]
    x = np.concatenate([np.linspace(-1.0, -eps, n // 2), np.linspace(eps, 1.0, n // 2)])
    err = float(np.abs(eval_composite_plain_sb(x, spec) - np.sign(x)).max())
    ok = err <= spec["tau_target"]
    if verbose:
        print(f"    평문 검증  최대 오차 {err:.3e}  허용 {spec['tau_target']:.3e}  → {'OK' if ok else 'FAIL'}")
    return err, ok


def legacy_cost_ms(alpha: int) -> tuple:
    """현재 경로 (pick_degrees + 성분 사이 표준 refresh + 마지막 sign_bootstrap 5) 의 비용."""
    from Hard_Tree.desilo.hard_core import BOOT_MS
    _, _, degs = pick_degrees(alpha, "depth")
    ms = sum(cheb_eval_ms(d) for d in degs) + (len(degs) - 1) * BOOT_MS + SIGNBOOT_MS[5]
    return degs, ms


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# CKKS 평가
# ══════════════════════════════════════════════════════════════════
def to01(engine, ct):
    """±1 → {0,1}.  (z+1)/2.  깊이 1."""
    half = engine.encode([0.5] * engine.slot_count)
    return engine.add(engine.multiply(ct, half), half)


def minimax_sign01(engine, ct, kp, alpha: int = 6, epsilon: float = None,
                   n_max: int = None, mode: str = "depth",
                   spec=None, verbose: bool = False, to_01: bool = True,
                   final_boot: bool = True, boot_stage: int = None):
    """
    minimax 합성으로 sign 근사.  |x| ≥ ε 에서 ±1, to_01 이면 {0,1}.

    ── 부트 구조 (확정) ─────────────────────────────────────────────
      성분 사이 :  **표준 bootstrap (small_bootstrap_key)**
                   논문(Appendix E) 차수표가 가정하는 순수 합성을 그대로 유지.
                   sign_bootstrap 을 끼우면 스케일 폴딩 충돌 · τ 조건부 ·
                   도메인 재설계가 줄줄이 얽혀 이득(곱셈 1~2) 대비 복잡도가 컸다.
      마지막    :  **sign_bootstrap**
                   합성이 끝나 값이 이미 ±1 근처다.  여기서
                     ① ±1 로 더 밀어주고  ② 레벨을 회복한다 (16 − stage_count).
    ────────────────────────────────────────────────────────────────

    입력 필요 레벨 = sign_depth(spec)  (성분 하나 + 마지막 부트 입력분)
    출력 레벨      = sign_out_level(...)  — 마지막 부트가 정한다
    """
    if spec is None:
        if epsilon is None:
            epsilon = 1.0 / (2.0 * n_max) if n_max else 2.0 ** (-alpha)
        spec = build_composite(alpha, epsilon, mode, verbose=verbose)

    from Hard_Tree.desilo.util.keypack import refresh as _std_boot, sign_refresh
    rlk = kp.relinearization_key
    z = engine.intt(ct)                          # 체비셰프 평가는 non-NTT 입력
    ncomp = len(spec["comps"])
    if spec.get("inter") == "sign":
        # ★ build_composite_sb 경로 — 성분마다 DP 가 정한 전이:
        #     '-'  부트 없음 (계수에 스케일 폴딩 반영됨)   'R'  refresh   'Ss'  sign_bootstrap(s)
        #   마지막 전이도 DP 계획에 포함되어 있으므로 final_boot 는 무시한다.
        #   FHE 검증: test/test_dp_sign.py
        trans = spec.get("trans", ["S%d" % s for s in spec["stages"]])
        for cf, tr in zip(spec["comps"], trans):
            z = engine.evaluate_chebyshev_polynomial(z, list(cf), rlk)
            if tr == "R":
                z = _std_boot(engine, z, kp)
            elif tr[0] == "S":
                z = sign_refresh(engine, z, kp, stage_count=int(tr[1]))
            try:
                z = engine.intt(z)
            except Exception:
                pass
        return to01(engine, z) if to_01 else z
    for i, cf in enumerate(spec["comps"]):
        z = engine.evaluate_chebyshev_polynomial(z, list(cf), rlk)
        if i + 1 < ncomp:
            z = _std_boot(engine, z, kp)         # ★ 성분 사이: 표준 bootstrap
            try:
                z = engine.intt(z)
            except Exception:
                pass
    if final_boot:
        # stage_count 는 남은 레벨에 맞춰 고른다 (회복 = 16 − stage).
        lv = getattr(z, "level", 99)
        st = boot_stage if boot_stage is not None else next(
            (c for c in (5, 4, 3) if lv >= c), None)
        if st is None or lv < st:
            raise RuntimeError(
                f"마지막 sign_bootstrap 불가: 레벨 {lv} < {st or BOOT_MIN_IN}. "
                f"시작 레벨을 높이거나 α 를 낮추세요.")
        z = sign_refresh(engine, z, kp, stage_count=st)
    return to01(engine, z) if to_01 else z


def sign_out_level(spec=None, start_level=None, to_01: bool = True) -> int:
    """
    minimax_sign01 출력 레벨.  마지막 sign_bootstrap 이 정한다 (회복 = 16 − stage).

    ★ minimax_sign01 은 남은 레벨이 허락하는 **가장 큰 stage** 를 고른다.
      stage 가 클수록 회복 레벨은 **낮다** (16−5=11 < 16−3=13).
      따라서 보수적 추정은 stage 5, 즉 회복 11 이다.
      (stage 3 을 가정하면 13 으로 과대평가해 후처리에서 레벨이 터진다)
    """
    if spec is not None and spec.get("inter") == "sign":
        return (16 - spec["stages"][-1]) - (1 if to_01 else 0)
    st = 5
    if spec is not None and start_level is not None:
        per = max(PAPER_DEP[d] + DESILO_OVERHEAD for d in spec["degs"])
        lv_last = (start_level if len(spec["degs"]) == 1 else 10) - per
        st = next((c for c in (5, 4, 3) if lv_last >= c), 3)
    return (16 - st) - (1 if to_01 else 0)


if __name__ == "__main__":
    print("=" * 74)
    print("minimax composite 계수 생성 · 평문 검증")
    print("=" * 74)
    for nm, n_max in [("Soybean", 7), ("Iris", 9), ("Wine/Breast", 11),
                      ("Cancer(Scott)", 18), ("Sepsis", 101)]:
        a = alpha_for_margin(n_max)
        eps = 1.0 / (2.0 * n_max)
        print(f"\n  ■ {nm}  n_max={n_max}")
        spec = build_composite(a, eps, "depth", verbose=True)
        verify_plain(spec)
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
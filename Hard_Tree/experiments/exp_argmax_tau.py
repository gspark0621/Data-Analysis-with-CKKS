# Hard_Tree/experiments/exp_argmax_tau.py
"""
평문 시뮬레이션 — argmax sign 의 출력 오차 τ 가 학습·추론에 주는 영향.

  exp_argmax_pipeline.py 는 sign01 을 clip(0.5 + z/2δ) 로 모델링했다
  (dead zone 밖은 정확한 0/1).  실제 DP 합성(build_composite_sb)은 dead zone 밖에서도
  최대 τ 의 오차가 남는다.  league 는 ns−1 개 인자를 곱하므로 τ 가 누적될 수 있다:
      승자     Π (1 − ε_j)       ≈ 1 − Σ ε_j
      패자들   Σ_i Π_j b_ij      ≈ 패자마다 ~ε → 교차 정리의 누적합 c 를 밀어낸다
  → sign 을 **실제 다항식 체인 + sin(πz/2)** (FHE 와 2.5e-6 이내로 일치 확인) 으로 바꿔 측정.

  점수:  v = log2(max(raw/4n, 2^-40)) / 40   (exp_argmax_pipeline 의 log 변형)
  변형
    ideal    정확한 argmax
    clip     clip 모델 α=10  + clip 교차 정리 α_post=6       (이전 결과 재현)
    dp10     DP δ=2^-10 τ≤2^-10 L8 [15,5,7]   4.87 s
    dp8      DP δ=2^-10 τ≤2^-8  L8 [15,63]    3.82 s
    dp8c     dp8 + 인자마다 bit_cleaning h(b)=3b²−2b³ 1회 (+2 레벨, +곱 2)
  교차 정리 (DP 변형):  σ((c − τ_r)/K), K = 8 공개,  DP δ=2^-5 τ≤2^-6 L8 [3,7]  2.65 s
                       |c−τ_r| > K 는 다항식 정의역 밖 → 횟수를 센다
  추론 sign (DP 변형): DP δ=1/16 τ≤2^-6 L12 [31]

실행:  PYTHONPATH=/home/junhyung/study/Data-Analysis-with-CKKS \
       ~/.pyenv/versions/temp/bin/python exp_argmax_tau.py
"""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E
from Hard_Tree.desilo import minimax_sign as MS

D, T, LR = 6, 128, 0.3
DATA = ["phoneme", "spambase", "magic", "bank", "adult"]
SEEDS = range(3)
T_LOG = 40
K_POST = 8.0
VARIANTS = ["ideal", "clip", "dp10", "dp8", "dp8c"]

_SPECS = {}


def spec(key):
    if key not in _SPECS:
        args = {"dp10": (2.0 ** -10, 2.0 ** -10, 8), "dp8": (2.0 ** -10, 2.0 ** -8, 8),
                "post": (2.0 ** -5, 2.0 ** -6, 8), "inf": (1.0 / 16, 2.0 ** -6, 12)}[key]
        _SPECS[key] = MS.build_composite_sb(args[0], args[1], level_in=args[2])
    return _SPECS[key]


def clip01(z, delta):
    return np.clip(0.5 + z / (2.0 * delta), 0.0, 1.0)


def poly01(z, sp):
    return 0.5 * (MS.eval_composite_plain_sb(np.clip(z, -1.0, 1.0), sp) + 1.0)


def h(b):
    return 3 * b * b - 2 * b * b * b


def make_fns(var):
    """(league_sign, post_sign, infer_sign) — 모두 z → [0,1]."""
    if var == "clip":
        return (lambda z: clip01(z, 2.0 ** -10),
                lambda z: clip01(z, 2.0 ** -6 / K_POST),     # z=(c−τ)/K → 원래 α_post=6 과 동일
                lambda z: clip01(z, 1.0 / 16))
    key = "dp8" if var.startswith("dp8") else "dp10"
    sl, sp_, si = spec(key), spec("post"), spec("inf")
    lf = (lambda z: h(poly01(z, sl))) if var == "dp8c" else (lambda z: poly01(z, sl))
    return lf, (lambda z: poly01(z, sp_)), (lambda z: poly01(z, si))


def run(args):
    name, X, y, var, seed = args
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=seed)
    edges = E.fit_bins(Xtr, E.N_MAX)
    Btr, Bte = E.apply_bins(Xtr, edges), E.apply_bins(Xte, edges)
    n, d = Btr.shape
    nb, nmax = E.N_MAX, E.N_MAX
    L = 1 << D
    lam = max(1.0, n / L)
    rng = np.random.default_rng(1000 + seed)
    rng_tau = np.random.default_rng(5000 + seed)
    Ftr = np.full(n, ytr.mean()); Fte = np.full(len(Bte), ytr.mean())
    xte = (Bte + 0.5) / nmax
    m = min(d, E.MTRY_MAX)
    fns = None if var == "ideal" else make_fns(var)
    st = dict(clean=0, empty=0, mixed=0, win=[], loser=[], ovf=0, lv=0)
    for _ in range(T):
        g = ytr - Ftr
        gu = g / max(np.linalg.norm(g), 1e-12)
        feats = np.arange(d) if m == d else np.sort(rng.choice(d, m, replace=False))
        Oh = np.zeros((n, m * nb))
        for jj, j in enumerate(feats):
            Oh[np.arange(n), jj * nb + Btr[:, j]] = 1.0
        W = np.ones((n, 1))
        levels = []
        cj = np.repeat(feats, nb - 1)
        ck = np.tile(np.arange(nb - 1), m)
        for t in range(D):
            G = (W.T @ (gu[:, None] * Oh)).reshape(-1, m, nb)
            H = (W.T @ Oh).reshape(-1, m, nb) / n
            GL, HL = np.cumsum(G, 2)[:, :, :nb - 1], np.cumsum(H, 2)[:, :, :nb - 1]
            Gp, Hp = G.sum(2, keepdims=True), H.sum(2, keepdims=True)
            raw = ((GL * Hp - Gp * HL) ** 2).sum(0).ravel()
            if fns is None:
                ind = np.zeros(raw.size); ind[int(np.argmax(raw))] = 1.0
            else:
                lf, pf, _ = fns
                v = np.log2(np.maximum(raw / (4.0 * n), 2.0 ** -T_LOG)) / T_LOG
                b = lf(v[:, None] - v[None, :])
                np.fill_diagonal(b, 1.0)
                ind0 = b.prod(1)
                top = int(np.argmax(raw))
                st["win"].append(ind0[top])
                st["loser"].append(ind0.sum() - ind0[top])
                # 교차 정리
                tau = rng_tau.uniform(0.25, 0.75)
                c = np.cumsum(ind0)
                zc = (c - tau) / K_POST
                st["ovf"] += int(np.any(np.abs(zc) > 1.0)); st["lv"] += 1
                up = pf(zc)
                ind = up - np.concatenate([[0.0], up[:-1]])
                tot = ind.sum()
                if tot < 0.5:
                    st["empty"] += 1
                elif ind.max() > 0.9 and tot - ind.max() < 0.1:
                    st["clean"] += 1
                else:
                    st["mixed"] += 1
            levels.append(ind)
            bl = (Btr[:, cj] <= ck) @ ind
            WL = W * bl[:, None]
            W = np.stack([WL, W - WL], 2).reshape(n, -1)
        Gl = W.T @ g
        Hl = W.sum(0)
        leaf = LR * Gl / (Hl + lam)
        Ftr += W @ leaf
        P = np.ones((len(Bte), 1))
        for ind in levels:
            sel = xte[:, cj] @ ind
            th = ((ck + 1.0) / nmax) @ ind
            z = sel - th
            bit = (z > 0).astype(float) if fns is None else fns[2](z)
            P = np.stack([P * (1 - bit)[:, None], P * bit[:, None]], 2).reshape(len(Bte), -1)
        Fte += P @ leaf
    tot = max(1, st["clean"] + st["empty"] + st["mixed"])
    summ = dict(clean=st["clean"] / tot, empty=st["empty"] / tot, mixed=st["mixed"] / tot,
                win_p1=float(np.percentile(st["win"], 1)) if st["win"] else 1.0,
                win_med=float(np.median(st["win"])) if st["win"] else 1.0,
                los_p99=float(np.percentile(st["loser"], 99)) if st["loser"] else 0.0,
                ovf=st["ovf"] / max(1, st["lv"]))
    return name, var, roc_auc_score(yte, Fte), summ


def main():
    for k in ("dp10", "dp8", "post", "inf"):
        sp = spec(k)
        print(f"  {k:<5} degs={sp['degs']} stages={sp['stages']}  {sp['cost_ms']/1000:.2f}s"
              f"  τ_final={sp['tau_final']:.2e}")
    data = {}
    for nm, oml, ver in E.DATASETS:
        if nm in DATA:
            X, y = E.load(nm, oml, ver)
            data[nm] = (np.nan_to_num(X, nan=-1e12), y)
    tasks = [(nm, *data[nm], v, s) for nm in DATA for v in VARIANTS for s in SEEDS]
    res = {}
    with ProcessPoolExecutor(30) as ex:
        for nm, v, auc, st in ex.map(run, tasks):
            res.setdefault((nm, v), []).append((auc, st))

    print(f"\nsymmetric D={D}, T={T}, seed {len(SEEDS)}개,  log t={T_LOG}")
    print(f"\n① AUC@{T}  (괄호: ideal 대비)")
    print(f"  {'data':<9}" + "".join(f"{v:>16}" for v in VARIANTS))
    for nm in DATA:
        ref = np.mean([a for a, _ in res[(nm, 'ideal')]])
        row = ""
        for v in VARIANTS:
            a = np.mean([x for x, _ in res[(nm, v)]])
            row += f"{a:>8.4f}({a - ref:+.4f})"
        print(f"  {nm:<9}{row}")
    vs = [v for v in VARIANTS if v != "ideal"]
    print("\n② 교차 정리 후 지시자  clean(최대>0.9, 나머지<0.1) / empty / mixed")
    print(f"  {'data':<9}" + "".join(f"{v:>20}" for v in vs))
    for nm in DATA:
        row = ""
        for v in vs:
            s = {k: np.mean([q[k] for _, q in res[(nm, v)]]) for k in ("clean", "empty", "mixed")}
            row += f"   {s['clean']:>4.0%}/{s['empty']:>3.0%}/{s['mixed']:>4.0%}"
        print(f"  {nm:<9}{row}")
    print("\n③ league 원시 출력 (교차 정리 전):  승자 값 1%·중앙 / 패자 질량 합 99% / 교차 정리 정의역 초과율")
    print(f"  {'data':<9}" + "".join(f"{v:>28}" for v in vs))
    for nm in DATA:
        row = ""
        for v in vs:
            q = [s for _, s in res[(nm, v)]]
            row += (f"   {np.mean([s['win_p1'] for s in q]):.3f}·{np.mean([s['win_med'] for s in q]):.3f}"
                    f" / {np.mean([s['los_p99'] for s in q]):.3f} / {np.mean([s['ovf'] for s in q]):.0%}")
        print(f"  {nm:<9}{row}")


if __name__ == "__main__":
    main()

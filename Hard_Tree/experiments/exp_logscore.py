# Hard_Tree/experiments/exp_logscore.py
"""
평문 실험 — 점수 정규화(나눗셈)를 log 영역 비교로 대체할 수 있나.

  아이디어:  레벨 안 모든 후보가 같은 분모로 나뉜다 → log 를 쓰면 분모는 공통 상수,
             league 의 차분 log s_i − log s_j 에서 지워진다.  나눗셈·Goldschmidt 불필요.
  log 입력:  u = raw/4  ∈ [0,1]  (raw ≤ 4·ΣH′³ ≤ 4, 공개 상수로만 나눔)
             v = log2(max(u, 2^-t)) / t  ∈ [−1, 0]    (HEaaN-NB ApproxLog 와 같은 하한 2^-t)
  비교 대상 (1등·2등을 가르는 데 필요한 α = ⌈−log2(gap)⌉)
    bound      s/(4ΣH′³)  정확한 나눗셈
    trunc      s·inv,  inv = Goldschmidt 6회 (현재 NORM_ITERS_MAX) → (1−e^64)/den
    max        s/max s    이상적 기준
    log-t      위 v,  t = 16, 24, 32

실행:  ~/.pyenv/versions/temp/bin/python exp_logscore.py
"""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E

D, T, LR = 6, 256, 0.3
DATA = ["phoneme", "spambase", "magic", "bank", "adult"]
TS = (16, 24, 32)
GS_ITERS = 6


def need(s):
    a, b = np.sort(s)[-2:]
    g = b - a
    return 60 if g <= 0 else int(np.ceil(-np.log2(g)))


def run(args):
    name, X, y, seed = args
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    edges = E.fit_bins(Xtr, E.N_MAX)
    Btr = E.apply_bins(Xtr, edges)
    n, d = Btr.shape
    nb = E.N_MAX
    lam = max(1.0, n / (1 << D))
    rng = np.random.default_rng(1000 + seed)
    Ftr = np.full(n, ytr.mean())
    m = min(d, E.MTRY_MAX)
    keys = ["bound", "trunc", "max"] + [f"log{t}" for t in TS]
    al = {k: [] for k in keys}
    top = []
    lv_trunc = {t: [] for t in range(D)}
    for _ in range(T):
        g = ytr - Ftr
        gu = g / max(np.linalg.norm(g), 1e-12)
        feats = np.arange(d) if m == d else np.sort(rng.choice(d, m, replace=False))
        nid = np.zeros(n, np.int64)
        for t in range(D):
            P = 1 << t
            GL, HL, Gp, Hp = E.level_hist(Btr, gu, nid, P, feats, nb)
            HL, Hp = HL / n, Hp / n
            raw = ((GL * Hp - Gp * HL) ** 2).sum(0).ravel()
            den = 4.0 * (Hp[:, 0, 0] ** 3).sum()
            z = den / 4.0                                      # Goldschmidt: z = x/hi, hi = 4
            inv = (1.0 - (1.0 - z) ** (2 ** GS_ITERS)) / den   # 6회 절단 근사
            lv_trunc[t].append(den * inv)                      # = 실효 축소 배율
            al["bound"].append(need(raw / den))
            al["trunc"].append(need(raw * inv))
            al["max"].append(need(raw / max(raw.max(), 1e-300)))
            u = raw / 4.0
            top.append(np.log2(max(u.max(), 1e-300)))
            for tt in TS:
                v = np.log2(np.maximum(u, 2.0 ** -tt)) / tt
                al[f"log{tt}"].append(need(v))
            jj, k = divmod(int(np.argmax(raw)), nb - 1)
            nid = 2 * nid + (Btr[:, feats[jj]] > k)
        L = 1 << D
        leaf = LR * np.bincount(nid, g, L) / (np.bincount(nid, None, L) + lam)
        Ftr += leaf[nid]
    return name, al, np.array(top), {t: np.array(v) for t, v in lv_trunc.items()}


def main():
    data = {}
    for nm, oml, ver in E.DATASETS:
        if nm in DATA:
            X, y = E.load(nm, oml, ver)
            data[nm] = (np.nan_to_num(X, nan=-1e12), y)
    res = {}
    with ProcessPoolExecutor(15) as ex:
        for nm, al, top, lt in ex.map(run, [(nm, *data[nm], s) for nm in DATA for s in range(3)]):
            r = res.setdefault(nm, {"al": {}, "top": [], "lt": {}})
            for k, v in al.items():
                r["al"].setdefault(k, []).extend(v)
            r["top"].extend(top)
            for t, v in lt.items():
                r["lt"].setdefault(t, []).extend(v)

    keys = ["bound", "trunc", "max"] + [f"log{t}" for t in TS]
    print("① 1등/2등을 가르는 데 필요한 α  (중앙값 / 90% / 99%)")
    print(f"  {'data':<9}" + "".join(f"{k:>14}" for k in keys))
    for nm in DATA:
        row = ""
        for k in keys:
            v = np.array(res[nm]["al"][k])
            row += f"{int(np.median(v)):>6}/{int(np.percentile(v, 90)):>2}/{int(np.percentile(v, 99)):>2}  "
        print(f"  {nm:<9}{row}")

    print("\n② 1등 점수 크기 log2(raw/4)  (1% / 중앙값 / 99%)  → log 하한 2^-t 선택용")
    for nm in DATA:
        v = np.array(res[nm]["top"])
        print(f"  {nm:<9}{np.percentile(v, 1):>8.1f}{np.median(v):>8.1f}{np.percentile(v, 99):>8.1f}")

    print(f"\n③ Goldschmidt {GS_ITERS}회 절단의 실효 축소 배율 (레벨별 중앙값, 1 이면 정확)")
    print(f"  {'data':<9}" + "".join(f"{'t='+str(t):>8}" for t in range(D)))
    for nm in DATA:
        print(f"  {nm:<9}" + "".join(f"{np.median(res[nm]['lt'][t]):>8.3f}" for t in range(D)))


if __name__ == "__main__":
    main()

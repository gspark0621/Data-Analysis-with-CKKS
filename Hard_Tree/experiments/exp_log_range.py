# Hard_Tree/experiments/exp_log_range.py
"""
평문 실험 — log 정규화 입력 u = raw/(4n) 의 실제 범위.
  · 1등 u 의 log2 분포 → log 하한 2^-t 와 범위 축소 단계 수를 정한다
  · 2등/1등 비 → 필요한 상대 정밀도
정확한 argmax 경로로 부스팅 (exp_argmax_tau 와 같은 정규화).

실행:  ~/.pyenv/versions/temp/bin/python exp_log_range.py
"""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E

D, T, LR = 6, 128, 0.3
DATA = [nm for nm, _, _ in E.DATASETS]


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
    top, ratio, umax = [], [], []
    for _ in range(T):
        g = ytr - Ftr
        gu = g / max(np.linalg.norm(g), 1e-12)
        feats = np.arange(d) if m == d else np.sort(rng.choice(d, m, replace=False))
        nid = np.zeros(n, np.int64)
        for t in range(D):
            GL, HL, Gp, Hp = E.level_hist(Btr, gu, nid, 1 << t, feats, nb)
            HL, Hp = HL / n, Hp / n
            raw = ((GL * Hp - Gp * HL) ** 2).sum(0).ravel()
            u = np.sort(raw / (4.0 * n))
            top.append(np.log2(max(u[-1], 1e-300)))
            ratio.append(u[-2] / max(u[-1], 1e-300))
            jj, k = divmod(int(np.argmax(raw)), nb - 1)
            nid = 2 * nid + (Btr[:, feats[jj]] > k)
        L = 1 << D
        leaf = LR * np.bincount(nid, g, L) / (np.bincount(nid, None, L) + lam)
        Ftr += leaf[nid]
    return name, n, top, ratio


def main():
    data = {}
    for nm, oml, ver in E.DATASETS:
        if nm in DATA:
            X, y = E.load(nm, oml, ver)
            data[nm] = (np.nan_to_num(X, nan=-1e12), y)
    res = {}
    with ProcessPoolExecutor(21) as ex:
        for nm, n, top, ratio in ex.map(run, [(nm, *data[nm], s) for nm in data for s in range(3)]):
            r = res.setdefault(nm, dict(n=n, top=[], ratio=[]))
            r["top"] += top; r["ratio"] += ratio
    print("log2(1등 u)  최소 / 1% / 중앙 / 99% / 최대      2등/1등 비  중앙 / 99%")
    for nm, r in res.items():
        tp = np.array(r["top"]); rt = np.array(r["ratio"])
        print(f"  {nm:<12} n={r['n']:>6}  {tp.min():7.1f}{np.percentile(tp,1):7.1f}{np.median(tp):7.1f}"
              f"{np.percentile(tp,99):7.1f}{tp.max():7.1f}      {np.median(rt):.3f} / {np.percentile(rt,99):.4f}")


if __name__ == "__main__":
    main()

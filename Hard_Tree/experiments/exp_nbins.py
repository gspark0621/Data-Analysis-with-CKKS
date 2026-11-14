# Hard_Tree/experiments/exp_nbins.py
"""
평문 실험 — bin 수(n_max) 가 symmetric 부스팅 정확도에 주는 영향.

  비교
    fixed-K     모든 feature 를 분위수 K 구간  (K = 4, 8, 16, 32, 255)
                255 는 LightGBM/XGBoost 기본 max_bin 수준 = 사실상 무손실 기준선
    adaptive-8  LightGBM 식 feature 별 bin 수.  고유값이 적은 feature 는 고유값마다
                bin 하나, 남는 후보 예산을 고유값이 많은 feature 에 재분배.
                전체 후보 수 Σ_j (n_j − 1) 는 fixed-8 과 같다 (= 7d)  → FHE 비용 동일

  나머지 설정은 exp_sym_vs_asym 의 "ours" (대리 gain, λ=n/2^D, lr=0.3, mtry ≤ 16).

실행:  ~/.pyenv/versions/temp/bin/python exp_nbins.py
"""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E

D = 6
T = 256
SEEDS = range(3)
VARIANTS = ["fixed-4", "fixed-8", "fixed-16", "fixed-32", "fixed-255", "adaptive-8"]


def quantile_edges(col, k):
    qs = np.linspace(0, 1, k + 1)[1:-1]
    return np.unique(np.quantile(col, qs))


def adaptive_edges(Xtr, per_feat=8):
    """
    feature 별 bin 수를 데이터로 정한다 (LightGBM 의 distinct-aware binning).
      · 고유값 u_j ≤ 예산이면 고유값마다 bin (경계 = 인접 고유값 중점) → 무손실
      · 남는 예산은 고유값이 많은 feature 들에 균등 재분배
      · 전체 후보 수 Σ(n_j − 1) ≤ (per_feat − 1)·d  (fixed-per_feat 과 같은 FHE 비용)
    """
    d = Xtr.shape[1]
    uniq = [np.unique(Xtr[:, j]) for j in range(d)]
    budget = (per_feat - 1) * d                     # 후보(경계) 수 총량
    need = np.array([len(u) - 1 for u in uniq])     # 무손실에 필요한 경계 수
    alloc = np.zeros(d, int)
    order = np.argsort(need)                        # 적게 필요한 것부터 채운다
    left, rest = budget, d
    for j in order:
        share = left // rest
        alloc[j] = min(need[j], share)
        left -= alloc[j]; rest -= 1
    edges = []
    for j in range(d):
        u = uniq[j]
        if alloc[j] >= len(u) - 1:
            edges.append((u[:-1] + u[1:]) / 2)      # 고유값마다 bin
        else:
            edges.append(quantile_edges(Xtr[:, j], alloc[j] + 1))
    return edges, alloc + 1


def task(args):
    name, X, y, var, seed = args
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=seed)
    if var.startswith("fixed"):
        k = int(var.split("-")[1])
        edges = [quantile_edges(Xtr[:, j], k) for j in range(X.shape[1])]
    else:
        edges, _ = adaptive_edges(Xtr, int(var.split("-")[1]))
    nb = max(len(e) for e in edges) + 1
    Btr, Bte = E.apply_bins(Xtr, edges), E.apply_bins(Xte, edges)
    E.T_MAX = T
    auc, _ = E.boost(Btr, ytr, Bte, yte, D, E.SETTINGS["ours"], True, seed, nb)
    n_cand = sum(len(e) for e in edges)
    return name, var, seed, auc, n_cand


def main():
    t0 = time.time()
    data, info = {}, {}
    for name, oml, ver in E.DATASETS:
        X, y = E.load(name, oml, ver)
        X = np.nan_to_num(X, nan=-1e12)
        data[name] = (X, y)
        u = [len(np.unique(X[:, j])) for j in range(X.shape[1])]
        info[name] = dict(d=X.shape[1], low_card=int(sum(v <= 8 for v in u)))
    tasks = [(nm, *data[nm], v, s) for nm in data for v in VARIANTS for s in SEEDS]
    res = {}
    with ProcessPoolExecutor(max_workers=30) as ex:
        for nm, v, s, auc, nc in ex.map(task, tasks):
            res.setdefault((nm, v), []).append((auc, nc))

    out = []
    print(f"symmetric D={D}, T≤{T}, seed {len(SEEDS)}개.  값 = 최고 AUC (괄호: fixed-255 대비)")
    print(f"  {'data':<9}{'d':>3}{'저카디널':>7} | " + "".join(f"{v:>15}" for v in VARIANTS))
    for nm in data:
        peaks = {}
        for v in VARIANTS:
            c = np.mean([a for a, _ in res[(nm, v)]], 0)
            peaks[v] = float(c.max())
            out.append(dict(dataset=nm, variant=v, peak=peaks[v],
                            n_cand=float(np.mean([n for _, n in res[(nm, v)]])),
                            curve=[float(x) for x in c]))
        ref = peaks["fixed-255"]
        cells = "".join(f"{peaks[v]:>8.4f}({peaks[v]-ref:+.3f})" for v in VARIANTS)
        print(f"  {nm:<9}{info[nm]['d']:>3}{info[nm]['low_card']:>7} | {cells}")
    os.makedirs(E.OUT_DIR, exist_ok=True)
    with open(os.path.join(E.OUT_DIR, "nbins.json"), "w") as f:
        json.dump(out, f)
    print(f"\n  저카디널 = 고유값 ≤ 8 인 feature 수.   총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

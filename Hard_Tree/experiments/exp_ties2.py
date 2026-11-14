# Hard_Tree/experiments/exp_ties2.py
"""
평문 실험 — 근접 동점의 원인과 "동점이면 랜덤 하나" 의 정확도 영향.

  ① 1등−2등 간격을 가르는 데 필요한 α = ⌈−log2(gap)⌉ 의 분포를 정규화 방식별로
       bound : s / (4·Σ_p H′³)      현재 코드 (normalize_score)
       sum   : s / Σ_i s_i          후보 합으로 나눔 (덧셈 + 나눗셈 1회)
       max   : s / max_i s_i        이상적 기준 (max 자체가 argmax 급 연산)
  ② naive 랜덤 : bound 정규화에서 top 과 2^-α 이내인 후보 집합 중 무작위 하나 선택
       → FHE 가 흐린 지시자 대신 "아무 근접 후보" 를 고른다고 가정할 때의 AUC

실행:  ~/.pyenv/versions/temp/bin/python exp_ties2.py
"""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E

D, T, LR = 6, 256, 0.3
DATA = ["phoneme", "spambase", "magic", "bank", "adult"]
MODES = ("exact", "rand8", "rand6", "rand4")      # randA = 2^-A 이내 무작위


def need_alpha(s):
    a, b = np.sort(s)[-2:]
    gap = b - a
    return 60 if gap <= 0 else int(np.ceil(-np.log2(gap)))


def run(args):
    name, X, y, mode, seed = args
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=seed)
    edges = E.fit_bins(Xtr, E.N_MAX)
    Btr, Bte = E.apply_bins(Xtr, edges), E.apply_bins(Xte, edges)
    n, d = Btr.shape
    nb = E.N_MAX
    lam = max(1.0, n / (1 << D))
    rng = np.random.default_rng(1000 + seed)
    Ftr = np.full(n, ytr.mean()); Fte = np.full(len(Bte), ytr.mean())
    m = min(d, E.MTRY_MAX)
    al = {"bound": [], "sum": [], "max": []}
    for _ in range(T):
        g = ytr - Ftr
        gu = g / max(np.linalg.norm(g), 1e-12)
        feats = np.arange(d) if m == d else np.sort(rng.choice(d, m, replace=False))
        nid = np.zeros(n, np.int64)
        splits = []
        for t in range(D):
            P = 1 << t
            GL, HL, Gp, Hp = E.level_hist(Btr, gu, nid, P, feats, nb)
            HL, Hp = HL / n, Hp / n
            raw = ((GL * Hp - Gp * HL) ** 2).sum(0).ravel()
            sb = raw / (4.0 * (Hp[:, 0, 0] ** 3).sum())
            al["bound"].append(need_alpha(sb))
            al["sum"].append(need_alpha(raw / max(raw.sum(), 1e-300)))
            al["max"].append(need_alpha(raw / max(raw.max(), 1e-300)))
            if mode == "exact":
                pick = int(np.argmax(sb))
            else:
                A = int(mode[4:])
                cand = np.nonzero(sb >= sb.max() - 2.0 ** -A)[0]
                pick = int(rng.choice(cand))
            jj, k = divmod(pick, nb - 1)
            sp = np.tile([feats[jj], k], (P, 1))
            splits.append(sp)
            nid = 2 * nid + (Btr[np.arange(n), sp[nid, 0]] > sp[nid, 1])
        L = 1 << D
        leaf = LR * np.bincount(nid, g, L) / (np.bincount(nid, None, L) + lam)
        Ftr += leaf[nid]; Fte += leaf[E.route(Bte, splits)]
    return name, mode, roc_auc_score(yte, Fte), {k: np.array(v) for k, v in al.items()}


def main():
    data = {}
    for nm, oml, ver in E.DATASETS:
        if nm in DATA:
            X, y = E.load(nm, oml, ver)
            data[nm] = (np.nan_to_num(X, nan=-1e12), y)
    tasks = [(nm, *data[nm], md, s) for nm in DATA for md in MODES for s in range(3)]
    res = {}
    with ProcessPoolExecutor(30) as ex:
        for nm, md, auc, al in ex.map(run, tasks):
            res.setdefault((nm, md), []).append((auc, al))

    print("① 1등/2등을 가르는 데 필요한 α  (중앙값 / 90% / 99%, exact 경로 기준)")
    print(f"  {'data':<9}{'bound (현재)':>22}{'sum':>18}{'max (이상)':>18}")
    for nm in DATA:
        row = ""
        for k in ("bound", "sum", "max"):
            v = np.concatenate([a[k] for _, a in res[(nm, "exact")]])
            row += f"{int(np.median(v)):>8}/{int(np.percentile(v, 90)):>3}/{int(np.percentile(v, 99)):>3}   "
        print(f"  {nm:<9}{row}")

    print(f"\n② AUC@{T}  — exact vs '2^-A 이내 후보 중 무작위 하나'")
    print(f"  {'data':<9}" + "".join(f"{m:>10}" for m in MODES))
    for nm in DATA:
        print(f"  {nm:<9}" + "".join(
            f"{np.mean([a for a, _ in res[(nm, m)]]):>10.4f}" for m in MODES))


if __name__ == "__main__":
    main()

# Hard_Tree/experiments/exp_ties.py
"""
평문 실험 — argmax 근접 동점이 얼마나 자주 생기고, 공개 난수로 깨면 정확도가 어떤가.

  FHE 와 같은 정규화 점수  s = Σ_p (G′_L·H′_p − G′_p·H′_L)² / (4·Σ_p H′_p³)
    g′ = g/‖g‖ (라운드마다 단위노름 — 최선의 경우),  H′ = H/n
  근접 동점  : 1등−2등 간격 < 2^-α  (α = 8, sign dead zone)
               → FHE argmax 가 0.5/0.5 같은 흐린 지시자를 낸다.
               h(x)=3x²−2x³ 는 0.5 가 고정점이라 bit_cleaning 으로도 못 고친다.
  랜덤 타이브레이크 : s ← s + ρ·u,  u ~ U(0,1) 공개 난수  (CatBoost random_strength 와 같은 발상)
    ρ = 0, 2^-8, 2^-6, 2^-4

실행:  ~/.pyenv/versions/temp/bin/python exp_ties.py
"""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E

ALPHA = 8
D, T, LR = 6, 256, 0.3
RHOS = (0.0, 2.0 ** -8, 2.0 ** -6, 2.0 ** -4)
DATA = ["phoneme", "spambase", "magic", "bank", "adult"]


def run(args):
    name, X, y, rho, seed = args
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
    close = tot = 0
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
            s = raw / (4.0 * (Hp[:, 0, 0] ** 3).sum())
            s = s + rho * rng.random(s.size)           # 공개 난수 타이브레이크
            top2 = np.sort(s)[-2:]
            close += int(top2[1] - top2[0] < 2.0 ** -ALPHA); tot += 1
            jj, k = divmod(int(np.argmax(s)), nb - 1)
            sp = np.tile([feats[jj], k], (P, 1))
            splits.append(sp)
            nid = 2 * nid + (Btr[np.arange(n), sp[nid, 0]] > sp[nid, 1])
        L = 1 << D
        leaf = LR * np.bincount(nid, g, L) / (np.bincount(nid, None, L) + lam)
        Ftr += leaf[nid]; Fte += leaf[E.route(Bte, splits)]
    return name, rho, roc_auc_score(yte, Fte), close / tot


def main():
    data = {}
    for nm, oml, ver in E.DATASETS:
        if nm in DATA:
            X, y = E.load(nm, oml, ver)
            data[nm] = (np.nan_to_num(X, nan=-1e12), y)
    tasks = [(nm, *data[nm], r, s) for nm in DATA for r in RHOS for s in range(3)]
    res = {}
    with ProcessPoolExecutor(30) as ex:
        for nm, r, auc, cf in ex.map(run, tasks):
            res.setdefault((nm, r), []).append((auc, cf))
    print(f"symmetric D={D}, T={T}, α={ALPHA}.  칸 = AUC@{T} / 근접동점 비율")
    print(f"  {'data':<9}" + "".join(f"{'ρ='+('0' if r == 0 else '2^'+str(int(np.log2(r)))):>20}" for r in RHOS))
    for nm in DATA:
        cells = ""
        for r in RHOS:
            a = np.mean([x for x, _ in res[(nm, r)]]); c = np.mean([y for _, y in res[(nm, r)]])
            cells += f"{a:>11.4f} / {c:>6.1%}"
        print(f"  {nm:<9}{cells}")


if __name__ == "__main__":
    main()

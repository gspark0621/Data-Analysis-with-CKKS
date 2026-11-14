# Hard_Tree/experiments/exp_sym_vs_asym.py
"""
평문 실험 — oblivious(symmetric) 트리 부스팅은 비대칭 트리 대비 몇 배의 트리가 필요한가.

  같은 조건에서 두 모델을 학습한다.
    sym   레벨마다 분기 하나 (j_t, k_t) 를 모든 노드가 공유
    asym  노드마다 자기 분기
  공통 설정 ("ours" = 현재 FHE 코드와 동일)
    · 분위수 구간화 n_max (train 분위수)          k ∈ [0, n_max−1),  left = bin ≤ k
    · 분할 기준  (G_L·H_p − G_p·H_L)²            나눗셈 없는 대리 기준 (boost_gain)
    · 리프값     lr·G/(H+λ),  λ = n/2^D           (leaf_values)
    · 제곱손실 잔차 부스팅,  F0 = ȳ
    · mtry = min(d, 16) 트리마다 무작위            (league argmax 의 nsp² ≤ M 제약)
    · 완전 이진트리 (FHE 는 깊이가 공개 · 고정)
  대조 설정 ("std") — 일반 GBDT 에 가까운 설정으로 결론이 설정에 의존하는지 확인
    · 정확한 gain  G_L²/(H_L+1) + G_R²/(H_R+1) − G_p²/(H_p+1),  λ = 1

  핵심 지표
    등정확도 배율  ratio(T_a) = min{ T_s : AUC_sym(T_s) ≥ AUC_asym(T_a) } / T_a
    이 값이 FHE 비용 배율 (2^D−1)/D 보다 작으면 같은 비용에서 sym 이 이긴다.

실행:  python -m Hard_Tree.experiments.exp_sym_vs_asym
"""
from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

N_MAX = 8
MTRY_MAX = 16
T_MAX = 512
DEPTHS = (4, 6)
SEEDS = range(5)
SETTINGS = {
    "ours": dict(crit="surrogate", lam="n/2^D", lr=0.3),
    "std":  dict(crit="exact",     lam=1.0,     lr=0.1),
}
T_REPORT = (8, 16, 32, 64, 128, 256, 512)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# (이름, OpenML 이름, 버전)  — None 이면 sklearn 내장
DATASETS = [
    ("cancer",   None,             None),
    ("diabetes", "diabetes",       1),
    ("phoneme",  "phoneme",        1),
    ("spambase", "spambase",       1),
    ("magic",    "MagicTelescope", 1),
    ("electric", "electricity",    1),
    ("bank",     "bank-marketing", 1),
    ("adult",    "adult",          2),
]


# ══════════════════════════════════════════════════════════════════
# 데이터
# ══════════════════════════════════════════════════════════════════
def load(name, oml, ver):
    if oml is None:
        X, y = load_breast_cancer(return_X_y=True)
        return X.astype(float), y.astype(int)
    X, y = fetch_openml(oml, version=ver, return_X_y=True, as_frame=False,
                        parser="liac-arff")
    X = np.asarray(X, float)
    classes = np.unique(y)
    assert len(classes) == 2, (name, classes)
    yb = (y == classes[1]).astype(int)
    if yb.mean() > 0.5:                       # 양성 = 소수 클래스
        yb = 1 - yb
    return X, yb


def fit_bins(Xtr, n_max):
    """train 분위수로 경계.  bin = searchsorted(경계, x, right) ∈ [0, n_max)."""
    qs = np.linspace(0, 1, n_max + 1)[1:-1]
    return [np.unique(np.quantile(Xtr[:, j], qs)) for j in range(Xtr.shape[1])]


def apply_bins(X, edges):
    B = np.empty(X.shape, np.int16)
    for j, e in enumerate(edges):
        B[:, j] = np.searchsorted(e, X[:, j], side="right")
    return B


# ══════════════════════════════════════════════════════════════════
# 분할 탐색
# ══════════════════════════════════════════════════════════════════
def level_hist(B, g, nid, P, feats, nb):
    """파티션 p · 후보 (j,k) 별 왼쪽 G_L, H_L 과 부모 G_p, H_p."""
    m = len(feats)
    G = np.empty((P, m, nb)); H = np.empty((P, m, nb))
    base = nid * nb
    for jj, j in enumerate(feats):
        idx = base + B[:, j]
        G[:, jj] = np.bincount(idx, g, P * nb).reshape(P, nb)
        H[:, jj] = np.bincount(idx, None, P * nb).reshape(P, nb)
    GL = np.cumsum(G, 2)[:, :, :nb - 1]          # bin ≤ k,  k ∈ [0, nb−1)
    HL = np.cumsum(H, 2)[:, :, :nb - 1]
    Gp = G.sum(2, keepdims=True)
    Hp = H.sum(2, keepdims=True)
    return GL, HL, Gp, Hp


def split_score(GL, HL, Gp, Hp, crit):
    """(P, m, nb−1).  클수록 좋다."""
    if crit == "surrogate":
        return (GL * Hp - Gp * HL) ** 2
    GR, HR = Gp - GL, Hp - HL
    return GL ** 2 / (HL + 1) + GR ** 2 / (HR + 1) - Gp ** 2 / (Hp + 1)


def fit_tree(B, g, D, feats, crit, sym, nb):
    """반환: splits[t] = (P_t, 2) 배열 (j, k).  sym 이면 모든 노드가 같은 행."""
    n = len(g)
    nid = np.zeros(n, np.int64)
    splits = []
    for t in range(D):
        P = 1 << t
        sc = split_score(*level_hist(B, g, nid, P, feats, nb), crit)
        if sym:
            s = sc.sum(0).ravel()
            jj, k = divmod(int(np.argmax(s)), nb - 1)
            sp = np.tile([feats[jj], k], (P, 1))
        else:
            flat = sc.reshape(P, -1).argmax(1)
            jj, k = np.divmod(flat, nb - 1)
            sp = np.stack([np.asarray(feats)[jj], k], 1)
        splits.append(sp)
        right = B[np.arange(n), sp[nid, 0]] > sp[nid, 1]
        nid = 2 * nid + right
    return splits, nid


def route(B, splits):
    nid = np.zeros(len(B), np.int64)
    rows = np.arange(len(B))
    for sp in splits:
        nid = 2 * nid + (B[rows, sp[nid, 0]] > sp[nid, 1])
    return nid


# ══════════════════════════════════════════════════════════════════
# 부스팅
# ══════════════════════════════════════════════════════════════════
def boost(Btr, ytr, Bte, yte, D, setting, sym, seed, nb):
    crit, lr = setting["crit"], setting["lr"]
    n, d = Btr.shape
    lam = max(1.0, n / (1 << D)) if setting["lam"] == "n/2^D" else float(setting["lam"])
    rng = np.random.default_rng(seed)
    Ftr = np.full(n, ytr.mean())
    Fte = np.full(len(Bte), ytr.mean())
    auc, acc = np.empty(T_MAX), np.empty(T_MAX)
    m = min(d, MTRY_MAX)
    for t in range(T_MAX):
        g = ytr - Ftr                                    # 잔차 (음의 gradient)
        feats = (np.arange(d) if m == d
                 else np.sort(rng.choice(d, m, replace=False)))
        splits, nid = fit_tree(Btr, g, D, feats, crit, sym, nb)
        L = 1 << D
        leaf = lr * np.bincount(nid, g, L) / (np.bincount(nid, None, L) + lam)
        Ftr += leaf[nid]
        Fte += leaf[route(Bte, splits)]
        auc[t] = roc_auc_score(yte, Fte)
        acc[t] = ((Fte > 0.5) == yte).mean()
    return auc, acc


def run_task(args):
    name, X, y, D, sname, seed = args
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=seed)
    edges = fit_bins(Xtr, N_MAX)
    Btr, Bte = apply_bins(Xtr, edges), apply_bins(Xte, edges)
    st = SETTINGS[sname]
    out = {}
    for sym in (True, False):
        out["sym" if sym else "asym"] = boost(Btr, ytr, Bte, yte, D, st, sym,
                                              seed, N_MAX)
    return name, D, sname, seed, out


# ══════════════════════════════════════════════════════════════════
# 집계
# ══════════════════════════════════════════════════════════════════
def iso_ratio(sym_curve, asym_curve, Ta):
    """AUC_sym(T_s) ≥ AUC_asym(T_a) 인 최소 T_s / T_a.  도달 못 하면 inf."""
    target = asym_curve[Ta - 1]
    hit = np.nonzero(sym_curve >= target)[0]
    return (hit[0] + 1) / Ta if len(hit) else float("inf")


def main():
    t0 = time.time()
    data = {}
    for name, oml, ver in DATASETS:
        X, y = load(name, oml, ver)
        X = np.nan_to_num(X, nan=-1e12)                  # 결측 → 가장 낮은 bin
        data[name] = (X, y)
        print(f"  {name:<9} n={len(y):>6} d={X.shape[1]:>3}  양성 {y.mean():.3f}",
              flush=True)

    tasks = [(nm, *data[nm], D, s, seed) for nm in data for D in DEPTHS
             for s in SETTINGS for seed in SEEDS]
    res = {}
    with ProcessPoolExecutor(max_workers=min(30, os.cpu_count() or 4)) as ex:
        for i, (nm, D, s, seed, out) in enumerate(ex.map(run_task, tasks), 1):
            res.setdefault((nm, D, s), []).append(out)
            if i % 20 == 0:
                print(f"    {i}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for (nm, D, s), runs in sorted(res.items()):
        mean = {k: (np.mean([r[k][0] for r in runs], 0),
                    np.mean([r[k][1] for r in runs], 0)) for k in ("sym", "asym")}
        cost = ((1 << D) - 1) / D
        row = dict(dataset=nm, n=int(len(data[nm][1])), d=int(data[nm][0].shape[1]),
                   D=D, setting=s, fhe_cost_ratio=cost,
                   auc_sym={T: float(mean["sym"][0][T - 1]) for T in T_REPORT},
                   auc_asym={T: float(mean["asym"][0][T - 1]) for T in T_REPORT},
                   acc_sym={T: float(mean["sym"][1][T - 1]) for T in T_REPORT},
                   acc_asym={T: float(mean["asym"][1][T - 1]) for T in T_REPORT},
                   iso={Ta: iso_ratio(mean["sym"][0], mean["asym"][0], Ta)
                        for Ta in (8, 16, 32, 64, 128)},
                   curve_sym=[float(v) for v in mean["sym"][0]],
                   curve_asym=[float(v) for v in mean["asym"][0]])
        summary.append(row)

    with open(os.path.join(OUT_DIR, "sym_vs_asym.json"), "w") as f:
        json.dump(summary, f)

    for s in SETTINGS:
        print(f"\n■ setting={s}  {SETTINGS[s]}")
        print(f"  {'data':<9}{'D':>2} {'비용배율':>7} | "
              + "".join(f"{'iso@'+str(T):>9}" for T in (8, 16, 32, 64, 128))
              + f" | {'AUC@64 s/a':>15}{'AUC@512 s/a':>16}")
        for r in summary:
            if r["setting"] != s:
                continue
            iso = "".join(f"{(v if v != float('inf') else 99):>9.2f}"
                          for v in r["iso"].values())
            print(f"  {r['dataset']:<9}{r['D']:>2} {r['fhe_cost_ratio']:>7.1f} | {iso} | "
                  f"{r['auc_sym'][64]:.4f}/{r['auc_asym'][64]:.4f}"
                  f"  {r['auc_sym'][512]:.4f}/{r['auc_asym'][512]:.4f}")
    print(f"\n  (iso 99 = {T_MAX} 트리 안에 도달 못 함)   총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

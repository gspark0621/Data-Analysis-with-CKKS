# Hard_Tree/experiments/exp_shift.py
"""
평문 실험 — DBSCAN 의 '반격자 평행이동' 을 league argmax 에 쓰면 흐린 지시자가 줄어드나.

  DBSCAN:  정수 count 에서 (min_pts − 0.5) 를 빼면 |x| ≥ 0.5 가 구조적으로 보장 → dead zone 회피.
  여기:    league 비교 b_ij = sign(s_i − s_j + ε)  (대각 제외),  ind_i = Π_j b_ij
           sign 은 |입력| < δ = 2^-α 에서 흐림(0<b<1), 그 밖은 깨끗한 0/1 로 모델링.
  분류 (레벨마다, 정확한 경로의 점수 벡터로)
    clean1   ind 가 정확히 하나만 1                → 그대로 사용 가능
    multi    깨끗한 1 이 둘 이상 (흐림 없음)        → 정수 누적합 + 0.5 평행이동으로 '첫 번째' 를 정확히 고를 수 있음
    fuzzy    흐린 값이 살아남은 ind 가 있음         → 정확한 처리 불가 (누적합 τ 랜덤 교차 등 필요)
  점수: bound 정규화 s / (4·Σ_p H′³) (현재 코드),  α = 8.

실행:  ~/.pyenv/versions/temp/bin/python exp_shift.py
"""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E

ALPHA = 8
D, T, LR = 6, 256, 0.3
DATA = ["phoneme", "spambase", "magic", "bank", "adult"]
DELTA = 2.0 ** -ALPHA
EPS = (0.0, 2 * DELTA, 2.0 ** -6, 2.0 ** -4)


def collect(args):
    """정확한 argmax 경로로 부스팅하며 레벨별 정규화 점수 벡터를 모은다."""
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
    out = []
    for _ in range(T):
        g = ytr - Ftr
        gu = g / max(np.linalg.norm(g), 1e-12)
        feats = np.arange(d) if m == d else np.sort(rng.choice(d, m, replace=False))
        nid = np.zeros(n, np.int64)
        for t in range(D):
            P = 1 << t
            GL, HL, Gp, Hp = E.level_hist(Btr, gu, nid, P, feats, nb)
            HL, Hp = HL / n, Hp / n
            s = ((GL * Hp - Gp * HL) ** 2).sum(0).ravel() / (4.0 * (Hp[:, 0, 0] ** 3).sum())
            out.append(s.astype(np.float64))
            jj, k = divmod(int(np.argmax(s)), nb - 1)
            j = feats[jj]
            nid = 2 * nid + (Btr[:, j] > k)
        L = 1 << D
        leaf = LR * np.bincount(nid, g, L) / (np.bincount(nid, None, L) + lam)
        Ftr += leaf[nid]
    return name, out


def classify(s, eps):
    z = s[:, None] - s[None, :] + eps          # b_ij 입력
    np.fill_diagonal(z, 1.0)                   # 대각 = 깨끗한 1
    zero = z < -DELTA                          # 깨끗한 0
    fuzzy = np.abs(z) <= DELTA
    alive = ~zero.any(1)                       # 0 인자가 없는 행
    fz = alive & fuzzy.any(1)                  # 흐린 인자가 살아남음
    if fz.any():
        return "fuzzy"
    return "clean1" if alive.sum() == 1 else "multi"


def main():
    data = {}
    for nm, oml, ver in E.DATASETS:
        if nm in DATA:
            X, y = E.load(nm, oml, ver)
            data[nm] = (np.nan_to_num(X, nan=-1e12), y)
    with ProcessPoolExecutor(15) as ex:
        res = {}
        for nm, out in ex.map(collect, [(nm, *data[nm], s) for nm in DATA for s in range(3)]):
            res.setdefault(nm, []).extend(out)
    print(f"α={ALPHA} (δ=2^-{ALPHA}).  칸 = clean1 / multi / fuzzy  (레벨 비율)")
    hdr = "".join(f"{'ε=' + ('0' if e == 0 else ('2δ' if e == 2 * DELTA else '2^' + str(int(np.log2(e))))):>24}" for e in EPS)
    print(f"  {'data':<9}{hdr}")
    for nm in DATA:
        row = ""
        for e in EPS:
            c = {"clean1": 0, "multi": 0, "fuzzy": 0}
            for s in res[nm]:
                c[classify(s, e)] += 1
            N = len(res[nm])
            row += f"   {c['clean1']/N:>5.0%} /{c['multi']/N:>4.0%} /{c['fuzzy']/N:>4.0%}"
        print(f"  {nm:<9}{row}")


if __name__ == "__main__":
    main()

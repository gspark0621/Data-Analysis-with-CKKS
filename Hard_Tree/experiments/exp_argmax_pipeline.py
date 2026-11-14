# Hard_Tree/experiments/exp_argmax_pipeline.py
"""
평문 시뮬레이션 — FHE argmax 의 흐린 지시자가 학습·추론에 주는 피해와 처리 방법 비교.

  ── FHE 동작 모델 ──────────────────────────────────────────────────
  sign01(z) ≈ clip(0.5 + z/(2δ), 0, 1),   δ = 2^-α
      |z| ≥ δ 이면 깨끗한 0/1,  dead zone 안에서는 선형으로 흐림.
  league:  b_ij = sign01(s_i − s_j)  (대각 1),   ind_i = Π_j b_ij
      (계층형 argmax 도 최종 흐림 구조는 같다고 가정 — 중간 C_max 는 거의 정확)
  흐린 ind 는 정제 없이 그대로 흘러간다 (FHE 와 동일):
      Model_t = Σ ind_i·onehot(j_i),  Θ_t = Σ ind_i·(k_i+1)/n_max
      b_left  = Σ ind_i·[x_{j_i} ≤ k_i]   → 파티션 마스크가 실수 가중치가 됨
      m_L = m·b_left,  m_R = m − m_L,   리프 G,H 도 가중합,  잔차 갱신도 가중 scatter
  추론:  Sel_t = Σ ind_i·x̃_{j_i},  bit_t = sign01(Sel_t − Θ_t) (δ_inf = 1/16, binning 마진)
         예측 = Σ_ℓ Π_t (bit 또는 1−bit) · leaf_ℓ
  ── 비교 변형 ───────────────────────────────────────────────────────
    ideal      정확한 argmax (평문 기준선)
    cur        상한 정규화, α=8                               (현재 코드)
    a16        상한 정규화, α=16                              (α 만 올림)
    post       상한 정규화, α=8  + 누적합 교차 정리
    resc       최댓값 정규화(s/C_max), α=10
    prop       최댓값 정규화, α=10 + 누적합 교차 정리          (제안)
  누적합 교차 정리:  c_i = Σ_{j≤i} ind_j,  w_i = σ(c_i − τ) − σ(c_{i−1} − τ)
      τ ~ U(0.25, 0.75) 공개 난수,  σ = sign01 with α_post = 6
      깨끗한 multi-hot 이면 c 가 정수 → 정확히 '첫 번째' 선택 (반격자 마진)
      흐림이면 질량 비율대로 랜덤 하나.  질량 < τ 면 선택 없음 → 레벨 건너뜀(무해)

실행:  ~/.pyenv/versions/temp/bin/python exp_argmax_pipeline.py
"""
from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import exp_sym_vs_asym as E

D, T, LR = 6, 128, 0.3
DATA = ["phoneme", "spambase", "magic", "bank", "adult"]
SEEDS = range(3)
VARIANTS = {
    "ideal": None,
    "cur":  dict(norm="bound", alpha=8,  post=False),
    "a16":  dict(norm="bound", alpha=16, post=False),
    "post": dict(norm="bound", alpha=8,  post=True),
    "resc": dict(norm="max",   alpha=10, post=False),
    "prop": dict(norm="max",   alpha=10, post=True),
}
ALPHA_POST = 6
DELTA_INF = 1.0 / 16


def sign01(z, alpha):
    return np.clip(0.5 + z / (2.0 * 2.0 ** -alpha), 0.0, 1.0)


def league(s, alpha):
    b = sign01(s[:, None] - s[None, :], alpha)
    np.fill_diagonal(b, 1.0)
    return b.prod(1)


def crossing(ind, rng):
    tau = rng.uniform(0.25, 0.75)
    c = np.cumsum(ind)
    up = sign01(c - tau, ALPHA_POST)
    return up - np.concatenate([[0.0], up[:-1]])


def run(args):
    name, X, y, var, seed = args
    cfg = VARIANTS[var]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=seed)
    edges = E.fit_bins(Xtr, E.N_MAX)
    Btr, Bte = E.apply_bins(Xtr, edges), E.apply_bins(Xte, edges)
    n, d = Btr.shape
    nb, nmax = E.N_MAX, E.N_MAX
    L = 1 << D
    lam = max(1.0, n / L)
    rng = np.random.default_rng(1000 + seed)       # mtry — 변형 간 동일
    rng_tau = np.random.default_rng(5000 + seed)
    Ftr = np.full(n, ytr.mean()); Fte = np.full(len(Bte), ytr.mean())
    xte = (Bte + 0.5) / nmax
    m = min(d, E.MTRY_MAX)
    stat = {"clean": 0, "empty": 0, "mixed": 0}
    for _ in range(T):
        g = ytr - Ftr
        gu = g / max(np.linalg.norm(g), 1e-12)
        feats = np.arange(d) if m == d else np.sort(rng.choice(d, m, replace=False))
        Oh = np.zeros((n, m * nb))
        for jj, j in enumerate(feats):
            Oh[np.arange(n), jj * nb + Btr[:, j]] = 1.0
        W = np.ones((n, 1))                         # 파티션 가중치 (FHE 마스크)
        levels = []                                  # (ind, cand_j, cand_k)
        cj = np.repeat(feats, nb - 1)
        ck = np.tile(np.arange(nb - 1), m)
        for t in range(D):
            G = (W.T @ (gu[:, None] * Oh)).reshape(-1, m, nb)
            H = (W.T @ Oh).reshape(-1, m, nb) / n
            GL, HL = np.cumsum(G, 2)[:, :, :nb - 1], np.cumsum(H, 2)[:, :, :nb - 1]
            Gp, Hp = G.sum(2, keepdims=True), H.sum(2, keepdims=True)
            raw = ((GL * Hp - Gp * HL) ** 2).sum(0).ravel()
            if cfg is None:
                ind = np.zeros(raw.size); ind[int(np.argmax(raw))] = 1.0
            else:
                if cfg["norm"] == "bound":
                    s = raw / (4.0 * (Hp[:, 0, 0] ** 3).sum())
                elif cfg["norm"] == "log":
                    # 공개 상한 4n 으로만 나눔 → u ∈ [0,1] 보장.  log2 후 1/t 로 [−1,0]
                    u = raw / (4.0 * n)
                    tt = cfg.get("t", 32)
                    s = np.log2(np.maximum(u, 2.0 ** -tt)) / tt
                else:
                    s = raw / max(raw.max(), 1e-300)
                ind = league(s, cfg["alpha"])
                if cfg["post"]:
                    ind = crossing(ind, rng_tau)
                tot = ind.sum()
                if tot < 0.5:
                    stat["empty"] += 1
                elif np.all((ind < 1e-6) | (ind > 1 - 1e-6)) and abs(tot - 1) < 1e-6:
                    stat["clean"] += 1
                else:
                    stat["mixed"] += 1
            levels.append(ind)
            bl = (Btr[:, cj] <= ck) @ ind                       # Σ ind_i·[x ≤ k]
            WL = W * bl[:, None]
            W = np.stack([WL, W - WL], 2).reshape(n, -1)        # 2p, 2p+1 순서
        Gl = W.T @ g
        Hl = W.sum(0)
        leaf = LR * Gl / (Hl + lam)
        Ftr += W @ leaf
        # ── 추론 (테스트) ──
        P = np.ones((len(Bte), 1))
        for ind in levels:
            sel = xte[:, cj] @ ind
            th = ((ck + 1.0) / nmax) @ ind
            bit = sign01(sel - th, 0) if False else np.clip(0.5 + (sel - th) / (2 * DELTA_INF), 0, 1)
            P = np.stack([P * (1 - bit)[:, None], P * bit[:, None]], 2).reshape(len(Bte), -1)
        Fte += P @ leaf
    tot = max(1, sum(stat.values()))
    return name, var, roc_auc_score(yte, Fte), {k: v / tot for k, v in stat.items()}


def main():
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

    print(f"symmetric D={D}, T={T}, seed {len(SEEDS)}개")
    print(f"\n① AUC@{T}  (괄호: ideal 대비)")
    print(f"  {'data':<9}" + "".join(f"{v:>16}" for v in VARIANTS))
    for nm in DATA:
        ref = np.mean([a for a, _ in res[(nm, 'ideal')]])
        row = ""
        for v in VARIANTS:
            a = np.mean([x for x, _ in res[(nm, v)]])
            row += f"{a:>8.4f}({a - ref:+.4f})"
        print(f"  {nm:<9}{row}")
    print("\n② 레벨별 지시자 상태  clean(정확히 one-hot) / empty(선택 없음) / mixed(흐림·복수)")
    vs = [v for v in VARIANTS if v != "ideal"]
    print(f"  {'data':<9}" + "".join(f"{v:>20}" for v in vs))
    for nm in DATA:
        row = ""
        for v in vs:
            st = {k: np.mean([s[k] for _, s in res[(nm, v)]]) for k in ("clean", "empty", "mixed")}
            row += f"   {st['clean']:>4.0%}/{st['empty']:>3.0%}/{st['mixed']:>4.0%}"
        print(f"  {nm:<9}{row}")


if __name__ == "__main__":
    main()

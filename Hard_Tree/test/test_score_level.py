# Hard_Tree/test/test_score_level.py
"""
T2 출력(점수) 레벨 · 정밀도 측정 — log 범위 축소 첫 단계를 어디서 할 수 있는지 정하기 위함.

  실제 부스팅 파이프라인 (test_s9_boost 와 같은 호출 순서) 을 돌리며 레벨마다
    · 주요 암호문의 레벨: pre, g, 마스크, G_L, 점수(정규화 전 raw), compact 점수
    · compact 점수의 정밀도: 평문 참조 raw 와의 절대 오차, 1등 상대 오차,
      1등 크기, 1·2등 간격 대비 오차, 1등 일치 여부
  분할은 평문 참조 argmax 로 고정 (FHE · 평문이 같은 트리를 따르게).
  2 라운드의 평문 참조는 **복호한 g** 로 계산 → 점수 계산 자체의 오차만 본다.

  ★ 점수는 정규화하지 않는다 (normalize=False).  log 설계는 Goldschmidt 정규화를 대체한다.

실행:  PYTHONPATH=/home/junhyung/study/Data-Analysis-with-CKKS \
       python -m Hard_Tree.test.test_score_level
"""
from __future__ import annotations
import gc
import math
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import train_core as TC
from Hard_Tree.desilo import hard_core as HC

CASES = [(100, 6, 8, 3, 2), (128, 8, 8, 4, 2)]     # (n, d, n_max, D, T)
LR = 0.3


def lv(x):
    if isinstance(x, (list, tuple)):
        return lv(x[0])
    return TC._lvl(x)


def ref_raw(Xb, g, parts, cands, n):
    """평문 참조: raw(j,k) = Σ_p (G_L·H_p − G_p·H_L)²,  sel = [g, 1/n]."""
    out = np.zeros(len(cands))
    for p in parts:
        Gp, Hp = g[p].sum(), p.sum() / n
        for i, (j, k) in enumerate(cands):
            left = p & (Xb[:, j] <= k)
            GL, HL = g[left].sum(), left.sum() / n
            out[i] += (GL * Hp - Gp * HL) ** 2
    return out


def run(engine, sk, kp, n, d, n_max, D, T, seed):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, n_max, (n, d))
    y = ((Xb[:, 0] + Xb[:, 1] + rng.normal(0, 1.5, n)) > n_max - 1).astype(float)   # 신호 있는 라벨

    td = TC.train_dims(n, d, n_max, 2, engine.slot_count)
    dp = HC.p2(d)
    tm = TC.make_train_masks(engine, td)
    am = TC.make_argmax_masks(engine, td, TC.auto_mtry(engine, td))
    cands, idx = am["cands"], am["idx"]
    leaf_oh = TC.make_leaf_onehots(engine, td, dp, D)

    tiles = TC.encode_onehot(engine, kp, Xb, td)
    print(f"    onehot {lv(tiles)}", end="")
    pre = TC.prefix_cumsum(engine, tiles, kp, td, tm)
    print(f"  → prefix {lv(pre)}")
    gvec = TC.encode_sample_all(engine, kp, y - y.mean(), td)

    rows = []
    for t in range(T):
        g_in = TC.decode_sample_all(engine, gvec, sk, td, n)        # 이 라운드 입력 g (복호)
        sel2 = TC.encode_boost_sel_ct(engine, gvec, kp, td)
        masks = [TC.encode_sample_all(engine, kp, np.ones(n), td)]
        Lq = [TC.root_stats(engine, pre, masks[0], sel2, kp, td, tm)]
        Nq = [TC.broadcast_all(engine, Lq[0], kp, td, tm)]
        parts = [np.ones(n, bool)]
        splits = []
        for lvl in range(D):
            score = TC.level_score(engine, Lq, Nq, kp, td, criterion="boost",
                                   level=lvl, normalize=False)
            sc = TC.compact_scores(engine, score, am, kp, td)
            dec = np.real(np.asarray(engine.decrypt(sc, sk)))
            fhe = np.array([dec[i] for i in idx])
            ref = ref_raw(Xb, g_in, parts, cands, n)
            order = np.argsort(ref)[::-1]
            top, second = ref[order[0]], ref[order[1]]
            err = np.abs(fhe - ref).max()
            rows.append(dict(round=t + 1, level=lvl, g=lv(gvec), mask=lv(masks), GL=lv(Lq[0][0]),
                             raw=lv(score), sc=lv(sc), top=top, gap=top - second, err=err,
                             same=int(np.argmax(fhe)) == int(order[0])))
            j, k = cands[int(order[0])]
            splits.append((j, k))
            bvec = TC.extract_branch(engine, pre, j, k, kp, td)
            masks, Lq, Nq = TC.next_level(engine, pre, masks, Lq, Nq, bvec,
                                          sel2, kp, td, tm, with_stats=(lvl < D - 1))
            b = Xb[:, j] <= k
            parts = sum([[p & b, p & ~b] for p in parts], [])
            del score, sc
        G, H = TC.leaf_stats(engine, masks, gvec, kp, td, dp)
        Leaf = TC.leaf_values(engine, G, H, kp, n, D, lr=LR)
        pred = TC.scatter_leaves(engine, Leaf, masks, leaf_oh, kp, td)
        gvec = TC.update_residual(engine, gvec, pred, kp, td, lr=1.0)
        del sel2, masks, Lq, Nq, G, H, Leaf, pred
        gc.collect()
    del pre, tiles, tm, am, leaf_oh, gvec
    gc.collect()
    return rows


def main():
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    print(f"slot_count={engine.slot_count:,}")
    for (n, d, n_max, D, T) in CASES:
        print(f"\n■ n={n} d={d} n_max={n_max} D={D} T={T}")
        rows = run(engine, sk, kp, n, d, n_max, D, T, seed=n + d)
        print(f"  {'라운드':>4}{'레벨':>4} │{'g':>4}{'mask':>5}{'G_L':>5}{'raw':>5}{'sc':>4} │"
              f"{'log2 1등':>9}{'절대오차':>10}{'log2 오차':>10}{'1등 상대오차':>12}{'간격/오차':>10}{'1등일치':>7}")
        for r in rows:
            rel = r["err"] / max(r["top"], 1e-300)
            print(f"  {r['round']:>4}{r['level']:>4} │{r['g']:>4}{r['mask']:>5}{r['GL']:>5}{r['raw']:>5}{r['sc']:>4} │"
                  f"{math.log2(max(r['top'], 1e-300)):>9.1f}{r['err']:>10.1e}{math.log2(max(r['err'], 1e-300)):>10.1f}"
                  f"{rel:>12.1e}{r['gap'] / max(r['err'], 1e-300):>10.1e}{'✓' if r['same'] else '✗':>7}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()

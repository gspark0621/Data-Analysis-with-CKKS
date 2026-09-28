# Hard_Tree/test/test_s7_argmax.py
"""
S7 — FHE argmax (league).  T3 를 평문에서 암호문으로.

  ind_i = Π_{j≠i} [ s_i ≥ s_j ]        sign 을 SIMD 로 1회
    · 토너먼트(HBDT FindMinPos) ⌈log₄ns⌉, Masalha ⌈log₂log₂ns⌉ 보다 적다
    · α(top) 만 필요 — 근접 동점은 전부 패자끼리라 결과에 안 나타난다
    · ns² ≤ M 이어야 하므로 mtry 로 후보를 줄인다

  확인
    ① 지시자가 깨끗한 one-hot 인가  (최댓값 / 2등 비율)
    ② 고른 후보가 평문 argmax 와 같은가
    ③ α 를 낮추면 언제 무너지는가

실행:  python -m Hard_Tree.test.test_s7_argmax
"""
from __future__ import annotations
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import train_core as TC
from Hard_Tree.desilo import hard_core as HC
from Hard_Tree.desilo import minimax_sign as MS

HC.SIGN_IMPL = "minimax"
CASES = [(64, 4, 8, None), (100, 6, 8, None), (200, 12, 8, 5), (379, 30, 8, 5)]
ALPHAS = (8,)          # ★ 작동점.  10/12 는 레벨 부족 (19 > 10)


def run(engine, sk, kp, n, d, n_max, mtry, alpha, seed):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, n_max, (n, d))
    g = rng.normal(0, 1, n)
    td = TC.train_dims(n, d, n_max, 2, engine.slot_count)
    feats = (sorted(rng.choice(d, mtry, replace=False)) if mtry else None)
    try:
        am = TC.make_argmax_masks(engine, td, feats)
    except AssertionError as e:
        return dict(skip=str(e))
    ns = am["ns"]

    tm = TC.make_train_masks(engine, td)
    tiles = TC.encode_onehot(engine, kp, Xb, td)
    pre = TC.prefix_cumsum(engine, tiles, kp, td, tm)
    sel2 = TC.encode_boost_sel(engine, kp, g, td)
    masks = [TC.encode_sample_all(engine, kp, np.ones(n), td)]
    Lq = [TC.root_stats(engine, pre, masks[0], sel2, kp, td, tm)]
    Nq = [TC.broadcast_all(engine, Lq[0], kp, td, tm)]
    score = TC.level_score(engine, Lq, Nq, kp, td, criterion="boost",
                           level=0, normalize=True)

    # 평문 참조 (Cauchy-Schwarz 정규화)
    gs = g / np.sqrt((g * g).sum())
    ref = []
    for (j, k) in am["cands"]:
        b = Xb[:, j] <= k
        HL = b.sum() / n; Hp = 1.0
        a = gs[b].sum() * Hp - gs.sum() * HL
        ref.append(a * a)
    ref = np.array(ref) / (4.0 * 1.0 ** 3)     # Σ_p H′_p³ = 1 (루트)

    sc = TC.compact_scores(engine, score, am, kp, td,
                           tiebreak=TC.recommended_tiebreak(am["ns"], alpha))
    # ★ 선형 인덱스가 i = jj·ntp + k 로 패딩돼 있다 (am["idx"])
    dall = np.real(engine.decrypt(sc, sk))
    dec = np.array([dall[i] for i in am["idx"]])
    tb = TC.recommended_tiebreak(ns, alpha) * np.arange(ns) / ns
    err_sc = np.abs(dec - (ref + tb)).max()
    ref = ref + tb

    spec = MS.build_composite(alpha, 2.0 ** (-alpha), "depth")
    lv_sc = HC._lvl(sc)
    # 레벨이 모자라면 argmax_league 내부에서 refresh 한다
    t0 = time.time()
    try:
        ind = TC.argmax_league(engine, sc, am, kp, spec, maximize=True,
                               secret_key=sk)
    except RuntimeError as e:
        return dict(skip=str(e)[:56])
    t_am = time.time() - t0
    ivall = np.real(engine.decrypt(ind, sk))
    iv = np.array([ivall[i * am["nsp"]] for i in am["idx"]])
    order = np.argsort(-iv)
    top, second = iv[order[0]], (iv[order[1]] if ns > 1 else 0.0)
    # ★ 진단: 실제 점수 크기와 top gap → 필요한 α 가 여기서 나온다
    sv = np.sort(np.unique(np.round(dec, 14)))
    smax = float(abs(dec).max())
    tgap = float(sv[-1] - sv[-2]) if len(sv) > 1 else 0.0
    return dict(ns=ns, err_sc=err_sc, top=top, second=second,
                smax=smax, tgap=tgap,
                a_need=int(np.ceil(-np.log2(max(tgap, 1e-30)))),
                ok=(int(order[0]) == int(np.argmax(ref))),
                sdep=MS.sign_depth(spec), t=t_am, lv=HC._lvl(ind))


def main():
    print("=" * 96)
    print("S7 — FHE argmax (league, 전쌍 비교)")
    print("=" * 96)
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    print(f"  {'n':>5}{'d':>4}{'mtry':>5}{'ns':>5}{'α':>3} | "
          f"{'score max':>11}{'top gap':>11}{'필요 α':>7} | "
          f"{'ind 최대':>9}{'ind 2등':>9}{'선택':>5}{'시간(s)':>8}")
    print("  " + "-" * 92)
    for (n, d, n_max, mtry) in CASES:
        for a in ALPHAS:
            r = run(engine, sk, kp, n, d, n_max, mtry, a, seed=n + d)
            if "skip" in r:
                print(f"  {n:>5}{d:>4}{str(mtry):>5}{'':>8}{a:>3}   {r['skip']}")
                continue
            clean = (r["top"] > 0.9 and r["second"] < 0.1)
            tag = "OK" if (r["ok"] and clean) else ("고름" if r["ok"] else "FAIL")
            print(f"  {n:>5}{d:>4}{str(mtry):>5}{r['ns']:>5}{a:>3} | "
                  f"{r['smax']:>11.2e}{r['tgap']:>11.2e}{r['a_need']:>7} | "
                  f"{r['top']:>9.4f}{r['second']:>9.4f}"
                  f"{'✓' if r['ok'] else '✗':>5}{r['t']:>8.2f}  [{tag}]")
        print()
    print("  · score max / top gap : ★ 실제 값.  필요 α = ⌈−log2(top gap)⌉")
    print("  · ind 최대/2등 : 깨끗한 one-hot 이면 1.0 / 0.0")
    print("  · league 의 곱은 오차를 ns 제곱으로 증폭한다 → bit_cleaning 삽입")


if __name__ == "__main__":
    main()
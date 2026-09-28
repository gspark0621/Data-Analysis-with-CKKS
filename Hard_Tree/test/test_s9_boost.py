# Hard_Tree/test/test_s9_boost.py
"""
S9 — 부스팅 라운드 (T4-b).  잔차 갱신을 **암호문에서**.

  라운드마다
    ① 트리 학습 (T0~T6)
    ② pred = Σ_ℓ Leaf_ℓ ⊙ masks[ℓ]     리프값을 샘플 공간으로
    ③ g ← g − lr·pred                   잔차 갱신
  → 클라이언트 왕복 없이 T 라운드.

  확인
    · 라운드마다 잔차 노름이 줄어드는가 (부스팅이 실제로 도는가)
    · 암호문 잔차가 평문 잔차와 일치하는가
    · 손실이 단조 감소하는가

실행:  python -m Hard_Tree.test.test_s9_boost
"""
from __future__ import annotations
import gc
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import train_core as TC
from Hard_Tree.desilo import hard_core as HC
from Hard_Tree.desilo import minimax_sign as MS

HC.SIGN_IMPL = "minimax"
#   ⚠️ GPU 메모리: 라운드마다 masks(2^D 개)·Lq·Nq 가 살아 있어 D=4 는 무겁다.
#      실측에서 D=4 가 CUDA OOM 이 났다 → 라운드 끝에서 명시적으로 해제한다.
#      그래도 부족하면 T 를 줄이거나 n 을 낮출 것.
CASES = [(80, 4, 6, 3, 4), (100, 6, 8, 3, 4), (128, 8, 8, 4, 3)]
LR = 0.3
ALPHA = TC.ARGMAX_ALPHA


def run(engine, sk, kp, n, d, n_max, D, T, seed):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, n_max, (n, d))
    y = rng.integers(0, 2, n).astype(float)

    td = TC.train_dims(n, d, n_max, 2, engine.slot_count)
    dp = HC.p2(d)
    tm = TC.make_train_masks(engine, td)
    # ★ make_split_masks / make_split_masks_linear 는 모델 추출용이라 S9 에선
    #   쓰지 않는다.  만들어 두기만 하면 평문 마스크가 GPU 메모리를 점유해
    #   D=4 에서 OOM 의 원인이 된다.
    am = TC.make_argmax_masks(engine, td, TC.auto_mtry(engine, td))
    spec = MS.build_composite(ALPHA, 2.0 ** (-ALPHA), "depth")
    leaf_oh = TC.make_leaf_onehots(engine, td, dp, D)

    tiles = TC.encode_onehot(engine, kp, Xb, td)
    pre = TC.prefix_cumsum(engine, tiles, kp, td, tm)

    # 잔차 g₀ = y − ȳ  (암호문)
    g_pt = y - y.mean()
    gvec = TC.encode_sample_all(engine, kp, g_pt, td)
    g_ref = g_pt.copy()                                   # 평문 참조

    t0 = time.time()
    hist = []
    for t in range(T):
        # ── 트리 하나 학습 ──
        sel2 = TC.encode_boost_sel_ct(engine, gvec, kp, td)
        masks = [TC.encode_sample_all(engine, kp, np.ones(n), td)]
        Lq = [TC.root_stats(engine, pre, masks[0], sel2, kp, td, tm)]
        Nq = [TC.broadcast_all(engine, Lq[0], kp, td, tm)]
        splits = []
        for lv in range(D):
            score = TC.level_score(engine, Lq, Nq, kp, td, criterion="boost",
                                   level=lv, normalize=True)
            sc = TC.compact_scores(engine, score, am, kp, td)
            ind = TC.argmax_league(engine, sc, am, kp, spec, maximize=True,
                                   secret_key=sk)
            iv = np.real(engine.decrypt(ind, sk))
            j, k = am["cands"][int(np.argmax([iv[i * am["nsp"]] for i in am["idx"]]))]
            splits.append((j, k))
            bvec = TC.extract_branch(engine, pre, j, k, kp, td)
            masks, Lq, Nq = TC.next_level(engine, pre, masks, Lq, Nq, bvec,
                                          sel2, kp, td, tm, with_stats=(lv < D - 1))

        G, H = TC.leaf_stats(engine, masks, gvec, kp, td, dp)
        Leaf = TC.leaf_values(engine, G, H, kp, n, D, lr=LR)

        # ── ② 리프값 → 샘플 공간,  ③ 잔차 갱신 ──
        pred = TC.scatter_leaves(engine, Leaf, masks, leaf_oh, kp, td)
        gvec = TC.update_residual(engine, gvec, pred, kp, td, lr=1.0)

        # ── 평문 참조 ──
        sel = [np.ones(n, bool)]
        for (j, k) in splits:
            b = Xb[:, j] <= k
            sel = sum([[s_ & b, s_ & ~b] for s_ in sel], [])
        lam = max(1.0, n / (1 << D))
        pr = np.zeros(n)
        for s_ in sel:
            if s_.sum():
                pr[s_] = LR * g_ref[s_].sum() / (s_.sum() + lam)
        g_ref = g_ref - pr

        dec = TC.decode_sample_all(engine, gvec, sk, td, n)
        nrm = float(np.sqrt((g_ref ** 2).sum()))
        err = float(np.abs(dec - g_ref).max())
        hist.append((nrm, err, err / max(nrm, 1e-12)))

        # ★ 라운드 간 암호문 해제 — 안 하면 D=4 에서 CUDA OOM
        del sel2, masks, Lq, Nq, G, H, Leaf, pred, score, sc, ind, bvec
        gc.collect()

    out = dict(hist=hist, t=time.time() - t0, splits=splits)
    # ★ 케이스 간에도 해제한다 — 안 하면 다음 케이스에서 OOM
    del pre, tiles, tm, am, leaf_oh, gvec
    gc.collect()
    return out


def main():
    print("=" * 92)
    print("S9 — 부스팅 라운드 (암호문 잔차 갱신)")
    print("=" * 92)
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    for (n, d, n_max, D, T) in CASES:
        r = run(engine, sk, kp, n, d, n_max, D, T, seed=n + d)
        gc.collect()
        h = r["hist"]
        mono = all(h[i][0] >= h[i + 1][0] - 1e-9 for i in range(len(h) - 1))
        # ★ 절대오차는 라운드마다 bootstrap 잡음이 쌓여 커진다.
        #   부스팅에 중요한 건 **상대오차** — 잔차 노름 대비 크기다.
        ok = mono and max(r for _, _, r in h) < 1e-4
        print(f"\n■ n={n} d={d} n_max={n_max} D={D} T={T}   ({r['t']:.1f}s)")
        print(f"  {'라운드':>7}{'‖잔차‖':>12}{'절대오차':>12}{'상대오차':>12}")
        for i, (nrm, err, rel) in enumerate(h):
            print(f"  {i+1:>7}{nrm:>12.5f}{err:>12.2e}{rel:>12.2e}")
        print(f"  단조 감소 {'✓' if mono else '✗'}   [{('OK' if ok else 'FAIL')}]")
    print()
    print("  · ‖잔차‖ : 라운드마다 줄어야 부스팅이 도는 것")
    print("  · 절대/상대 오차 : 암호문 잔차 vs 평문 잔차 (상대 = 절대/‖잔차‖)")
    print("  · 절대오차는 라운드마다 bootstrap 잡음이 쌓여 커진다 — 상대오차로 본다")
    print("  · 감소폭이 작은 건 정상 — λ=max(1,n/2^D) 라 축소율 H/(H+λ)≈0.5,")
    print("    거기에 lr 이 곱해진다.  무작위 라벨이라 신호도 없다.")


if __name__ == "__main__":
    main()
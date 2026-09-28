# Hard_Tree/test/test_s5_endtoend.py
"""
S5 — 학습 → 모델 → 추론 전 경로 (symmetric, 부스팅).

  T0  subsample 마스크
  T1  파티션별 통계 (방식 C)
  T2  부스팅 분할 기준  (G′_L·H′_p − G′_p·H′_L)²   ★ 나눗셈 없음, argmax
  T3  argmax           평문 / league(FHE) 전환 가능
  T4  분기 → 모델 좌표계
  T5  마스크 갱신 (방식 C)
  T6  리프값  +lr·G/(H+λ)   Goldschmidt 역수
  ──  infer_symmetric  분기 D 개만 평가

실행:  python -m Hard_Tree.test.test_s5_endtoend
"""
from __future__ import annotations
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import train_core as TC
from Hard_Tree.desilo import hard_core as HC
from Hard_Tree.desilo import minimax_sign as MS

# ═════════ 설정 ═════════
SIGN_MODE = "minimax"       # "plain" | "minimax"
ARGMAX_MODE = "league"      # ★ 기본.  "plain" 은 argmax 만 복호해 비교하는 검증용
#   league = T3 도 FHE 로 수행 (전쌍 비교 · sign SIMD 1회)
ARGMAX_ALPHA = 8            # ★ 실제 FHE 확정 (2~7 필요, 8 이 상한)
MODEL_CLEANING = False      # refresh 노이즈 대응.  레벨 2 를 먹으므로 기본 꺼둠
N_TEST = 20

# (n, d, n_max, c, D, slot_count)
#   slot_count 를 작게 주면 큰 데이터의 레이아웃을 작은 데이터로 재현할 수 있다.
#   ★ 실제 암호문은 항상 engine.slot_count 이므로 GPU 에서는 OOM 이 난다.
#   ★ n_max 는 비용 두 축에 다르게 들어간다 (train_core.binning_report 참고)
#     T1: n_cat = 2^⌈log2 n_max⌉   argmax: ntp = 2^⌈log2(n_max−1)⌉
#     n_max ≤ 8 이 경계.  9 부터 n_cat, 10 부터 ntp 가 뛴다.
CASES = [(80,  4,  6, 2, 3, 32768), (100, 6,  8, 2, 3, 32768),
         (128, 8,  8, 2, 4, 32768), (200, 12,  8, 2, 4, 32768),
         (100, 6,  8, 2, 3,   512), (100, 6,  8, 2, 3,   128),
         (200, 5,  8, 2, 3,   128)]
RUN_LAYOUT_CASES = False

HC.SIGN_IMPL = SIGN_MODE


# ══════════════════════════════════════════════════════════════════
# 평문 참조
# ══════════════════════════════════════════════════════════════════
def ref_tree(Xb, g, D):
    """부스팅 대리 기준 (G_L·H_p − G_p·H_L)², argmax."""
    n, d = Xb.shape
    n_max = int(Xb.max()) + 1
    sel = [np.ones(n, bool)]
    splits = []
    for _ in range(D):
        best, bs = (0, 0), -np.inf
        for j in range(d):
            for k in range(n_max - 1):
                b = Xb[:, j] <= k
                tot = 0.0
                for s_ in sel:
                    HL = float((s_ & b).sum()); Hp = float(s_.sum())
                    tot += (g[s_ & b].sum() * Hp - g[s_].sum() * HL) ** 2
                if tot > bs:
                    bs, best = tot, (j, k)
        splits.append(best)
        b = Xb[:, best[0]] <= best[1]
        sel = sum([[s_ & b, s_ & ~b] for s_ in sel], [])
    return splits


def ref_leaf(Xb, g, splits, D, lam):
    """평문 리프값  +G/(H+λ)  (lr=1).  g 는 잔차."""
    n = len(Xb)
    sel = [np.ones(n, bool)]
    for (j, k) in splits:
        b = Xb[:, j] <= k
        sel = sum([[s & b, s & ~b] for s in sel], [])
    return np.array([g[s].sum() / (s.sum() + lam) for s in sel])


# ══════════════════════════════════════════════════════════════════
def run(engine, sk, kp, n, d, n_max, c, D, seed, slot_count=32768):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, n_max, (n, d))
    y = rng.integers(0, c, n).astype(float)
    resid = y - y.mean()

    td = TC.train_dims(n, d, n_max, c, slot_count)
    dp = HC.p2(d)
    if td["nct"] > 96 or (1 << D) * dp > engine.slot_count:
        return dict(skip="크기 초과")

    t0 = time.time()
    tm = TC.make_train_masks(engine, td)
    # ★ sm 은 plain 분기(extract_split)에서만 쓴다.  league 에서 만들어 두면
    #   평문 마스크가 GPU 메모리를 그냥 점유한다 (S9 에서 OOM 원인이었다).
    sm = TC.make_split_masks(engine, td, dp, D) if ARGMAX_MODE == "plain" else None
    tiles = TC.encode_onehot(engine, kp, Xb, td)
    pre = TC.prefix_cumsum(engine, tiles, kp, td, tm)

    # ★ 부스팅 선택 벡터 = [잔차/n, 1/n]
    sel2 = TC.encode_boost_sel(engine, kp, resid, td)
    submask = TC.sample_mask_pt(engine, np.ones(n), td)          # T0
    masks = [TC.encode_sample_all(engine, kp, np.ones(n), td)]
    base0 = [engine.multiply(a, b) for a, b in zip(masks[0], submask)]
    Lq = [TC.root_stats(engine, pre, base0, sel2, kp, td, tm)]
    Nq = [TC.broadcast_all(engine, Lq[0], kp, td, tm)]

    am = slm = None
    if ARGMAX_MODE == "league":
        try:
            am = TC.make_argmax_masks(engine, td, TC.auto_mtry(engine, td))
            slm = TC.make_split_masks_linear(engine, am, td, dp, D)
            spec_am = MS.build_composite(ARGMAX_ALPHA, 2.0 ** (-ARGMAX_ALPHA), "depth")
        except AssertionError as e:
            return dict(skip=str(e)[:40])

    S, n_cat, per_ct = td["S"], td["n_cat"], td["per_ct"]
    Model = Theta = None
    splits = []
    for lv in range(D):
        # ★ league 는 점수가 [0,1] 이어야 sign 이 성립 → normalize 필수.
        #   plain 은 복호해서 비교하므로 정규화가 불필요(레벨만 아낀다).
        score = TC.level_score(engine, Lq, Nq, kp, td, criterion="boost",
                               level=lv, normalize=(am is not None))

        if am is None:                                  # T3: 평문 argmax
            sd = [np.real(engine.decrypt(score[t], sk)) for t in range(td["nb_ct"])]
            best, bs = None, -np.inf
            for j in range(d):
                for k in range(n_max - 1):
                    blk = j * n_cat + k
                    t, off = blk // per_ct, blk % per_ct
                    if sd[t][off * S] > bs:
                        bs, best = sd[t][off * S], (j, k)
            splits.append(best)
            sel_t = []
            blk = best[0] * n_cat + best[1]
            for t in range(td["nb_ct"]):
                v = [0.0] * engine.slot_count
                if blk // per_ct == t:
                    v[(blk % per_ct) * S] = 1.0
                sel_t.append(engine.encrypt(v, kp.public_key))
        else:                                           # T3: FHE league argmax
            rho = TC.recommended_tiebreak(am["ns"], ARGMAX_ALPHA)
            sc = TC.compact_scores(engine, score, am, kp, td, tiebreak=rho)
            ind = TC.argmax_league(engine, sc, am, kp, spec_am, maximize=True,
                                   secret_key=sk)
            # ★ 타일로 되돌리지 않고 선형 좌표에서 바로 모델로 (2.1배 빠름)
            lin = TC._apply_diag(engine, ind, kp, am["back"])
            ivall = np.real(engine.decrypt(ind, sk))
            iv = np.array([ivall[i * am["nsp"]] for i in am["idx"]])
            splits.append(am["cands"][int(np.argmax(iv))])

        if am is None:
            Mc, Tc = TC.extract_split(engine, sel_t, sm, kp, td, lv)     # T4
        else:
            Mc, Tc = TC.extract_split_linear(engine, lin, slm, am, kp, td, lv)
        Model = Mc if Model is None else engine.add(Model, Mc)
        Theta = Tc if Theta is None else engine.add(Theta, Tc)

        bvec = TC.extract_branch(engine, pre, splits[-1][0], splits[-1][1], kp, td)
        masks, Lq, Nq = TC.next_level(engine, pre, masks, Lq, Nq, bvec,   # T5
                                      sel2, kp, td, tm, with_stats=(lv < D - 1))

    # ── T6 리프값 ──
    gvec = TC.encode_sample_all(engine, kp, resid, td)
    G, H = TC.leaf_stats(engine, masks, gvec, kp, td, dp)
    Leaf = TC.leaf_values(engine, G, H, kp, n, D, lr=1.0)
    t_tr = time.time() - t0

    Ld = np.real(engine.decrypt(Leaf, sk))
    lam = max(1.0, n / (1 << D))
    lref = ref_leaf(Xb, resid, splits, D, lam)
    err_leaf = max(abs(Ld[l * dp] - lref[l]) for l in range(1 << D))

    # ── 추론 ──
    #   ★ league 경로는 모델 레벨이 낮아진다 → refresh (small 키, 회복 10)
    from Hard_Tree.desilo.util.keypack import refresh as _rf, bit_cleaning
    if 0 <= HC._lvl(Model) < 10:
        Model = _rf(engine, Model, kp)
        Theta = _rf(engine, Theta, kp)
        # ⚠️ 일반 bootstrap 은 노이즈를 주입한다 (준형님 실측 0.999996 → 0.99839).
        #    Model 은 {0,1} 이라 bit_cleaning 으로 되돌릴 수 있지만 레벨을 2 먹는다.
        #    레벨 10 에서는 그 2 가 더 아프다 (목 실험: 오히려 악화).
        #    → 실제 FHE 에서 노이즈가 실제로 문제인지 먼저 확인할 것.
        if MODEL_CLEANING:
            Model = bit_cleaning(engine, Model, kp, n_iters=1)
    smk = HC.make_sym_masks(engine, d, D)
    lv_m = HC._lvl(Model)
    ib = (0 <= lv_m < 18) and (lv_m >= 12)
    spec = MS.build_composite(MS.alpha_for_margin(n_max), 1.0 / (2.0 * n_max),
                              "depth")
    pl = HC.plan_inference(d, D, MS.sign_depth(spec), lv_m)
    if not pl["ok"]:
        return dict(skip=pl["why"][:40])

    ok, times = 0, []
    for _ in range(N_TEST):
        xb = rng.integers(0, n_max, d)
        X = HC.encode_input(engine, kp, xb, n_max, dp)
        t1 = time.time()
        acc = HC.infer_symmetric(engine, Model, Theta, X, smk, kp, secret_key=sk,
                                 sign_spec=spec, boot_pre=pl["boot_pre"],
                                 boot_post=pl["boot_post"],
                                 boot_stage=pl.get("stage", 3))
        times.append(time.time() - t1)
        dec = np.real(engine.decrypt(acc, sk))
        got = int(np.argmin(np.abs(dec[:smk["L"] * dp:dp])))
        ok += int(got == HC.ref_leaf_symmetric(xb, splits, n_max))

    return dict(ok=ok, n=N_TEST, t_tr=t_tr, t_inf=float(np.mean(times)),
                err_leaf=err_leaf, lv_m=lv_m, R=pl["R"],
                bp=pl["boot_pre"], bs=pl["boot_post"],
                tree_ok=(splits == ref_tree(Xb, resid, D)), splits=splits)


def bt(r):
    return f"{r['bp']}+{r['bs']}"


def main():
    print("=" * 96)
    print(f"S5 — 학습 → 모델 → 추론 (symmetric, 부스팅)   "
          f"sign={SIGN_MODE}, argmax={ARGMAX_MODE}")
    print("=" * 96)
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    print(f"  {'n':>5}{'d':>4}{'n_max':>7}{'c':>3}{'D':>3} | "
          f"{'트리':>6}{'추론일치':>10}{'모델lv':>8}{'boot':>6}"
          f"{'리프오차':>11}{'학습(s)':>9}{'추론(s)':>9}")
    print("  " + "-" * 88)
    for (n, d, n_max, c, D, M_) in CASES:
        if M_ != engine.slot_count and not RUN_LAYOUT_CASES:
            continue
        try:
            r = run(engine, sk, kp, n, d, n_max, c, D, seed=n + d, slot_count=M_)
        except RuntimeError as e:
            print(f"  {n:>5}{d:>4}{n_max:>7}{c:>3}{D:>3}   {str(e)[:52]}")
            continue
        if "skip" in r:
            print(f"  {n:>5}{d:>4}{n_max:>7}{c:>3}{D:>3}   {r['skip']}")
            continue
        tag = ("OK" if (r["ok"] == r["n"] and r["tree_ok"]
                        and r["err_leaf"] < 1e-4) else "FAIL")
        print(f"  {n:>5}{d:>4}{n_max:>7}{c:>3}{D:>3} | "
              f"{'일치' if r['tree_ok'] else '불일치':>6}{r['ok']:>6}/{r['n']:<3}"
              f"{r['lv_m']:>8}{bt(r):>6}"
              f"{r['err_leaf']:>11.1e}{r['t_tr']:>9.2f}{r['t_inf']:>9.3f}  [{tag}]")
    print()
    print("  · 트리     : 학습된 분기가 평문 부스팅 대리 기준과 같은지")
    print("  · 추론일치 : 암호문 모델로 추론한 리프가 평문과 같은지")
    print("  · 리프오차 : T6  +lr·G/(H+λ)  vs 평문 (Goldschmidt 역수)")
    print("  · ARGMAX_MODE='league' 로 바꾸면 T3 도 FHE 로 수행")


if __name__ == "__main__":
    main()
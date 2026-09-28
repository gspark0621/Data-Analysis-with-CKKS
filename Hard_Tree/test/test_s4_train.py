# Hard_Tree/test/test_s4_train.py
"""
S4 — 학습 루프 검증 (T0·T1·T2·T5).  T3(argmin)은 임시로 평문.

  확인할 것
    ① prefix 누적합이 [x_j ≤ k] 와 일치하는가            (비교 없이 회전만)
    ② 파티션별 클래스 카운트 L[p][j][k][q] 가 평문과 일치
    ③ left-only MGI 점수가 평문 argmin 과 같은 분기를 고르는가
    ④ 방식 C 로 만든 자식 마스크가 평문과 일치
    ⑤ 전체 트리 구조가 평문 symmetric CART 와 일치

실행:  python -m Hard_Tree.test.test_s4_train
"""
from __future__ import annotations
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import train_core as TC

# ★ slot_count 를 작게 준 케이스는 **레이아웃 경로 확인용**이다.
#   실제 암호문은 항상 engine.slot_count(32768) 슬롯이므로, 작은 slot_count 로
#   타일을 늘리면 메모리가 그 배수만큼 늘어 GPU 에서 OOM 이 난다.
#   목(mock) 이나 CPU 에서만 켤 것.
RUN_LAYOUT_CASES = False

# (n, d, n_max, c, D, slot_count)
#   slot_count 를 작게 주면 큰 데이터의 레이아웃(feature 분할·샘플 청크)을
#   작은 데이터로 재현할 수 있다.
CASES = [(80,   4,  6, 2, 3, 32768),   # 1 타일
         (100,  6,  8, 2, 3, 32768),
         (64,   5,  4, 3, 2, 32768),   # 다중 클래스
         (256, 12, 10, 2, 4, 32768),
         (512,  8, 16, 2, 4, 32768),
         (200, 30, 18, 2, 3, 32768),   # cancer 규모
         (100,  6,  8, 2, 3,  1024),   # ★ per_ct=8=n_cat  경계
         (100,  6,  8, 2, 3,   512),   # ★ split_feat (per_ct=4 < n_cat=8)
         (100,  6,  8, 2, 3,   128),   # ★ split_feat (per_ct=1)  — adult 형태
         (200,  5,  8, 2, 3,   128),   # ★ 샘플 청크 2개 (n_pad 256 > M 128)
         (400,  4,  4, 2, 3,   128)]   # ★ 청크 4개 + split_feat
SUB = 1.0


# ── 평문 참조 ─────────────────────────────────────────────────────
def ref_counts(Xb, y, sel, j, k, c):
    m = sel & (Xb[:, j] <= k)
    return np.array([np.sum(m & (y == q)) for q in range(c)], float)


def ref_mgi(Xb, y, sel_list, d, n_max, c):
    """레벨 전체 파티션에 대한 left-only MGI 합.  낮을수록 좋음."""
    sc = np.zeros((d, n_max))
    for j in range(d):
        for k in range(n_max - 1):
            tot = 0.0
            for sel in sel_list:
                Lq = ref_counts(Xb, y, sel, j, k, c)
                Nq = np.array([np.sum(sel & (y == q)) for q in range(c)], float)
                L, N = Lq.sum(), Nq.sum()
                tot += 2 * (L * L - N * L + (Nq * Lq).sum() - (Lq * Lq).sum())
            sc[j, k] = tot
    return sc


def ref_tree(Xb, y, D, c):
    n, d = Xb.shape
    n_max = int(Xb.max()) + 1
    sel = [np.ones(n, bool)]
    splits = []
    for _ in range(D):
        s = ref_mgi(Xb, y, sel, d, n_max, c)
        s[:, n_max - 1] = np.inf
        j, k = np.unravel_index(np.argmin(s), s.shape)
        splits.append((int(j), int(k)))
        nxt = []
        for m in sel:
            b = Xb[:, j] <= k
            nxt += [m & b, m & ~b]
        sel = nxt
    return splits


# ── FHE ───────────────────────────────────────────────────────────
def run(engine, sk, kp, n, d, n_max, c, D, seed, slot_count=32768):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, n_max, (n, d))
    y = rng.integers(0, c, n)
    td = TC.train_dims(n, d, n_max, c, slot_count)
    if td["nct"] > 96:
        return dict(skip=f"타일 {td['nct']}개 — 너무 큼")

    t0 = time.time()
    tm = TC.make_train_masks(engine, td)              # ★ 평문 마스크 1회 생성
    tiles = TC.encode_onehot(engine, kp, Xb, td)
    pre = TC.prefix_cumsum(engine, tiles, kp, td, tm)
    t_enc = time.time() - t0

    # ① prefix 검증
    n_pad, n_cat, per_ct = td["S"], td["n_cat"], td["per_ct"]
    err_pre = 0.0
    S = td["S"]
    for ch in range(td["n_chunk"]):                     # ★ 청크마다
        s0 = ch * S
        m = min(n, s0 + S) - s0
        if m <= 0:
            continue
        for t in range(td["nb_ct"]):
            dec = np.real(engine.decrypt(pre[TC._tile(td, t, ch)], sk))
            lo, _ = TC._blk_range(td, t)
            for blk in range(lo, min(td["blocks"], lo + per_ct)):
                j, k = divmod(blk, n_cat)
                if j >= d or k >= n_max:
                    continue
                got = dec[(blk - lo) * S:(blk - lo) * S + m]
                err_pre = max(err_pre, np.abs(got - (Xb[s0:s0 + m, j] <= k)).max())

    # 클래스 지시자 (샘플 축)
    ycls = [TC.encode_sample_all(engine, kp, (y == q).astype(float), td)
            for q in range(c)]
    submask = TC.sample_mask_pt(engine, np.ones(n), td)

    # ── 레벨 루프  (★ 방식 C) ──
    masks = [TC.encode_sample_all(engine, kp, np.ones(n), td)]
    base0 = [engine.multiply(a, b) for a, b in zip(masks[0], submask)]
    Lq_list = [TC.root_stats(engine, pre, base0, ycls, kp, td, tm)]
    Nq_list = [TC.broadcast_all(engine, Lq_list[0], kp, td, tm)]
    sel_ref = [np.ones(n, bool)]
    splits, err_L, err_split = [], 0.0, 0
    for lv in range(D):
        # T2 : left-only MGI 를 레벨 전체에서 합산
        score = TC.level_score(engine, Lq_list, Nq_list, kp, td)

        # ② 카운트 검증 (파티션 0)
        dec0 = np.real(engine.decrypt(Lq_list[0][0][0], sk))
        for j in range(min(d, 3)):
            for k in range(min(n_max - 1, 3)):
                blk = j * n_cat + k
                if blk >= per_ct:
                    continue
                exp = ref_counts(Xb, y, sel_ref[0], j, k, c)[0]
                err_L = max(err_L, abs(dec0[blk * n_pad] - exp))

        # T3 : 임시 평문 argmin
        sdec = [np.real(engine.decrypt(score[t], sk)) for t in range(td["nb_ct"])]
        best, bs = None, np.inf
        for j in range(d):
            for k in range(n_max - 1):
                blk = j * n_cat + k
                t, off = blk // per_ct, blk % per_ct
                v = sdec[t][off * n_pad]
                if v < bs:
                    bs, best = v, (j, k)
        splits.append(best)

        # ③ 평문 argmin 과 일치?
        s_ref = ref_mgi(Xb, y, sel_ref, d, n_max, c)
        s_ref[:, n_max - 1] = np.inf
        jr, kr = np.unravel_index(np.argmin(s_ref), s_ref.shape)
        err_split += int(best != (int(jr), int(kr)))

        # T1+T5 : 방식 C 로 다음 레벨 (왼쪽만 계산, 오른쪽은 뺄셈)
        bvec = TC.extract_branch(engine, pre, best[0], best[1], kp, td)
        masks, Lq_list, Nq_list = TC.next_level(
            engine, pre, masks, Lq_list, Nq_list, bvec, ycls, kp, td, tm,
            with_stats=(lv < D - 1))       # ★ 마지막 레벨은 마스크만
        b_ref = Xb[:, best[0]] <= best[1]
        sel_ref = sum([[m & b_ref, m & ~b_ref] for m in sel_ref], [])

    # ④ 마스크 검증
    err_m = 0.0
    for mp, sr in zip(masks[:4], sel_ref[:4]):
        for ch in range(td["n_chunk"]):                 # ★ 청크마다
            s0 = ch * S
            m = min(n, s0 + S) - s0
            if m <= 0:
                continue
            dec = np.real(engine.decrypt(mp[ch], sk))[:m]
            err_m = max(err_m, np.abs(dec - sr[s0:s0 + m].astype(float)).max())

    # ⑤ 트리 구조
    ref = ref_tree(Xb, y, D, c)
    return dict(t=time.time() - t0, t_enc=t_enc, nct=td["nct"],
                per_ct=td["per_ct"], nch=td["n_chunk"], sf=td["split_feat"],
                err_pre=err_pre, err_L=err_L, err_m=err_m,
                split_mismatch=err_split, tree_ok=(splits == ref),
                splits=splits, ref=ref)


def main():
    print("=" * 88)
    print("S4 — 학습 루프 (T0·T1·T2·T5).  T3 는 임시 평문 argmin")
    print("=" * 88)
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    print(f"  {'n':>5}{'d':>4}{'n_max':>7}{'c':>3}{'D':>3}{'레이아웃':>14} | "
          f"{'prefix':>9}{'카운트':>9}{'마스크':>9}{'분기불일치':>11}{'트리':>7}{'시간(s)':>9}")
    print("  " + "-" * 84)
    for (n, d, n_max, c, D, M_) in CASES:
        if M_ != engine.slot_count and not RUN_LAYOUT_CASES:
            continue
        r = run(engine, sk, kp, n, d, n_max, c, D, seed=n + d, slot_count=M_)
        if "skip" in r:
            print(f"  {n:>5}{d:>4}{n_max:>7}{c:>3}{D:>3}   {r['skip']}")
            continue
        lay = f"per{r['per_ct']}/ch{r['nch']}" + ("/SF" if r['sf'] else "")
        tag = "OK" if (r["err_pre"] < 1e-6 and r["err_L"] < 1e-4
                       and r["err_m"] < 1e-6 and r["tree_ok"]) else "FAIL"
        print(f"  {n:>5}{d:>4}{n_max:>7}{c:>3}{D:>3}{lay:>14} | "
              f"{r['err_pre']:>9.1e}{r['err_L']:>9.1e}{r['err_m']:>9.1e}"
              f"{r['split_mismatch']:>11}{'일치' if r['tree_ok'] else '불일치':>7}"
              f"{r['t']:>9.2f}  [{tag}]")
        if not r["tree_ok"]:
            print(f"        FHE {r['splits']}")
            print(f"        평문 {r['ref']}")
    print()
    print("  · prefix  : 비교 없이 회전만으로 [x_j ≤ k] 를 만드는지")
    print("  · 카운트  : 파티션별 클래스 카운트 L[p][j][k][q]")
    print("  · 마스크  : 방식 C (오른쪽 = 부모 − 왼쪽) 결과")
    print("  · 트리    : 전체 구조가 평문 symmetric CART 와 같은지")


if __name__ == "__main__":
    main()
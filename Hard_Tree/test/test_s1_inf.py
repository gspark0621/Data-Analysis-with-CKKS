# soft_tree/test/test_s1_inference.py
"""
S1 — 비교 기반 추론을 실제 CKKS 에서 검증.

  확인할 것
    ① 회전 규약이 시뮬레이션(np.roll)과 실제로 일치하는가
    ② 리프 선택이 평문 트리와 일치하는가        (목표: 상관/일치율 100%)
    ③ 레벨을 실제로 몇 개 먹는가                (예산 재계산용)

  sign 은 hard_core.SIGN_IMPL = "plain" (복호→np.sign→재암호).
  ★ 검증 전용. 회전 규약을 먼저 확정한 뒤 S2 에서 minimax 로 교체.

실행:  python -m Hard_Tree.test.test_s1_inference
"""
from __future__ import annotations
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import hard_core as HC
from Hard_Tree.desilo import minimax_sign as MS

HC.SIGN_IMPL = "minimax"

# ═══════════════ 설정 ═══════════════
CASES = [           # (d, n_max, depth)
    (4,   9,  4),   # Iris 규모 — 가장 작게 시작
    (13, 11,  5),   # Wine
    (30, 18,  5),   # Cancer (Scott)
    (30, 18,  8),   # Cancer, 깊이 확장  ← HBDT 가 못 가는 지점
    (3, 101, 10),   # Sepsis 모양, 깊이 10
]
N_TRIALS = 20       # 케이스당 샘플 수
SEED = 0
# ════════════════════════════════════


def lvl(ct):
    try:
        return ct.level
    except Exception:
        return "?"


def run_case(engine, sk, kp, d, n_max, depth, n_trials, seed):
    rng = np.random.default_rng(seed)
    dm = HC.dims(d, n_max, depth)
    dp, NI, L = dm["dp"], dm["NI"], dm["L"]
    need = (1 << depth) * dp
    if need > engine.slot_count:
        return dict(skip=f"슬롯 초과 {need} > {engine.slot_count}")

    # ── 랜덤 트리 (평문) ──
    feat = rng.integers(0, d, NI)
    thr_bin = rng.integers(0, n_max - 1, NI)      # k ∈ [0, n_max-2]

    t0 = time.time()
    tree = HC.encode_model_plain(engine, kp, feat, thr_bin, d, n_max, depth)
    spec = MS.build_composite(MS.alpha_for_margin(n_max), 1.0 / (2.0 * n_max), "depth")
    sdep = MS.sign_depth(spec)
    R, nb = HC.auto_spread_bits(d, depth, sdep)
    masks = HC.make_infer_masks(engine, dm, seq_bits=R)
    t_setup = time.time() - t0

    lv_model = lvl(tree.Model)
    ok, times, lv_acc = 0, [], None
    for _ in range(n_trials):
        xb = rng.integers(0, n_max, d)
        X = HC.encode_input(engine, kp, xb, n_max, dp)

        t1 = time.time()
        acc = HC.infer_packed(engine, tree, X, masks, kp, secret_key=sk,
                              sign_spec=spec, n_boot=nb)
        times.append(time.time() - t1)
        if lv_acc is None:
            lv_acc = lvl(acc)

        dec = np.real(engine.decrypt(acc, sk))
        got = int(np.argmin(np.abs(dec[:L * dp:dp])))
        exp = HC.ref_leaf(xb, feat, thr_bin, depth)
        ok += int(got == exp)

    return dict(ok=ok, n=n_trials, need=need, t_setup=t_setup, R=R, nb=nb,
                t_inf=float(np.mean(times)), lv_model=lv_model, lv_acc=lv_acc,
                depth_used=(lv_model - lv_acc) if isinstance(lv_acc, int) else "?")


def main():
    print("=" * 78)
    print("S1 — 비교 기반 추론 검증 (sign = plain, 회전 규약 확인용)")
    print("=" * 78)
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    print(f"  슬롯 {engine.slot_count},  최대 깊이 (d=30): D≤{HC.max_depth(30, engine.slot_count)}")
    print()
    print(f"  {'d':>4}{'n_max':>7}{'D':>4}{'일치':>10}{'슬롯':>8}"
          f"{'추론(s)':>10}{'레벨':>7}{'R':>4}{'boot':>6}{'준비(s)':>9}")
    print("  " + "-" * 78)

    for (d, n_max, depth) in CASES:
        r = run_case(engine, sk, kp, d, n_max, depth, N_TRIALS, SEED + d * 100 + depth)
        if "skip" in r:
            print(f"  {d:>4}{n_max:>7}{depth:>4}   {r['skip']}")
            continue
        tag = "OK" if r["ok"] == r["n"] else "FAIL"
        print(f"  {d:>4}{n_max:>7}{depth:>4}{r['ok']:>6}/{r['n']:<3}{r['need']:>8}"
              f"{r['t_inf']:>10.3f}{str(r['depth_used']):>7}{r['R']:>4}{r['nb']:>6}"
              f"{r['t_setup']:>9.2f}  [{tag}]")

    print()
    print("  판정")
    print("   · 전부 OK          → 회전 규약 확정.  S2(minimax sign) 로 진행")
    print("   · 일부만 FAIL      → 해당 (d, D) 조합의 마스크/대각 생성 확인")
    print("   · 전부 FAIL        → 회전 부호 규약 재확인 (rotate(+s) = np.roll(+s)?)")
    print()
    print("  ※ 레벨소비는 sign 을 제외한 값.  실제 예산 = 이 값 + dep(minimax)")


if __name__ == "__main__":
    main()
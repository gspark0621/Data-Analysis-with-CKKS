# Hard_Tree/test/test_s3_extract.py
"""
S3 — 학습부 모델 갱신 검증.

  FindMinPos 출력 (HBDT 좌표계)  →  (onehot_j, θ)  (새 좌표계)  →  추론

  ★ 검증 방식: 추출한 모델로 추론한 리프가 평문 트리와 일치하는지.
    S1 의 encode_model_plain 을 extract_model 이 대체할 수 있어야 한다.

실행:  python -m Hard_Tree.test.test_s3_extract
"""
from __future__ import annotations
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import hard_core as HC
from Hard_Tree.desilo import minimax_sign as MS

SIGN_MODE = "minimax"
HC.SIGN_IMPL = SIGN_MODE
CASES = [(4, 9, 4), (13, 11, 5), (30, 18, 5), (30, 18, 8), (3, 101, 10)]
EXTRACT_MAX_DEPTH = 4      # 추출 깊이 상한.  낮출수록 모델 레벨이 높아진다
                           #   None = 제한 없음(깊이 최대, 회전 최소)
N_TRIALS = 20


def encode_argmin(engine, kp, feat, thr, dm, level, q0, nq):
    """FindMinPos 출력 시뮬레이션 (실제로는 학습 중 암호문으로 나온다)."""
    sc, n_pad, blk = engine.slot_count, dm["n_pad"], dm["blk"]
    base = (1 << level) - 1
    v = [0.0] * sc
    for q in range(nq):
        i = base + q0 + q
        v[q * blk + int(feat[i]) * n_pad + int(thr[i])] = 1.0
    return engine.encrypt(v, kp.public_key)


def run(engine, sk, kp, d, n_max, depth, seed):
    dm = HC.dims(d, n_max, depth)
    dp, NI, L, blk = dm["dp"], dm["NI"], dm["L"], dm["blk"]
    if (1 << depth) * dp > engine.slot_count:
        return None
    rng = np.random.default_rng(seed)
    feat = rng.integers(0, d, NI)
    thr = rng.integers(0, n_max - 1, NI)

    spec = MS.build_composite(MS.alpha_for_margin(n_max), 1.0 / (2.0 * n_max), "depth")
    sdep = MS.sign_depth(spec)
    masks = None   # 모델 레벨을 안 뒤에 R 을 정한다

    # ── 레벨별 추출 후 누적 ──
    t0 = time.time()
    Model = Theta = None
    ex_depth, buf_ct = 0, 0
    for lv in range(depth):
        chunks = HC.extract_chunks(engine, dm, lv)
        buf_ct = max(buf_ct, len(chunks))
        for (q0, nq) in chunks:
            ex = HC.make_extract_masks(engine, dm, lv, q0, nq,
                                       max_depth=EXTRACT_MAX_DEPTH)
            ex_depth = max(ex_depth, ex["depth"])
            A = encode_argmin(engine, kp, feat, thr, dm, lv, q0, nq)
            M, T = HC.extract_model(engine, A, ex, dm, kp)
            Model = M if Model is None else engine.add(Model, M)
            Theta = T if Theta is None else engine.add(Theta, T)
    t_ex = time.time() - t0

    # ── 추출한 모델로 추론 ──
    tree = HC.EncHardTreePacked(Model, Theta, None, dm)
    lv_raw = HC._lvl(Model)
    plan = HC.plan_inference(d, depth, sdep, lv_raw)      # ★ 재암호화 없이
    if not plan["ok"]:
        return dict(skip=plan["why"])
    R = plan["R"]
    masks = HC.make_infer_masks(engine, dm, seq_bits=R)
    t_fin = 0.0
    ok, times = 0, []
    for _ in range(N_TRIALS):
        xb = rng.integers(0, n_max, d)
        X = HC.encode_input(engine, kp, xb, n_max, dp)
        t1 = time.time()
        acc = HC.infer_packed(engine, tree, X, masks, kp, secret_key=sk,
                              sign_spec=spec, **HC.plan_kwargs(plan))
        times.append(time.time() - t1)
        dec = np.real(engine.decrypt(acc, sk))
        got = int(np.argmin(np.abs(dec[:L * dp:dp])))
        ok += int(got == HC.ref_leaf(xb, feat, thr, depth))
    return dict(ok=ok, n=N_TRIALS, t_ex=t_ex, t_inf=float(np.mean(times)),
                ex_depth=ex_depth, R=R, buf=buf_ct, lv_raw=lv_raw, t_fin=t_fin,
                need=HC.required_level(sdep, R),
                bp=plan["boot_pre"], bs=plan["boot_post"],
                stage=plan.get("stage"))


def boot_tag(r):
    t = f"{r['bp']}+{r['bs']}"
    return t + (f"(sc{r['stage']})" if r['bs'] else "")


def main():
    print("=" * 82)
    print(f"S3 — 학습부 모델 갱신 (argmin → (onehot_j, θ) → 추론),  sign = {SIGN_MODE}")
    print("=" * 82)
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    print(f"  {'d':>4}{'n_max':>7}{'D':>4}{'일치':>10}{'추출(s)':>9}"
          f"{'모델lv':>8}{'R':>4}{'boot':>11}{'추론(s)':>9}{'추출깊이':>9}{'전이ct':>7}")
    print("  " + "-" * 82)
    for (d, n_max, depth) in CASES:
        r = run(engine, sk, kp, d, n_max, depth, d * 100 + depth)
        if r is None or "skip" in (r or {}):
            print(f"  {d:>4}{n_max:>7}{depth:>4}   {(r or {}).get('skip','슬롯 초과')}")
            continue
        tag = "OK" if r["ok"] == r["n"] else "FAIL"
        print(f"  {d:>4}{n_max:>7}{depth:>4}{r['ok']:>6}/{r['n']:<3}{r['t_ex']:>9.2f}"
              f"{r['lv_raw']:>8}{r['R']:>4}"
              f"{boot_tag(r):>11}"
              f"{r['t_inf']:>9.3f}"
              f"{r['ex_depth']:>9}{r['buf']:>7}  [{tag}]")
    print()
    print("  · 전부 OK  → extract_model 확정.  S4(학습 루프)로 진행")
    print("  · 모델lv  = 학습 산출 모델의 레벨 (fresh 26 이 아니다)")
    print("  · boot    = sign 직전 + 직후 refresh 횟수.  재암호화 없이 서버 단독")
    print("  · R       = 클수록 임의 shift 회전이 줄지만 깊이를 먹는다")
    print("  · 추출깊이 = 레벨당 소모.  argmin 출력 레벨에서 이만큼 빠진다")
    print("  · 전이ct  = argmin 버퍼에 필요한 암호문 수 (레벨 단위, 누적 아님)")


if __name__ == "__main__":
    main()
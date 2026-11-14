# Hard_Tree/test/test_dp_sign2.py
"""
FHE 단위 테스트 — 확장 DP (성분 뒤 전이: '-' 부트 없음 / 'R' refresh / 'Ss' sign_bootstrap).

  test_dp_sign.py 는 '모든 성분 뒤 sign_bootstrap' 계획만 확인했다.
  여기서는 '-'(스케일 폴딩 후 바로 다음 성분) 와 'R' 을 쓰는 계획을 실제 FHE 로 돌려
    · |x| ≥ δ 에서 |out − sgn x| ≤ τ
    · FHE ↔ 평문 체인 모델 차이
    · 시간 (DP 추정 대비), 레벨
  를 본다.  입력은 refresh 를 거치지 않은 암호문 (fresh → 곱셈으로 L_in 까지 내림).

실행:  PYTHONPATH=/home/junhyung/study/Data-Analysis-with-CKKS \
       python -m Hard_Tree.test.test_dp_sign2
"""
import math
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import minimax_sign as MS

PLANS = [  # (이름, α, τ, L_in, allow_refresh)
    ("α10 L8", 10, 2.0 ** -10, 8, False),
    ("α12 L13", 12, 2.0 ** -10, 13, False),
    ("α14 L16", 14, 2.0 ** -10, 16, False),
    ("α16 L16", 16, 2.0 ** -10, 16, False),
    ("α16 L20", 16, 2.0 ** -10, 20, False),
    ("α18 L20", 18, 2.0 ** -10, 20, False),
    ("α20 L20", 20, 2.0 ** -10, 20, False),
    ("α20 L26", 20, 2.0 ** -10, 26, False),
    ("α24 L26", 24, 2.0 ** -10, 26, False),
    ("α16 L10 +R", 16, 2.0 ** -10, 10, True),
    ("α24 L12 +R", 24, 2.0 ** -10, 12, True),
]

# 계획 먼저 (Remez 가 느리므로 GPU 키 생성 전에)
specs = []
for name, a, tau, L, r in PLANS:
    t0 = time.time()
    sp = MS.build_composite_sb(2.0 ** -a, tau, level_in=L, allow_refresh=r)
    specs.append(sp)
    print(f"  {name:<11} {MS.plan_str(sp):<26} DP {sp['cost_ms']/1000:.2f}s  ({time.time()-t0:.0f}s)", flush=True)

engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
SC = engine.slot_count
ONE = engine.encode([1.0] * SC)
N = min(SC, 1 << 14)


def enc(v):
    v = np.asarray(v, float)
    return engine.encrypt(list(v) + [0.0] * (SC - len(v)), kp.public_key)


def dec(ct, n):
    return np.real(np.asarray(engine.decrypt(ct, sk)))[:n]


def drop_to(ct, target):
    while ct.level > target:
        ct = engine.multiply(ct, ONE)
    return ct


def make_input(delta, n):
    k = n // 8
    pos = np.concatenate([np.geomspace(delta, 1.0, 3 * k), np.linspace(delta, 1.0, k)])
    xs = np.concatenate([pos, -pos])
    return np.concatenate([xs, np.linspace(-delta, delta, n - len(xs))]), len(xs)


print(f"\nslot_count={SC:,}  fresh level={enc([0.0]).level}")
print("=" * 96)
ok_all = True
for (name, a, tau, L, r), sp in zip(PLANS, specs):
    delta = 2.0 ** -a
    xv, nout = make_input(delta, N)
    ct = drop_to(enc(xv), L)
    try:
        t0 = time.time()
        out = MS.minimax_sign01(engine, ct, kp, spec=sp, to_01=False)
        ms = (time.time() - t0) * 1000
        y = dec(out, N)
        e_true = np.abs(y[:nout] - np.sign(xv[:nout])).max()
        e_model = np.abs(y - MS.eval_composite_plain_sb(xv, sp)).max()
        ok = e_true <= tau
        ok_all &= ok
        print(f"  {name:<11} {MS.plan_str(sp):<26} lv {ct.level}→{out.level}  {ms/1000:5.2f}s "
              f"(DP {sp['cost_ms']/1000:.2f})  |out−sgn| {e_true:.2e} (≤{tau:.1e})  "
              f"vs 평문 {e_model:.1e}  {'OK' if ok else 'FAIL'}", flush=True)
    except Exception as ex:
        ok_all = False
        print(f"  {name:<11} {MS.plan_str(sp):<26} 실패 — {type(ex).__name__}: {ex}", flush=True)
print("\n요약:", "모두 OK" if ok_all else "실패 있음 (위 확인)")

# Hard_Tree/test/test_dp_sign.py
"""
FHE 단위 테스트 — 성분 사이 sign_bootstrap DP (minimax_sign.build_composite_sb).

  A. sign_bootstrap 전달 함수:  z ∈ [−2, 2] 를 넣어 sin(πz/2) 모델과 비교 (stage 3/4/5)
     ★ DP 의 핵심 가정.  기존 실측(7.7e-7)은 ±1 근처 입력에서만 잰 값이다.
  B. 체비셰프 차수별 실제 레벨 소모 vs cheb_dep (DP 제약식)
  C. DP 계획을 minimax_sign01 로 실행 → 평문 모델 · 참 부호와 비교, 시간 측정
     기존 경로(pick_degrees + refresh 사이) α=10 도 같은 입력으로 비교

실행:  PYTHONPATH=/home/junhyung/study/Data-Analysis-with-CKKS \
       python -m Hard_Tree.test.test_dp_sign
"""
import math
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine, sign_refresh
from Hard_Tree.desilo import minimax_sign as MS

engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
SC = engine.slot_count
rlk = kp.relinearization_key
ONE = engine.encode([1.0] * SC)


def enc(v):
    v = np.asarray(v, float)
    return engine.encrypt(list(v) + [0.0] * (SC - len(v)), kp.public_key)


def dec(ct, n):
    return np.real(np.asarray(engine.decrypt(ct, sk)))[:n]


def drop_to(ct, target):
    while ct.level > target:
        ct = engine.multiply(ct, ONE)
    return ct


print(f"slot_count={SC:,}   fresh level={enc([0.0]).level}")

# ══════════════════════════════════════════════════════════════════
# A. sign_bootstrap 전달 함수
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("A. sign_bootstrap(z) vs sin(πz/2)")
print("=" * 72)
z = np.concatenate([np.linspace(-2.0, 2.0, SC // 2),
                    np.geomspace(1e-3, 1.0, SC // 4), -np.geomspace(1e-3, 1.0, SC // 4)])
z = z[:SC]
ref = np.sin(np.pi * z / 2.0)
REG = [("|z|<0.05", np.abs(z) < 0.05),
       ("0.05~0.5", (np.abs(z) >= 0.05) & (np.abs(z) < 0.5)),
       ("0.5~1.5", (np.abs(z) >= 0.5) & (np.abs(z) <= 1.5)),
       ("1.5~1.98", (np.abs(z) > 1.5) & (np.abs(z) <= 1.98)),
       ("전체≤1.98", np.abs(z) <= 1.98)]
print(f"  {'stage':>5}{'in→out':>9}{'ms':>9}" + "".join(f"{r:>12}" for r, _ in REG))
for st in (3, 4, 5):
    ct = drop_to(enc(z), st + 1)
    t0 = time.time()
    out = sign_refresh(engine, ct, kp, stage_count=st)
    ms = (time.time() - t0) * 1000
    e = np.abs(dec(out, SC) - ref)
    print(f"  {st:>5}{f'{ct.level}→{out.level}':>9}{ms:>9.1f}"
          + "".join(f"{e[m].max():>12.2e}" for _, m in REG))
# 출력 하한 확인: 작은 z 에서의 기울기 sin(πz/2)/z → π/2
ct = drop_to(enc(z), 6)
out = dec(sign_refresh(engine, ct, kp, stage_count=5), SC)
m = (z > 0.01) & (z < 0.05)
print(f"  기울기 out/z (z∈(0.01,0.05)) = {np.median(out[m] / z[m]):.4f}   (π/2 = {math.pi/2:.4f})")

# ══════════════════════════════════════════════════════════════════
# B. 체비셰프 레벨 소모
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("B. 체비셰프 차수별 레벨 소모  (DP 가정 cheb_dep = ⌈log2(d+1)⌉ + 1)")
print("=" * 72)
rng = np.random.default_rng(0)
x = rng.uniform(-0.9, 0.9, 64)
for d in MS.DP_DEGREES:
    cf, _ = MS._remez_unit(d, 2.0 ** -6)
    ct = drop_to(enc(x), 14)
    t0 = time.time()
    out = engine.evaluate_chebyshev_polynomial(engine.intt(ct), list(cf), rlk)
    ms = (time.time() - t0) * 1000
    err = np.abs(dec(out, len(x)) - np.polynomial.chebyshev.chebval(x, cf)).max()
    used = ct.level - out.level
    print(f"  deg {d:>3}  실측 {used}  가정 {MS.cheb_dep(d)}  {'OK' if used <= MS.cheb_dep(d) else '★초과'}"
          f"   {ms:>7.1f} ms (추정 {MS.cheb_eval_ms(d):>6.1f})   평가오차 {err:.1e}")

# ══════════════════════════════════════════════════════════════════
# C. DP 계획 end-to-end
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("C. minimax_sign01 (inter='sign') end-to-end   입력 |x|≥δ 에서 |out−sgn x| ≤ τ ?")
print("=" * 72)


def make_input(delta, n):
    """δ 근처를 촘촘히: ±geomspace(δ,1) + 균등 + dead zone 안쪽 일부."""
    k = n // 8
    pos = np.concatenate([np.geomspace(delta, 1.0, 3 * k),
                          np.linspace(delta, 1.0, k)])
    xs = np.concatenate([pos, -pos])
    dz = np.linspace(-delta, delta, n - len(xs))       # dead zone (참고용)
    return np.concatenate([xs, dz]), len(xs)


PLANS = [  # (이름, δ, τ, L_in)
    ("argmax τ2^-10 L8", 2.0 ** -10, 2.0 ** -10, 8),
    ("argmax τ2^-10 L10", 2.0 ** -10, 2.0 ** -10, 10),
    ("argmax τ2^-10 L12", 2.0 ** -10, 2.0 ** -10, 12),
    ("argmax τ2^-8 L8", 2.0 ** -10, 2.0 ** -8, 8),
    ("infer δ1/16 L12", 1.0 / 16, 2.0 ** -6, 12),
]
N = min(SC, 1 << 14)
rows = []
for name, delta, tau, L in PLANS:
    spec = MS.build_composite_sb(delta, tau, level_in=L)
    need = MS.sign_depth(spec, to_01=False)
    xv, nout = make_input(delta, N)
    ct = drop_to(enc(xv), L)
    try:
        t0 = time.time()
        out = MS.minimax_sign01(engine, ct, kp, spec=spec, to_01=False)
        ms = (time.time() - t0) * 1000
        y = dec(out, N)
        e_true = np.abs(y[:nout] - np.sign(xv[:nout])).max()
        e_model = np.abs(y - MS.eval_composite_plain_sb(xv, spec)).max()
        e_plain, _ = MS.verify_plain_sb(spec, verbose=False)
        ok = e_true <= tau
        rows.append((name, spec, ct.level, out.level, ms, e_true, e_model, e_plain, ok))
        print(f"  {name:<18} degs={spec['degs']} st={spec['stages']}  lv {ct.level}→{out.level}"
              f" (필요 {need})  {ms/1000:5.2f}s (DP 추정 {spec['cost_ms']/1000:.2f}s)")
        print(f"  {'':<18} 오차 vs sgn {e_true:.2e} (허용 {tau:.2e}, 평문 {e_plain:.2e})"
              f"  vs 평문모델 {e_model:.2e}  → {'OK' if ok else 'FAIL'}")
    except Exception as ex:
        print(f"  {name:<18} degs={spec['degs']} st={spec['stages']}  실패 — {type(ex).__name__}: {ex}")

# 기존 경로 α=10 (refresh 사이, 마지막 sign_bootstrap) — fresh 레벨에서
print("\n  ── 기존 경로 (pick_degrees, refresh 사이) ──")
for alpha in (10,):
    delta = 2.0 ** -alpha
    spec = MS.build_composite(alpha, delta, "depth")
    xv, nout = make_input(delta, N)
    ct = enc(xv)
    try:
        t0 = time.time()
        out = MS.minimax_sign01(engine, ct, kp, spec=spec, to_01=False)
        ms = (time.time() - t0) * 1000
        y = dec(out, N)
        e_true = np.abs(y[:nout] - np.sign(xv[:nout])).max()
        print(f"  α={alpha} degs={spec['degs']}  lv {ct.level}→{out.level}  {ms/1000:5.2f}s"
              f" (추정 {MS.legacy_cost_ms(alpha)[1]/1000:.2f}s)  오차 vs sgn {e_true:.2e}")
    except Exception as ex:
        print(f"  α={alpha} 실패 — {type(ex).__name__}: {ex}")

print("\n요약:", "모두 OK" if rows and all(r[-1] for r in rows) else "실패 있음 (위 확인)")

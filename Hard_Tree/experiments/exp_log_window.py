# Hard_Tree/experiments/exp_log_window.py
"""
평문 프로토타입 — log2(u) 를 CKKS 친화적으로 계산하는 '병렬 창 + 순차 단계' 범위 축소.

  HEaaN-NB ApproxLog 는 x 를 2^-b (b=0…t) 와 한 번에 병렬 비교해 [1/√2,√2] 로 환원한 뒤 HermesLog.
  우리 점수 u 는 1등도 2^-28 까지 내려가서, 절대값 경계 2^-28 을 가르는 sign (정밀도 ~2^-30) 이 필요 → 불가.
  대신
    단계마다  p_b = sign01(2^-b − x)       b = 1…W  (W 개 경계를 SIMD 로 sign 1회)
              P = Σ_b p_b,   x ← x·E(P),   e ← e + P,     E(P) ≈ 2^P (체비셰프)
    ★ x 와 e 를 같은 P 로 갱신 → sign 이 흐려 P 가 비정수여도 log 값이 정확히 유지
    마지막    log2 u ≈ ln-다항식(x) / ln2 − e,    x ∈ [≈0.5, ≈1.1]
  sign 은 실제 DP 계획 (build_composite_sb, sign_bootstrap = sin(πz/2) 모델) 으로 평가.

실행:  PYTHONPATH=/home/junhyung/study/Data-Analysis-with-CKKS \
       ~/.pyenv/versions/temp/bin/python exp_log_window.py
"""
from __future__ import annotations
import math
import numpy as np
from numpy.polynomial import chebyshev as C

from Hard_Tree.desilo import minimax_sign as MS

W = 8          # 병렬 경계 2^-1 … 2^-W
R = 3.0        # sign 입력 스케일 (x ≤ ~2.7 이 되도록 여유)


def cheb_fit(f, a, b, deg):
    x = np.cos(np.pi * (np.arange(4 * deg) + 0.5) / (4 * deg))
    return C.chebfit(x, f((x + 1) * (b - a) / 2 + a), deg), a, b


def cheb_ev(cab, x):
    c, a, b = cab
    return C.chebval((2 * x - a - b) / (b - a), c)


def best_sign_plan(delta):
    best = None
    for L in (8, 10, 12):
        try:
            sp = MS.build_composite_sb(delta, 2.0 ** -6, level_in=L, allow_refresh=False)
        except RuntimeError:
            continue
        if best is None or sp["cost_ms"] < best[1]["cost_ms"]:
            best = (L, sp)
    return best


def run(range_bits, sign_plan, E):
    L, sp = sign_plan
    nst = math.ceil(range_bits / W)
    u = np.concatenate([np.geomspace(2.0 ** -range_bits, 1.0, 300000),
                        np.random.default_rng(0).uniform(0, 1, 50000)])
    x = u + 2.0 ** -range_bits
    e = np.zeros_like(x)
    thr = 2.0 ** -np.arange(1, W + 1)
    for s in range(nst):
        z = (thr[None, :] - x[:, None]) / R
        assert np.abs(z).max() <= 1, "sign 입력이 [-1,1] 을 벗어남"
        p = 0.5 * (MS.eval_composite_plain_sb(z, sp) + 1)
        P = p.sum(1)
        x = x * cheb_ev(E, P)
        e = e + P
        print(f"    단계 {s}: x ∈ [{x.min():.3e}, {x.max():.3f}]   P ∈ [{P.min():.3f}, {P.max():.3f}]")
    lo, hi = x.min() * 0.98, x.max() * 1.02
    ref = np.log2(u + 2.0 ** -range_bits)
    for deg in (7, 11, 15):
        lnf = cheb_fit(np.log, lo, hi, deg)
        err = np.abs(cheb_ev(lnf, x) / math.log(2) - e - ref).max()
        print(f"  범위 {range_bits} bit: {nst} 단계 × sign(L{L} {MS.plan_str(sp)}, {sp['cost_ms']/1000:.2f}s)"
              f"  ln deg {deg:>2} on [{lo:.2f},{hi:.2f}] → log2 오차 {err:.2e}")


def main():
    delta = 2.0 ** -W / (2 * R)
    plan = best_sign_plan(delta)
    E = cheb_fit(lambda P: 2.0 ** P, -0.1, W + 0.1, 18)
    PP = np.linspace(-0.1, W + 0.1, 20001)
    erel = np.max(np.abs(cheb_ev(E, PP) / 2.0 ** PP - 1))
    print(f"sign δ=2^{math.log2(delta):.2f}  → L{plan[0]} {MS.plan_str(plan[1])} {plan[1]['cost_ms']/1000:.2f}s"
          f"   E(P)=2^P deg 18 상대오차 {erel:.1e}")
    for bits in (32, 16):
        run(bits, plan, E)


if __name__ == "__main__":
    main()

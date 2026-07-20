# label_noise_calibration.py
#
# gap_decode τ 상한식의 미지 상수 실측:
#   [A] 표준 bootstrap 노이즈 ε_boot(L): 메시지 크기 L 의존성
#       (지배항 후보 — {0,1} 도메인 실측 1.6e-3이 라벨 크기에서 얼마인가)
#   [B] fhe_max 단발 오차 vs |u−v|: sgn 오차·dead zone 실측
#       (dead zone 폭 L·δ, 관측 오차 ≤ |u−v|·ε_s/2 이론치와 대조)
#   [C] fhe_max 연쇄 m회 누적: 선형 누적 여부
#
# 사용: python label_noise_calibration.py   (GPU 환경, 프로젝트 루트에서)
# 출력: 표 3개 → E_slot ≈ Σ|Δ|·ε_s/2 + n_tie·Lδ/2 + n_refresh·ε_boot(L) + n_app·ε_eval
#       의 상수 결정 → τ 하한 = G_intra 상한 추정에 사용.

import numpy as np
from core.ciphertext_single.Client_main import setup_fhe_engine
from core.ciphertext_single.Label_Propagation import fhe_max, _refresh

DELTA = 2.0 ** -15


def _enc(engine, keypack, vec, slot_count):
    return engine.encrypt(list(vec) + [0.0] * (slot_count - len(vec)),
                          keypack.public_key)


def _dec(engine, ct, sk, n):
    return np.real(engine.decrypt(ct, sk))[:n]


def test_A_bootstrap(engine, keypack, sk, slot_count, n=256, reps=5):
    print("\n[A] 표준 bootstrap 노이즈 ε_boot(L) — 크기 의존성")
    print(f"{'L':>8} {'1회 max|err|':>14} {'1회 std':>12} "
          f"{'{reps}회 max|err|':>16} {'회당 기울기':>12}")
    rng = np.random.default_rng(0)
    for L in [1.0, 10.0, 100.0, 440.0, 847.0]:
        vec = rng.uniform(0.5, 1.0, n) * L          # 크기 L대 무작위 라벨
        ct = _enc(engine, keypack, vec, slot_count)
        errs = []
        cur = ct
        for r in range(reps):
            cur = _refresh(engine, cur, keypack)
            errs.append(np.abs(_dec(engine, cur, sk, n) - vec))
        e1, eR = errs[0], errs[-1]
        slope = (eR.max() - e1.max()) / max(reps - 1, 1)
        print(f"{L:>8.0f} {e1.max():>14.3e} {e1.std():>12.3e} "
              f"{eR.max():>16.3e} {slope:>12.3e}")
    print("  → ε_boot(L)이 L에 비례(상대오차형)인지, 상수(절대오차형)인지 판별.")
    print("    비례형이면 refresh가 지배항: E ≈ n_refresh × ε_rel × L")


def test_B_fhe_max_single(engine, keypack, sk, slot_count, N=400):
    L = 1.1 * N
    print(f"\n[B] fhe_max 단발 오차 vs |u−v|  (label_scale={L:.0f}, "
          f"dead zone 폭 L·δ={L*DELTA:.3e})")
    # 슬롯별로 다른 |u−v|를 한 번에: u=기준 300, v=300−diff
    diffs = np.array([0.0, L*DELTA*0.25, L*DELTA*0.5, L*DELTA, L*DELTA*2,
                      0.1, 0.5, 1.0, 5.0, 20.0, 100.0, 300.0])
    n = len(diffs)
    base = np.full(n, 300.0)
    u = _enc(engine, keypack, base, slot_count)
    v = _enc(engine, keypack, base - diffs, slot_count)
    out = fhe_max(engine, u, v, n, keypack, label_scale=L, secret_key=None)
    got = _dec(engine, out, sk, n)
    print(f"{'|u−v|':>12} {'err(max−u)':>14} {'이론상한 |Δ|ε_s/2 (ε_s=2^-15)':>30}")
    for d, g in zip(diffs, got):
        print(f"{d:>12.4e} {g-300.0:>+14.3e} {d*2**-15/2:>30.3e}")
    print("  → dead zone(|Δ|<Lδ) 안팎의 오차 거동과 ε_s 유효치 확인.")


def test_C_chain(engine, keypack, sk, slot_count, N=400, m=40):
    L = 1.1 * N
    print(f"\n[C] fhe_max 연쇄 {m}회 누적 (동점 u=v — 수렴 후 상태 모사)")
    n = 64
    vec = np.linspace(200.0, 314.0, n)
    ct = _enc(engine, keypack, vec, slot_count)
    cur = ct
    for i in range(m):
        cur = fhe_max(engine, cur, ct, n, keypack, label_scale=L,
                      secret_key=None)
    err = np.abs(_dec(engine, cur, sk, n) - vec)
    print(f"  {m}회 후 max|err|={err.max():.3e}, mean={err.mean():.3e} "
          f"→ 회당 ε_eval≈{err.max()/m:.3e}")


if __name__ == "__main__":
    engine, sk, keypack = setup_fhe_engine(verbose=False)
    sc = engine.slot_count
    test_A_bootstrap(engine, keypack, sk, sc)
    test_B_fhe_max_single(engine, keypack, sk, sc)
    test_C_chain(engine, keypack, sk, sc)
    print("\n결론 사용법: E_slot ≈ Σ_jump |Δ|·ε_s/2 + n_tie·Lδ/2 "
          "+ n_refresh·ε_boot(L) + n_app·ε_eval")
    print("τ 안전창: max_c G_intra < τ < ΔM_min − D_drift  "
          "(ΔM_min은 compute_lp_schedule의 fix_all에서 정확히 산출)")
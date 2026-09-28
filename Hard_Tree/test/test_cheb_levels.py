# soft_tree/test/test_cheb_levels.py
"""
minimax 에 실제로 쓸 차수(7, 13, 15, 27)의 레벨 소비 실측.

Lee et al. Table 1 (odd BSGS + 최적 레벨 소비):
    dep(7)=3  dep(13)=4  dep(15)=4  dep(27)=5
앞선 테스트에서 3차가 2 대신 3 을 먹었으므로, DesiloFHE 는 성분마다 +1 로 추정.
실측해서 깊이 예산을 확정한다.

실행:  python -m Hard_Tree.test.test_cheb_levels
"""
import numpy as np
from Hard_Tree.desilo.util.keypack import setup_fhe_engine

DEGREES = [3, 5, 7, 9, 13, 15, 27, 31]
PAPER_DEP = {3: 2, 5: 3, 7: 3, 9: 4, 11: 4, 13: 4, 15: 4, 27: 5, 29: 5, 31: 5}

engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
rlk, sc = kp.relinearization_key, engine.slot_count
rng = np.random.default_rng(0)

print(f"{'차수':>5}{'논문 dep':>10}{'실측':>7}{'차이':>7}   비고")
print("-" * 46)
for deg in DEGREES:
    coeffs = [0.0] * (deg + 1)
    for i in range(1, deg + 1, 2):                # 홀수차만 (sign 은 기함수)
        coeffs[i] = float(rng.normal(0, 0.3 / i))
    ct = engine.encrypt(list(rng.uniform(-0.8, 0.8, 16)) + [0.0] * (sc - 16),
                        kp.public_key)
    before = ct.level
    try:
        out = engine.evaluate_chebyshev_polynomial(engine.intt(ct), coeffs, rlk)
        used = before - out.level
        pd = PAPER_DEP.get(deg, "?")
        diff = (used - pd) if isinstance(pd, int) else "?"
        print(f"{deg:>5}{str(pd):>10}{used:>7}{str(diff):>7}")
    except Exception as e:
        print(f"{deg:>5}{str(PAPER_DEP.get(deg,'?')):>10}{'실패':>7}{'':>7}   {type(e).__name__}: {e}")

print()
print("깊이 예산 =  2 (pre-sign)  +  Σ dep(성분)  +  3 (post-sign)")
print("  fresh 21 레벨 안에 들어가면 부트스트랩 0회.")
print("  차이가 일관되게 +1 이면 BSGS 직접 구현으로 성분 수만큼 회수 가능.")
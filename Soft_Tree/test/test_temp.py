#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_cheb_chain.py — minimax composite 구현 가능 여부 확인 (한 파일)

확인할 것:  evaluate_chebyshev_polynomial 의 출력을
            다시 같은 함수의 입력으로 넣을 수 있는가?
            (minimax composite 는 p_k ∘ ... ∘ p_1 이므로 필수)

실행:  python test_cheb_chain.py
"""
import numpy as np
from numpy.polynomial.chebyshev import chebval
from Soft_Tree.desilo.util.keypack import setup_fhe_engine

C1 = [0.0, 0.9, 0.0, -0.1]        # 아무 홀수차 다항식 2개
C2 = [0.0, 1.1, 0.0, -0.2]

engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
rlk, sc = kp.relinearization_key, engine.slot_count

x = np.linspace(-0.8, 0.8, 16)
ct = engine.encrypt(list(x) + [0.0] * (sc - 16), kp.public_key)

# ── 1단계 ──
z = engine.intt(ct)
y1 = engine.evaluate_chebyshev_polynomial(z, C1, rlk)
d1 = np.real(engine.decrypt(y1, sk))[:16]
e1 = np.abs(d1 - chebval(x, C1)).max()
print(f"[1단계] max err = {e1:.3e}   level = {getattr(y1, 'level', '?')}")

# ── 2단계: 출력을 그대로 재입력 ──
for tag, inp in (("그대로", y1), ("intt 후", None)):
    try:
        z2 = inp if inp is not None else engine.intt(y1)
        y2 = engine.evaluate_chebyshev_polynomial(z2, C2, rlk)
        d2 = np.real(engine.decrypt(y2, sk))[:16]
        e2 = np.abs(d2 - chebval(chebval(x, C1), C2)).max()
        ok = "OK" if e2 < 1e-3 else "값 불일치"
        print(f"[2단계 {tag}] max err = {e2:.3e}  → {ok}   "
              f"level = {getattr(y2, 'level', '?')}")
    except Exception as ex:
        print(f"[2단계 {tag}] 예외: {type(ex).__name__}: {ex}")

print("\n판정:  둘 중 하나라도 OK  →  minimax composite 구현 가능")
print("       둘 다 실패        →  baby-step giant-step 직접 구현 필요")
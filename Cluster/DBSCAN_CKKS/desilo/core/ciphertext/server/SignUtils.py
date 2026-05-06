# core/ciphertext/server/SignUtils.py
"""
Lifting + sign_bootstrap 기반 sign 함수 모음.

[FIX] sign_bootstrap 은 coefficient form 암호문을 요구함.
      multiply/subtract 등 연산 후 NTT form으로 남은 암호문을
      sign_bootstrap 직전 engine.intt() 로 변환.
"""

import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack


_BOOTSTRAP_INTERVAL = 3


def _poly_step(engine: Engine, ct: Ciphertext, kp: KeyPack) -> Ciphertext:
    """
    1.5x - 0.5x³ 한 번 적용 (2 레벨 소모).
    입력이 [-1,1] 안에 있을 때 ±1 방향으로 수렴.
    """
    relin = kp.relinearization_key
    x_sq  = engine.square(ct, relin)
    x_cub = engine.multiply(x_sq, ct, relin)
    t1    = engine.multiply(ct,    1.5)
    t2    = engine.multiply(x_cub, 0.5)
    return engine.subtract(t1, t2)


def compute_lifting_depth(min_abs_input: float, tau: float = 0.1) -> int:
    """
    lifting 반복 횟수 계산.
    1.5^n * min_abs_input >= (1 - tau) 조건.
    """
    if min_abs_input <= 0:
        raise ValueError("min_abs_input must be > 0")
    target = 1.0 - tau
    depth  = math.ceil(math.log(target / min_abs_input) / math.log(1.5))
    return max(depth + 2, 4)


def refresh_via_sign(
    engine: Engine,
    ct:     Ciphertext,
    kp:     KeyPack,
    scale:  float,
) -> Ciphertext:
    """
    sign_bootstrap 을 이용한 레벨 복원 대체.
    """
    # ① 정규화: [-1, 1]
    scale_inv_pt = engine.encode([1.0 / scale] * engine.slot_count)
    normalized   = engine.multiply(ct, scale_inv_pt)

    # ② [FIX] NTT form → coefficient form 변환 후 sign_bootstrap
    normalized = engine.intt(normalized)
    snapped = engine.sign_bootstrap(
        normalized,
        kp.relinearization_key,
        kp.conjugation_key,
        kp.sign_bootstrap_key,
    )

    # ③ 복원: ×scale → 원래 크기
    scale_pt = engine.encode([scale] * engine.slot_count)
    return engine.multiply(snapped, scale_pt)


def lifting_to_pm1(
    engine: Engine,
    ct:     Ciphertext,
    kp:     KeyPack,
    depth:  int,
) -> Ciphertext:
    """
    Step 1 – Lifting: [-1,1] → [-1,-1+τ] ∪ [1-τ,1]
    """
    current = ct
    for i in range(depth):
        current = _poly_step(engine, current, kp)
        if (i + 1) % _BOOTSTRAP_INTERVAL == 0 and (i + 1) < depth:
            # [FIX] NTT form → coefficient form 변환 후 sign_bootstrap
            current = engine.intt(current)
            current = engine.sign_bootstrap(
                current,
                kp.relinearization_key,
                kp.conjugation_key,
                kp.sign_bootstrap_key,
            )
    return current


def fhe_sign_lifted(
    engine: Engine,
    ct:     Ciphertext,
    kp:     KeyPack,
    depth:  int,
) -> Ciphertext:
    """
    Step 1 + Step 2: lifting → sign_bootstrap → {-1, +1}
    """
    lifted = lifting_to_pm1(engine, ct, kp, depth)
    # [FIX] NTT form → coefficient form 변환 후 sign_bootstrap
    lifted = engine.intt(lifted)
    return engine.sign_bootstrap(
        lifted,
        kp.relinearization_key,
        kp.conjugation_key,
        kp.sign_bootstrap_key,
    )


def fhe_heaviside_lifted(
    engine: Engine,
    ct:     Ciphertext,
    kp:     KeyPack,
    depth:  int,
) -> Ciphertext:
    """
    Heaviside 함수: ≈ 1 if ct > 0, else ≈ 0.
    = (1 + sign(ct)) / 2
    """
    sign_ct = fhe_sign_lifted(engine, ct, kp, depth)
    return engine.multiply(engine.add(sign_ct, 1.0), 0.5)
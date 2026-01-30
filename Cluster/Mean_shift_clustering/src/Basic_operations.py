import desilofhe

def Plain_Inv(engine, x, m, d, relinearization_key,conjugation_key, bootstrap_key):
    """
    m(input 범위)가 plaintext인 버전

    x: 입력값(ciphertext) / 0 < x < m,
    d: 반복횟수(plaintext) / d \in N
    m: 스케일링 팩터(plaintext) / input \in (0,m)일 경우, 2/m을 곱해주어 input \in (0,2)로 맞춰줌(Goldschmidt 방식)
    """
    x_temp = engine.multiply(x, 2/m)  # x_temp = (2/m) * x
    a = engine.subtract(2, x_temp)  # a = 2 - (2/m) * x
    b = engine.subtract(1, x_temp)  # b = 1 - (2/m) * x
    for _ in range(d):
        b = engine.multiply(b, b, relinearization_key)  # b = b^2
        a = engine.multiply(a, engine.add(1, b), relinearization_key)  # a = a * (1 + b)
        if a.level < 1 or b.level < 1:
            a = engine.bootstrap(a,relinearization_key, conjugation_key, bootstrap_key)
            b = engine.bootstrap(b,relinearization_key, conjugation_key, bootstrap_key)
            print("Bootstrapping performed")
    return a

# def MixIdx(engine, enc_values, n, d_a, d_b, t, m, relinearization_key, rotation_key, conjugation_key=None, bootstrap_key=None):
    
#     # 1) Initial 1-norm normalization: b_j = a_j / sum(a)
#     sum_a = rotate_sum(engine, enc_values, n, rotation_key)                     # Σ a_j
#     inv_sum = Basic_Inv(engine, sum_a, d_a, relinearization_key)                # ≈ 1 / Σ a_j
#     b = engine.multiply(enc_values, inv_sum, relinearization_key)               # b ← a * inv(Σ a)

#     # 2) Iterations: b ← (b^m) / (Σ b^m)
#     for _ in range(t):
#         # Compute b^m
#         b_m = b

#         # Optional: fixed-position bootstrap to reduce run-to-run variability
#         if b.level < (m + d_b + 2) or b_m.level < (m + d_b + 2):
#             b = engine.bootstrap(b, relinearization_key, conjugation_key, bootstrap_key)
#             b_m = engine.bootstrap(b_m, relinearization_key, conjugation_key, bootstrap_key)
#             print("Bootstrap performed")

#         for _ in range(m - 1):
#             b_m = engine.multiply(b_m, b, relinearization_key)

#         # Sum s = Σ b^m and invert
#         s = rotate_sum(engine, b_m, n, rotation_key)
#         inv_s = Basic_Inv(engine, s, d_b, relinearization_key)

#         # Re-normalize: b ← b^m * inv_s
#         b = engine.multiply(b_m, inv_s, relinearization_key)

#     # Output: b is approximately one-hot on the max index
#     return b

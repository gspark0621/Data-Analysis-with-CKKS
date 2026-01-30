def Basic_Inv(engine, x,d, relineralization_key):  # 0 < x < 2
    #Goldschmidt 역수 근사
    #TODO: newton rapshon 방법으로 수정
    a = engine.subtract(2, x)  # a = 2 - x
    b = engine.subtract(1, x)  # b = 1 - x
    for _ in range(d):
        b = engine.multiply(b, b, relineralization_key)  # b = b^2
        a = engine.multiply(a, engine.add(1, b), relineralization_key)  # a = a * (1 + b)

    return a

def Basic_sqrt(engine, x, d, relinearization_key, conjugation_key, bootstrap_key):  # 0 <= x <= 1
    a = x
    b = engine.subtract(x, 1)  # b = x - 1
    
    for _ in range(d):
        # a = a * (1 - b / 2)
        factor = engine.subtract(1, engine.multiply(0.5, b))
        a = engine.multiply(a, factor, relinearization_key)

        # b = b^2 * (b - 3)/4
        b_squared = engine.square(b, relinearization_key)
        b_minus_3 = engine.subtract(b, 3)
        numerator = engine.multiply(b_squared, b_minus_3, relinearization_key)
        b = engine.multiply(numerator, 0.25)
        if factor.level < 5 or a.level < 5 or b.level < 5:
            a = engine.bootstrap(a,relinearization_key, conjugation_key, bootstrap_key)
            b = engine.bootstrap(b,relinearization_key, conjugation_key, bootstrap_key)
            print("Bootstrapping performed")
    return a

def CKKS_min(engine, a, b, d, relinearization_key, conjugation_key, bootstrap_key): # 0 <= x < 1
    x = engine.multiply(engine.add(a,b),0.5) # x= (a+b)/2
    y = engine.multiply(engine.subtract(a,b),0.5) # y= (a-b)/2
    y_2 = engine.square(y, relinearization_key)
    z= Basic_sqrt(engine, y_2, d, relinearization_key, conjugation_key, bootstrap_key)
    return engine.subtract(x,z)

def CKKS_max(engine, a, b, d, relinearization_key, conjugation_key, bootstrap_key): # 0 <= x < 1
    x = engine.multiply(engine.add(a,b),0.5) # x= (a+b)/2
    y = engine.multiply(engine.subtract(a,b),0.5) # y= (a-b)/2
    y_2 = engine.square(y, relinearization_key)
    z= Basic_sqrt(engine, y_2, d, relinearization_key, conjugation_key, bootstrap_key)
    return engine.add(x,z)

# TODO: rotate_sum함수 수정 필요
# def rotate_sum(engine, enc_vec, num_slots, rotation_key):
#     result = engine.clone(enc_vec)
#     rotated = enc_vec
#     for _ in range(num_slots):
#         rotated = engine.rotate(rotated, rotation_key, -1)
#         result = engine.add(result, rotated)
#     return result

def MaxIdx(engine, enc_values, n, d_a, d_b, t, m, relinearization_key, rotation_key, conjugation_key=None, bootstrap_key=None):
    
    # 1) Initial 1-norm normalization: b_j = a_j / sum(a)
    sum_a = rotate_sum(engine, enc_values, n, rotation_key)                     # Σ a_j
    inv_sum = Basic_Inv(engine, sum_a, d_a, relinearization_key)                # ≈ 1 / Σ a_j
    b = engine.multiply(enc_values, inv_sum, relinearization_key)               # b ← a * inv(Σ a)

    # 2) Iterations: b ← (b^m) / (Σ b^m)
    for _ in range(t):
        # Compute b^m
        b_m = b

        # Optional: fixed-position bootstrap to reduce run-to-run variability
        if b.level < (m + d_b + 2) or b_m.level < (m + d_b + 2):
            b = engine.bootstrap(b, relinearization_key, conjugation_key, bootstrap_key)
            b_m = engine.bootstrap(b_m, relinearization_key, conjugation_key, bootstrap_key)
            print("Bootstrap performed")

        for _ in range(m - 1):
            b_m = engine.multiply(b_m, b, relinearization_key)

        # Sum s = Σ b^m and invert
        s = rotate_sum(engine, b_m, n, rotation_key)
        inv_s = Basic_Inv(engine, s, d_b, relinearization_key)

        # Re-normalize: b ← b^m * inv_s
        b = engine.multiply(b_m, inv_s, relinearization_key)

    # Output: b is approximately one-hot on the max index
    return b

def CKKS_comp(engine, a, b, d_a, d_b, t, m, relinearization_key, conjugation_key, bootstrap_key): # 1/2 <= x < 3/2
    # 입력 스케일: 1/2 <= a,b < 3/2 가정
    a_2  = engine.multiply(a, 0.5)
    a_b  = engine.add(a, b)
    ab_2 = engine.multiply(a_b, 0.5)

    # Step 1–2: 1-노름 정규화 a <- a/(a+b), b <- 1-a
    inv0 = Basic_Inv(engine, ab_2, d_a, relinearization_key)
    a    = engine.multiply(a_2, inv0, relinearization_key)
    b    = engine.subtract(1, a)

    # Step 3: (a_m, b_m) 거듭제곱 강조 후 합으로 나눈 정규화, t회 반복
    for _ in range(t):
        a_m = a
        b_m = b

        # 레벨 부족 시 부트스트래핑으로 복구 (선택적)
        if (a.level   < (d_b + 3) or b.level   < (d_b + 3) or
            a_m.level < (d_b + 3) or b_m.level < (d_b + 3)):
            a   = engine.bootstrap(a,   relinearization_key, conjugation_key, bootstrap_key)
            b   = engine.bootstrap(b,   relinearization_key, conjugation_key, bootstrap_key)
            a_m = engine.bootstrap(a_m, relinearization_key, conjugation_key, bootstrap_key)
            b_m = engine.bootstrap(b_m, relinearization_key, conjugation_key, bootstrap_key)

        # a_m = a^m, b_m = b^m
        for _ in range(m - 1):
            a_m = engine.multiply(a_m, a, relinearization_key)
            b_m = engine.multiply(b_m, b, relinearization_key)

        # 필요시 재부트스트랩
        if (a.level   < (d_b + 3) or b.level   < (d_b + 3) or
            a_m.level < (d_b + 3) or b_m.level < (d_b + 3)):
            a   = engine.bootstrap(a,   relinearization_key, conjugation_key, bootstrap_key)
            b   = engine.bootstrap(b,   relinearization_key, conjugation_key, bootstrap_key)
            a_m = engine.bootstrap(a_m, relinearization_key, conjugation_key, bootstrap_key)
            b_m = engine.bootstrap(b_m, relinearization_key, conjugation_key, bootstrap_key)

        # 정규화: a <- a_m / (a_m + b_m), b <- 1 - a
        denom_inv = Basic_Inv(engine, engine.add(a_m, b_m), d_b, relinearization_key)
        a = engine.multiply(a_m, denom_inv, relinearization_key)
        b = engine.subtract(1, a)

    # 반환: 비교 지표(0~1), 1에 가까우면 a>b
    return a


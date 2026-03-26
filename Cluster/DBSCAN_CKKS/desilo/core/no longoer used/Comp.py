from desilofhe import Engine, Ciphertext, RelinearizationKey


def inv_goldschmidts(engine, x,m,d, relinearization_key): #x: 입력값, m: 0<x<m - plaintext, d: 반복횟수
    # Goldschmidts inverse 함수(scaling X)
    a = engine.subtract(2, engine.multiply(x, (2/m)))  # a = 2 - (2/m)*x
    b = engine.subtract(1, engine.multiply(x, (2/m)))  # b = 1 - (2/m)*x
    for _ in range(d):
        b = engine.multiply(b, b, relinearization_key)  # b = b^2
        a = engine.multiply(a, engine.add(1, b), relinearization_key)  # a = a * (1 + b)
    return engine.multiply(a, (2/m))

def inv_goldschmidts_scaled(engine, x,d, relinearization_key):  # 0 < x < 2, d: 반복횟수
    # Goldschmidts inverse 함수(scaling 이후)
    a = engine.subtract(2, x)  # a = 2 - x
    b = engine.subtract(1, x)  # b = 1 - x
    for _ in range(d):
        b = engine.multiply(b, b, relinearization_key)  # b = b^2
        a = engine.multiply(a, engine.add(1, b), relinearization_key)  # a = a * (1 + b)

    return a

def Comp(engine, a, b, d_a, d_b, t, m, relinearization_key, conjugation_key, bootstrap_key): # 1/2 <= x < 3/2
    # 입력 스케일: 1/2 <= a,b < 3/2 가정
    a_2  = engine.multiply(a, 0.5)
    a_b  = engine.add(a, b)
    ab_2 = engine.multiply(a_b, 0.5)

    # Step 1–2: 1-노름 정규화 a <- a/(a+b), b <- 1-a
    inv0 = inv_goldschmidts_scaled(engine, ab_2, d_a, relinearization_key)
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
        denom_inv = inv_goldschmidts_scaled(engine, engine.add(a_m, b_m), d_b, relinearization_key)
        a = engine.multiply(a_m, denom_inv, relinearization_key)
        b = engine.subtract(1, a)

    # 반환: 비교 지표(0~1), 1에 가까우면 a>b
    return a 
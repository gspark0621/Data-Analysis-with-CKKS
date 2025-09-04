def Basic_Inv(engine, x,d, relineralization_key):  # 0 < x < 2
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
        if factor.level < 3 or a.level < 3 or b.level < 3:
            a = engine.bootstrap(a,relinearization_key, conjugation_key, bootstrap_key)
            b = engine.bootstrap(b,relinearization_key, conjugation_key, bootstrap_key)
            print("Bootstrapping performed")
    return a

def min(engine, a, b, d, relinearization_key, conjugation_key, bootstrap_key): # 0 <= x < 1
    x = engine.multiply(engine.add(a,b),0.5) # x= (a+b)/2
    y = engine.multiply(engine.subtract(a,b),0.5) # y= (a-b)/2
    y_2 = engine.square(y, relinearization_key)
    z= Basic_sqrt(engine, y_2, d, relinearization_key, conjugation_key, bootstrap_key)
    return engine.subtract(x,z)

def max(engine, a, b, d, relinearization_key, conjugation_key, bootstrap_key): # 0 <= x < 1
    x = engine.multiply(engine.add(a,b),0.5) # x= (a+b)/2
    y = engine.multiply(engine.subtract(a,b),0.5) # y= (a-b)/2
    y_2 = engine.square(y, relinearization_key)
    z= Basic_sqrt(engine, y_2, d, relinearization_key, conjugation_key, bootstrap_key)
    return engine.add(x,z)

def MaxIdx(a_list, d_a,d_b, m, t):
    # Step 1. Initial normalization
    total = sum(a_list)
    inv_total = Basic_Inv(total, d_a)
    b = [ai * inv_total for ai in a_list]  # b_j = a_j / sum(a_list)
    b_n= 1 - sum(b)
    b[-1] = b_n  # 마지막 원소는 나머지 부분(1 - sum(b))
    # Step 2. Iterative refinement
    for _ in range(t):
        total_bm = sum([bj ** m for bj in b])
        inv_total_bm = Basic_Inv(total_bm, d_b)
        b = [(bj ** m) * inv_total_bm for bj in b]
        b_n= 1 - sum(b)
        b[-1]=b_n  # 마지막 원소는 나머지 부분(1 - sum(b))  
    return b  # b[i] ≈ 1 (최댓값 위치), ≈ 0 (그외)


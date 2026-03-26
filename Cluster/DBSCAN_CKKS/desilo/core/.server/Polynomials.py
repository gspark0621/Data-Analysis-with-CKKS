# core/Polynomial.py
import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack

def fhe_sign_unit(engine: Engine, x_ct: Ciphertext, num_grids: int, keypack: KeyPack, depth: int = 4) -> Ciphertext:
    """사용자가 작성한 Sign 근사 다항식 (1.5x - 0.5x^3)"""
    relin_key = keypack.relinearization_key
    c15_pt = engine.encode([1.5 for _ in range(num_grids)])
    c05_pt = engine.encode([0.5 for _ in range(num_grids)])

    current_x = x_ct
    for _ in range(depth):
        x_sq = engine.square(current_x, relin_key)
        x_cub = engine.multiply(x_sq, current_x, relin_key)
        term1 = engine.multiply(current_x, c15_pt)
        term2 = engine.multiply(x_cub, c05_pt)
        current_x = engine.subtract(term1, term2)
        
    return current_x

def fhe_fast_max_unit(engine: Engine, A_ct: Ciphertext, B_ct: Ciphertext, num_grids: int, keypack: KeyPack) -> Ciphertext:
    """사용자가 작성한 암호문 간의 Max(A, B) 반환 함수 (ReLU 기반)"""
    relin_key = keypack.relinearization_key
    half_pt = engine.encode([0.5 for _ in range(num_grids)])

    diff_ct = engine.subtract(B_ct, A_ct)
    
    # 안정성을 위해 N 대신 num_grids로 스케일링
    scale_factor = 1.0 / float(num_grids)
    scale_pt = engine.encode([scale_factor for _ in range(num_grids)])
    scaled_diff_ct = engine.multiply(diff_ct, scale_pt)
    
    sign_ct = fhe_sign_unit(engine, scaled_diff_ct, num_grids, keypack, depth=4)

    diff_sign_ct = engine.multiply(diff_ct, sign_ct, relin_key)
    relu_ct = engine.add(diff_ct, diff_sign_ct)
    relu_ct = engine.multiply(relu_ct, half_pt)

    return engine.add(A_ct, relu_ct)

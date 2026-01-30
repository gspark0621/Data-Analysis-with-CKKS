# euclidean_ct.py
from desilofhe import engine
from EncryptModule import CKKSEncryptor


def encrypted_euclidean1(p_enc, q_enc, engine, rotation_key, relinearization_key, num_of_slots):
    """
    여러 attribute(num_of_slots개)를 가진 2개의 점들간 유클리디안 거리 계산
    
    수정해야할 점
    1. 굳이 num_of_slots를 지정해줘야 하나?
    """
    diff = engine.subtract(p_enc, q_enc)
    diff_squared = engine.square(diff, relinearization_key)
    squared_sum_enc = sum_encrypted_vector1(diff_squared, engine, rotation_key, num_of_slots)
    return squared_sum_enc

def sum_encrypted_vector1(enc_vector, engine, rotation_key, num_of_slots):
    result = enc_vector
    step = 1
    while step < num_of_slots:
        rotated = engine.rotate(result, rotation_key, step)
        result = engine.add(result, rotated)
        step *= 2
    return result


def encrypted_euclidean_2d(engine, p_enc, q_enc, rotation_key, relinearization_key):
    """2차원 점들 간 거리 계산 시 사용
    Ex. p_enc = (x1,y1,x2,y2,x3,y3....)
    q_enc = (x4,y4,x5,y5,x6,y6....)"""
    # 암호문 차이 계산
    diff = engine.subtract(p_enc, q_enc)
    
    # 각 슬롯별 제곱 (x_diff^2, y_diff^2, ...)
    diff_squared = engine.square(diff, relinearization_key)
    
    # rotation 1 슬롯 (y_diff^2, x_next_diff^2, ...)
    rotated = engine.rotate(diff_squared, rotation_key, 1)
    
    # 두 벡터를 더해 각 점별 거리 제곱 계산 (x_diff^2 + y_diff^2)
    distance_squared_enc = engine.add(diff_squared, rotated)
    
    mask = [1 if i % 2 == 0 else 0 for i in range(engine.slot_count)]
    dist_masked = engine.multiply(distance_squared_enc, mask)

    return dist_masked
# euclidean_ct.py
from desilofhe import Engine
from EncryptModule import CKKSEncryptor
def encrypted_euclidean(p_enc, q_enc, engine, rotation_key, relinearization_key, num_of_slots):
    """
    p_enc와 q_enc는 암호화된 벡터
    제곱된 거리를 반환
    
    수정해야할 점
    1. 굳이 num_of_slots를 지정해줘야 하나?
    """
    diff = engine.subtract(p_enc, q_enc)
    diff_squared = engine.square(diff, relinearization_key)
    squared_sum_enc = sum_encrypted_vector(diff_squared, engine, rotation_key, num_of_slots)
    return squared_sum_enc

def sum_encrypted_vector(enc_vector, engine, rotation_key, num_of_slots):
    result = enc_vector
    step = 1
    while step < num_of_slots:
        rotated = engine.rotate(result, rotation_key, step)
        result = engine.add(result, rotated)
        step *= 2
    return result

# 테스트용 메인 함수
from time import time
def main():
    start = time()
    encryptor = CKKSEncryptor()

    engine = encryptor.get_engine()
    secret_key = encryptor.get_secret_key()
    public_key = encryptor.get_public_key()
    relinearization_key = encryptor.get_relinearization_key()
    rotation_key = encryptor.get_rotation_key()
    num_of_slots = encryptor.slot_count

    data1 = [1, 2, 3]
    data2 = [4, 5, 6]
    encrypted_data1 = engine.encrypt(data1, public_key)
    encrypted_data2 = engine.encrypt(data2, public_key)

    result = encrypted_euclidean(
        encrypted_data1, encrypted_data2,
        engine, rotation_key, relinearization_key, num_of_slots
    )
    end = time()
    print(f"Final result: {engine.decrypt(result, secret_key)[0]}\nTime taken: {(end - start):.3f} seconds, Time per slot: {(end - start)/len(data1):.3f} seconds")


if __name__ == "__main__":
    main()
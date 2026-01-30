import numpy as np
import pytest
from desilofhe import Engine
from src.Basic_operations import Plain_Inv

@pytest.fixture(scope="module")
def engine():
    return Engine(use_bootstrap=True, mode="gpu")

@pytest.fixture(scope="module")
def secret_key(engine):
    return engine.create_secret_key()

@pytest.fixture(scope="module")
def keys(engine, secret_key):
    public_key = engine.create_public_key(secret_key)
    relin_key = engine.create_relinearization_key(secret_key)
    conj_key = engine.create_conjugation_key(secret_key)
    boot_key = engine.create_bootstrap_key(secret_key)
    
    return public_key, relin_key, conj_key, boot_key

# ==========================================
# Test Case
# ==========================================

@pytest.mark.parametrize(
    "m, d, data_size",
    [
        (4.0, 8, 1024),   # m=4, 반복 8회
        (10.0, 10, 1024), # m=10, 반복 10회 (정밀도 향상)
        (2.0, 8, 1024),   # m=2, 반복 8회
    ],
    scope="function",
)


def test_Plain_Inv(engine, secret_key, keys, m, d, data_size):
    public_key, relin_key, conj_key, boot_key = keys

    # Goldschmidt 알고리즘의 수렴 조건은 scaled value가 (0, 2) 사이여야 함.
    # 따라서 원본 데이터 x는 (0, m) 사이여야 함.
    # 0에 너무 가까우면 역수가 폭발하므로 0.1 ~ m*0.9 범위로 안전하게 생성
    x_input = np.random.uniform(0.1, m * 0.9, size=data_size)

    # 2. Numpy Ground Truth
    x_temp_np = x_input * (2 / m)
    numpy_output = 1 / x_temp_np

    # 3. HE Encryption
    ciphertext = engine.encrypt(x_input, public_key)
    result_cipher = Plain_Inv(
        engine, 
        ciphertext, 
        m, 
        d, 
        relinearization_key=relin_key, 
        conjugation_key=conj_key, 
        bootstrap_key=boot_key
    )

    he_output = engine.decrypt(result_cipher, secret_key)
    he_output = he_output[:data_size]

    np.allclose(numpy_output, he_output, rtol=1e-2, atol=1e-1)

import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack
from dataclasses import dataclass

# 실제 모듈 경로에 맞게 수정해주세요
# 예: from core.comp_functions import inv_goldschmidts, inv_goldschmidts_scaled, Comp, sign_newton_raphson_01
from core.Comp import inv_goldschmidts, inv_goldschmidts_scaled, Comp, sign_newton_raphson_01

# ==========================================
# Test Configuration Helper
# ==========================================
@dataclass
class ComparisonConfig:
    name: str
    n_samples: int
    d: int         # Goldschmidt 반복 횟수 (or Newton depth)
    m: int = 10    # inv_goldschmidts의 max range 혹은 Comp의 power degree
    t: int = 2     # Comp의 정규화 반복 횟수
    slot_counts: int = 1 << 15

# ==========================================
# Fixtures
# ==========================================
@pytest.fixture(scope="module")
def engine():
    # 부트스트래핑과 GPU 모드 사용 (Comp 함수에 필수)
    return Engine(use_bootstrap=True, mode="gpu") 

@pytest.fixture(scope="module")
def secret_key(engine):
    return engine.create_secret_key()

@pytest.fixture(scope="module")
def keypack(engine, secret_key):
    # Comp 함수 실행을 위해 모든 종류의 Key 생성
    return KeyPack(
        engine.create_public_key(secret_key),
        engine.create_rotation_key(secret_key),
        engine.create_relinearization_key(secret_key),
        engine.create_conjugation_key(secret_key), 
        engine.create_bootstrap_key(secret_key),  
    )

# ==========================================
# Test Logic
# ==========================================

# --- A. Inverse Function Tests (Goldschmidt) ---
@pytest.mark.parametrize(
    "config",
    [
        ComparisonConfig("Inv_LowIter", n_samples=100, d=4, m=10, t=0),
        ComparisonConfig("Inv_HighIter", n_samples=100, d=8, m=10, t=0),
    ],
    ids=lambda c: f"{c.name}_d={c.d}"
)
def test_inv_goldschmidts(engine, keypack, secret_key, config):
    """
    inv_goldschmidts 함수 검증: 1/x 근사 확인
    입력 범위: 0 < x < m
    """
    pk = keypack.public_key
    rlk = keypack.relinearization_key
    
    # 1. Data Generation (0.5 ~ m-0.1)
    rng = np.random.default_rng(42)
    effective_n = min(config.n_samples, config.slot_counts)
    
    x_data = rng.uniform(0.5, config.m - 0.1, effective_n)
    
    # Padding
    x_pad = np.pad(x_data, (0, config.slot_counts - effective_n), mode='constant', constant_values=1.0)

    # 2. Encrypt
    x_enc = engine.encrypt(x_pad, pk)
    
    # 3. Execution
    res_enc = inv_goldschmidts(engine, x_enc, config.m, config.d, rlk)
    
    # 4. Verify
    res_dec = engine.decrypt(res_enc, secret_key)
    he_out = np.array(res_dec[:effective_n])
    expected = 1.0 / x_data
    
    # Tolerance 설정 (반복 횟수가 적으면 오차가 클 수 있음)
    tolerance = 1e-2 if config.d >= 8 else 0.2
    
    print(f"\n[Check Inv] First 3 results: HE={he_out[:3]}, Real={expected[:3]}")
    assert np.allclose(he_out, expected, rtol=tolerance, atol=tolerance), \
        f"Failed inv_goldschmidts on {config.name}"


@pytest.mark.parametrize("d_iter", [4, 8])
def test_inv_goldschmidts_scaled(engine, keypack, secret_key, d_iter):
    """
    inv_goldschmidts_scaled 함수 검증: 1/x 근사 (0.2 < x < 1.8)
    """
    pk = keypack.public_key
    rlk = keypack.relinearization_key
    n_samples = 100
    slot_counts = 1 << 15
    
    rng = np.random.default_rng(42)
    x_data = rng.uniform(0.2, 1.8, n_samples)
    x_pad = np.pad(x_data, (0, slot_counts - n_samples), mode='constant', constant_values=1.0)
    
    x_enc = engine.encrypt(x_pad, pk)
    res_enc = inv_goldschmidts_scaled(engine, x_enc, d_iter, rlk)
    
    res_dec = engine.decrypt(res_enc, secret_key)
    he_out = np.array(res_dec[:n_samples])
    expected = 1.0 / x_data
    
    tolerance = 1e-2 if d_iter >= 8 else 0.2
    assert np.allclose(he_out, expected, rtol=tolerance, atol=tolerance)


# --- B. Comparison Function Tests (Goldschmidt) ---
@pytest.mark.parametrize(
    "config",
    [
        ComparisonConfig("Comp_Standard", n_samples=50, d=4, m=2, t=2), 
    ],
    ids=lambda c: c.name
)
def test_Comp_iterative(engine, keypack, secret_key, config):
    """
    Comp 함수 검증: 반복적인 정규화를 통한 대소 비교
    입력: 1/2 <= a, b < 3/2
    출력: a > b 이면 1에 수렴, a < b 이면 0에 수렴
    """
    pk = keypack.public_key
    rlk = keypack.relinearization_key
    cjk = keypack.conjugation_key
    btk = keypack.bootstrap_key
    
    n = config.n_samples
    half = n // 2
    
    # 1. Data Gen: [a > b 구간] + [a < b 구간]
    # a > b: (0.9 vs 0.6)
    # a < b: (0.6 vs 0.9)
    a_data = np.concatenate([np.full(half, 0.9), np.full(n-half, 0.6)])
    b_data = np.concatenate([np.full(half, 0.6), np.full(n-half, 0.9)])
    
    a_pad = np.pad(a_data, (0, config.slot_counts - n), constant_values=1.0)
    b_pad = np.pad(b_data, (0, config.slot_counts - n), constant_values=1.0)
    
    a_enc = engine.encrypt(a_pad, pk)
    b_enc = engine.encrypt(b_pad, pk)
    
    # Execution
    res_enc = Comp(
        engine, a_enc, b_enc, 
        config.d, config.d, 
        config.t, config.m, 
        rlk, cjk, btk
    )
    
    res_dec = engine.decrypt(res_enc, secret_key)
    he_out = np.array(res_dec[:n])
    
    print(f"\n[Comp Goldschmidt] Res (First 3 >): {he_out[:3]}, (First 3 <): {he_out[half:half+3]}")
    
    # Check Case 1: a > b (Result -> 1)
    # 근사 연산이므로 0.5보다 큰지(혹은 0.8 이상인지) 확인
    assert np.all(he_out[:half] > 0.6), "Failed Case a > b (Goldschmidt)"
    
    # Check Case 2: a < b (Result -> 0)
    assert np.all(he_out[half:] < 0.4), "Failed Case a < b (Goldschmidt)"


# --- C. Comparison Function Tests (Newton-Raphson) ---
@pytest.mark.parametrize(
    "depth", [5, 10]
)
def test_sign_newton_raphson(engine, keypack, secret_key, depth):
    """
    sign_newton_raphson_01 함수 검증: 0/1 부호 판별
    입력: -1 < x < 1 (정규화된 차이값)
    출력: x > 0 -> 1, x < 0 -> 0
    """
    pk = keypack.public_key
    rlk = keypack.relinearization_key
    
    n_samples = 50
    slot_counts = 1 << 15
    half = n_samples // 2
    
    # 1. Data Gen
    # 양수(이웃): 0.2 ~ 0.8
    # 음수(이웃X): -0.8 ~ -0.2
    pos_data = np.random.uniform(0.2, 0.8, half)
    neg_data = np.random.uniform(-0.8, -0.2, n_samples - half)
    
    x_data = np.concatenate([pos_data, neg_data])
    x_pad = np.pad(x_data, (0, slot_counts - n_samples), constant_values=0.5)
    
    x_enc = engine.encrypt(x_pad, pk)
    
    # 2. Execution
    res_enc = sign_newton_raphson_01(engine, x_enc, depth, rlk)
    
    # 3. Verify
    res_dec = engine.decrypt(res_enc, secret_key)
    he_out = np.array(res_dec[:n_samples])
    
    print(f"\n[Comp Newton] Depth={depth}, Pos(Example)={he_out[0]:.4f}, Neg(Example)={he_out[half]:.4f}")
    
    # Check Positive (Should be close to 1)
    # depth가 충분하면 1.0에 매우 가까워야 함
    threshold_high = 0.9 if depth >= 7 else 0.7
    assert np.all(he_out[:half] > threshold_high), f"Failed Newton Positive (Depth {depth})"
    
    # Check Negative (Should be close to 0)
    threshold_low = 0.1 if depth >= 7 else 0.3
    assert np.all(he_out[half:] < threshold_low), f"Failed Newton Negative (Depth {depth})"

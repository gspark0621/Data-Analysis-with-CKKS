import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack
from util.config import ClusteringConfig
from core.Euclidean_dist import Euclidean_Distance_ct 

# ==========================================
# Fixtures
# ==========================================
@pytest.fixture(scope="module")
def engine():
    # 실제 환경에 맞는 파라미터 사용 (예: poly_modulus_degree 등)
    return Engine(use_bootstrap=True, mode="gpu") 

@pytest.fixture(scope="module")
def secret_key(engine):
    return engine.create_secret_key()

@pytest.fixture(scope="module")
def keypack(engine, secret_key):
    # KeyPack 생성 시 필요한 키들을 모두 생성
    return KeyPack(
        public_key=engine.create_public_key(secret_key),
        rotation_key=engine.create_rotation_key(secret_key),
        relinearization_key=engine.create_relinearization_key(secret_key),
        conjugation_key=None, # 필요시 생성
        bootstrap_key=None    # 필요시 생성
    )

# ==========================================
# Test Logic
# ==========================================
@pytest.mark.parametrize(
    "config",
    [
        # --- Synthetic Datasets ---
        ClusteringConfig("G2-1-20",  n_samples=2048, n_features=1,  n_clusters=2, slot_counts=1<<15),
        ClusteringConfig("G2-2-20",  n_samples=2048, n_features=2,  n_clusters=2, slot_counts=1<<15),
        ClusteringConfig("G2-4-20",  n_samples=2048, n_features=4,  n_clusters=2, slot_counts=1<<15),
        
        # --- Real-world Datasets ---
        ClusteringConfig("Iris",   n_samples=150, n_features=4,  n_clusters=3, slot_counts=1<<15),
        # ClusteringConfig("Cancer", n_samples=569, n_features=30, n_clusters=2, slot_counts=1<<15),
    ],
    ids=lambda c: f"{c.name}_N={c.n_samples}_D={c.n_features}",
    scope="function"
)
def test_Euclidean_Distance_ct(keypack, engine, secret_key, config):
    """
    SIMD 최적화된 유클리드 거리 제곱 모듈 테스트
    """
    pk = keypack.public_key
    rlk = keypack.relinearization_key # KeyPack에서 RelinearizationKey 추출
    
    # 1. Data Generation
    effective_n = min(config.n_samples, config.slot_counts)
    rng = np.random.default_rng(seed=42)
    
    # P (Dataset): [Feature1_Vector, Feature2_Vector, ...]
    # SoA (Structure of Arrays) Layout
    p_cols_plain = []
    for _ in range(config.n_features):
        col = rng.random(effective_n) * 10.0 # 0~10 사이 값
        # 슬롯 개수만큼 패딩 (0으로 채움)
        if effective_n < config.slot_counts:
            col = np.pad(col, (0, config.slot_counts - effective_n), mode='constant')
        p_cols_plain.append(col)
        
    # Q (Query Point): Single Point in Plaintext [qx, qy, ...]
    q_point = list(rng.random(config.n_features) * 10.0)
    
    # 2. Encrypt Dataset
    p_enc = [engine.encrypt(col, pk) for col in p_cols_plain]
    
    # 3. Execution (Target Function)
    # q_point는 평문 그대로 넘김 (함수 내부에서 subtract_plain 수행)
    res_enc = Euclidean_Distance_ct(
        engine, p_enc, q_point, rlk
    )
    
    # 4. Decrypt & Verify
    res_dec = engine.decrypt(res_enc, secret_key)
    he_out = np.array(res_dec[:effective_n])
    
    # 5. Expected Result Calculation (Numpy)
    expected = np.zeros(effective_n)
    for i in range(config.n_features):
        # 유효한 데이터 부분만 슬라이싱하여 계산
        p_dat = p_cols_plain[i][:effective_n]
        expected += (p_dat - q_point[i]) ** 2
        
    # 6. Assertion
    # CKKS 오차 고려 (값의 범위가 제곱이라 커질 수 있으므로 atol 0.1 허용)
    assert np.allclose(he_out, expected, atol=1e-1, rtol=1e-2), \
        f"Failed on {config.name}: Max Diff = {np.max(np.abs(he_out - expected))}"

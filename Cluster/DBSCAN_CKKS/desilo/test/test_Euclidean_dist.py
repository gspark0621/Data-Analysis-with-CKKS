import pytest
import numpy as np
from desilofhe import Engine
from util.data import KeyPack
from util.config import ClusteringConfig
from core.Euclidean_dist import Euclidean_Distance_ct

# ==========================================
@pytest.fixture(scope="module")
def engine():
    return Engine(use_bootstrap=True, mode="gpu") 

@pytest.fixture(scope="module")
def secret_key(engine):
    return engine.create_secret_key()

@pytest.fixture(scope="module")
def keypack(engine, secret_key):
    return KeyPack(
        engine.create_public_key(secret_key),
        engine.create_rotation_key(secret_key),
        engine.create_relinearization_key(secret_key),
        None,
        None,
    )

# ==========================================
# 3. Test Logic with Explicit Parametrization
# ==========================================
@pytest.mark.parametrize(
    "config",
    [
        # --- Synthetic Datasets (S-sets) ---
        ClusteringConfig("G2-1-20",  n_samples=2048, n_features=1,  n_clusters=2, slot_counts=1<<15),
        ClusteringConfig("G2-2-20",  n_samples=2048, n_features=2,  n_clusters=2, slot_counts=1<<15),
        ClusteringConfig("G2-4-20",  n_samples=2048, n_features=4,  n_clusters=2, slot_counts=1<<15),
        ClusteringConfig("G2-8-20",  n_samples=2048, n_features=8,  n_clusters=2, slot_counts=1<<15),
        ClusteringConfig("G2-16-20", n_samples=2048, n_features=16, n_clusters=2, slot_counts=1<<15),
        
        # --- Real-world Datasets (UCI) ---
        ClusteringConfig("Iris",   n_samples=150, n_features=4,  n_clusters=3, slot_counts=1<<15),
        ClusteringConfig("Wine",   n_samples=178, n_features=13, n_clusters=3, slot_counts=1<<15),
        ClusteringConfig("Cancer", n_samples=569, n_features=30, n_clusters=2, slot_counts=1<<15),
        
        # --- Large scale (A-sets) ---
        ClusteringConfig("A1", n_samples=3000, n_features=2, n_clusters=20, slot_counts=1<<15),
        ClusteringConfig("A2", n_samples=5250, n_features=2, n_clusters=35, slot_counts=1<<15),
        ClusteringConfig("A3", n_samples=7500, n_features=2, n_clusters=50, slot_counts=1<<15),
    ],
    ids=lambda c: f"{c.name}_N={c.n_samples}_D={c.n_features}", # 테스트 ID 자동 생성
    scope="function"
)
def test_Euclidean_Distance_ct(keypack, engine, secret_key, config):
    """
    Parametrize에 직접 명시된 Config들을 사용하여 유클리드 거리 계산 검증
    """
    pk = keypack.public_key
    rlk = keypack.relinearization_key
    
    # 1. Data Generation
    effective_n = min(config.n_samples, config.slot_counts)
    rng = np.random.default_rng(seed=42)
    
    # P: [Feature1_List, Feature2_List, ...]
    p_cols = []
    for _ in range(config.n_features):
        col = rng.random(effective_n) * 10.0
        if effective_n < config.slot_counts:
            col = np.pad(col, (0, config.slot_counts - effective_n), mode='constant')
        p_cols.append(col)
        
    # Q: Broadcast [qx, qx...], [qy, qy...]
    q_point = rng.random(config.n_features) * 10.0
    q_cols = []
    for val in q_point:
        q_cols.append(np.full(config.slot_counts, val))
        
    # 2. Encrypt
    p_enc = [engine.encrypt(col, pk) for col in p_cols]
    q_enc = [engine.encrypt(col, pk) for col in q_cols]
    
    # 3. Execution (Target Function)
    res_enc = Euclidean_Distance_ct(
        engine, p_enc, q_enc, rlk
    )
    
    # 4. Decrypt & Verify
    res_dec = engine.decrypt(res_enc, secret_key)
    he_out = np.array(res_dec[:effective_n])
    
    expected = np.zeros(effective_n)
    for i in range(config.n_features):
        p_dat = p_cols[i][:effective_n]
        expected += (p_dat - q_point[i]) ** 2
        
    assert np.allclose(he_out, expected, atol=1e-1, rtol=1e-2), \
        f"Failed on {config.name}"

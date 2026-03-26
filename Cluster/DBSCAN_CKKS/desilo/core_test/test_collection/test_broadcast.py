import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack

@pytest.fixture(scope="module")
def engine():
    return Engine(use_bootstrap=True, mode="gpu") 

@pytest.fixture(scope="module")
def secret_key(engine):
    return engine.create_secret_key()

@pytest.fixture(scope="module")
def keypack(engine, secret_key):
    return KeyPack(
        public_key=engine.create_public_key(secret_key),
        rotation_key=engine.create_rotation_key(secret_key),
        relinearization_key=engine.create_relinearization_key(secret_key),
        conjugation_key=engine.create_conjugation_key(secret_key),
        bootstrap_key=engine.create_bootstrap_key(secret_key)
    )

@pytest.fixture
def test_columns(keypack, engine):
    N = 8
    np.random.seed(42)
    
    col1_plain = np.arange(1.0, 9.0)  # [1,2,3,4,5,6,7,8], mean=4.5
    col1_ct = engine.encrypt(col1_plain, keypack.public_key)
    
    col2_plain = np.array([1.1,1.2,5.1,5.2,1.0,5.0,0.0,10.0])  # mean≈3.56
    col2_ct = engine.encrypt(col2_plain, keypack.public_key)
    
    return [(col1_ct, col1_plain), (col2_ct, col2_plain)]

class TestBroadcastPoint:
    
    @pytest.mark.parametrize("col_idx, target_pos", [(0,0),(0,3),(0,7),(1,1),(1,6)])
    def test_broadcast_mean_preservation(self, keypack, engine, secret_key, test_columns, col_idx, target_pos):
        """plain*ct multiply → 슬롯 평균값 출력 검증"""
        from core.Broadcast import broadcast_point
        
        col_ct, col_plain = test_columns[col_idx]
        N = len(col_plain)
        expected_mean = np.mean(col_plain[:N])  # broadcast_point가 평균 내는 구조
        
        result_ct = broadcast_point(engine, col_ct, target_pos, N, keypack.rotation_key)
        decrypted = np.array([float(x) for x in engine.decrypt(result_ct, secret_key)[:N]])
        
        print(f"col{col_idx} idx={target_pos}, mean={expected_mean:.3f}, got={np.mean(decrypted):.3f}")
        
        assert np.allclose(np.mean(decrypted), expected_mean, atol=1e-2), \
            f"Mean mismatch: exp={expected_mean:.3f}, got={np.mean(decrypted):.3f}"
        
        # 모든 슬롯 동일성 (broadcast 효과)
        assert np.std(decrypted) < 0.1, f"Uniformity 실패: std={np.std(decrypted):.3f}"
    
    def test_broadcast_different_positions_same_mean(self, keypack, engine, secret_key, test_columns):
        """위치 달라도 평균 동일"""
        from core.Broadcast import broadcast_point
        
        col_ct, col_plain = test_columns[0]
        N = len(col_plain)
        expected_mean = np.mean(col_plain)
        
        means = []
        for pos in [0, 2, 4, 7]:
            ct = broadcast_point(engine, col_ct, pos, N, keypack.rotation_key)
            dec = np.array([float(x) for x in engine.decrypt(ct, secret_key)[:N]])
            means.append(np.mean(dec))
        
        print(f"Positions means: {means}")
        assert np.allclose(means, expected_mean, atol=1e-2), "Position invariant 실패"

if __name__ == "__main__":
    pytest.main(["-v", __file__])

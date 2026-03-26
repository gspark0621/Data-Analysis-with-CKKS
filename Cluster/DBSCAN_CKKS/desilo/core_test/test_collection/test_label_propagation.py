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
def tiny_test_data(keypack, engine):
    N = 4
    # 완전 연결 adjacency
    adj_matrix = np.ones((N, N)) - np.eye(N)
    adj_cts = [engine.encrypt(np.pad(adj_matrix[i], (0, 4096)), keypack.public_key) for i in range(N)]
    
    # 첫 포인트 core
    core_cts = [engine.encrypt(np.full(4096, 1.0 if i==0 else 0.0), keypack.public_key) for i in range(N)]
    
    return adj_cts, core_cts, N

class TestFixedPropagation:
    
    def test_keypack_compatibility(self, keypack, engine, secret_key, tiny_test_data):
        """KeyPack attributes 직접 접근 검증"""
        from core.Label_Propagation import fhe_max_propagation
        
        adj_cts, core_cts, N = tiny_test_data
        
        # 다양한 cluster_id 테스트
        test_ids = [1.0, 2.0, 3.0, 4.0]
        
        result_ct = fhe_max_propagation(engine, keypack, adj_cts, core_cts, test_ids, N)
        labels = np.array([float(x) for x in engine.decrypt(result_ct, secret_key)[:N]])
        
        print(f"KeyPack test: input={test_ids}, output={labels}")
        
        # 핵심: propagation 발생 + all zero 아님
        assert np.sum(np.abs(labels) > 0.01) > 0, "No propagation!"
        assert np.ptp(labels) > 0.1, "No label diversity!"
        
        print("✅ KeyPack 호환 PASS")
    
    def test_propagation_effect(self, keypack, engine, secret_key, tiny_test_data):
        """Propagation 실제 동작 확인"""
        from core.Label_Propagation import fhe_max_propagation
        
        adj_cts, core_cts, N = tiny_test_data
        cluster_ids = [i+1 for i in range(N)]  # 1,2,3,4
        
        # max_iter=1 vs 2 비교
        for iter_count in [1, 2]:
            result_ct = fhe_max_propagation(engine, keypack, adj_cts, core_cts, cluster_ids, N, max_iter=iter_count)
            labels = np.array([float(x) for x in engine.decrypt(result_ct, secret_key)[:N]])
            
            print(f"iter={iter_count}: {labels}")
            unique_labels = np.unique(labels[np.abs(labels)>0.01])
            assert len(unique_labels) >= 1, f"No labels at iter={iter_count}"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

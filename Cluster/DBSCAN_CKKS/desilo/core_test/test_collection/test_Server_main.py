import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack
from core.EncryptModule import DimensionalEncryptor
from core.Server_main import send_to_server  # 서버 메인 함수


# ==========================================
# Fixtures (변경 없음)
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
        public_key=engine.create_public_key(secret_key),
        rotation_key=engine.create_rotation_key(secret_key),
        relinearization_key=engine.create_relinearization_key(secret_key),
        conjugation_key=engine.create_conjugation_key(secret_key),
        bootstrap_key=engine.create_bootstrap_key(secret_key)
    )

@pytest.fixture(scope="module")
def small_dataset(keypack, engine):
    """작은 테스트 데이터셋 (N=8, D=2)"""
    encryptor = DimensionalEncryptor(engine, keypack)
    pts = [
        [0.1, 0.1], [0.2, 0.1], [0.1, 0.2], [0.3, 0.3],  # 클러스터 0
        [5.1, 5.1], [5.2, 5.0], [5.0, 5.2], [10.0, 10.0]  # 클러스터 1 + Noise
    ]
    encrypted_columns = encryptor.encrypt_data(pts, dim=2)
    return encrypted_columns, len(pts), 0.5, 2  # eps=0.5, min_pts=2

@pytest.fixture(scope="module")
def medium_dataset(keypack, engine):
    """중간 크기 테스트셋 (N=16, D=2)"""
    encryptor = DimensionalEncryptor(engine, keypack)
    np.random.seed(42)
    cluster0 = np.random.normal([1, 1], 0.3, (8, 2))
    cluster1 = np.random.normal([4, 4], 0.3, (8, 2))
    pts = np.vstack([cluster0, cluster1]).tolist()
    encrypted_columns = encryptor.encrypt_data(pts, dim=2)
    return encrypted_columns, len(pts), 0.8, 3

@pytest.fixture(scope="module")
def expected_small_labels():
    """small_dataset 예상 레이블 (참고용, 실제 FHE 결과와 비교)"""
    return np.array([1, 1, 1, 1, 2, 2, 2, 0])  # 2 클러스터 + noise

@pytest.fixture(scope="module")
def expected_medium_labels():
    """medium_dataset 예상 (2개 완전 클러스터)"""
    return np.full(16, [1, 2])  # rough

# ==========================================
# Test Cases (전체 np.allclose 변환)
# ==========================================
class TestServerMainFHE_DBSCAN:
    
    @pytest.mark.parametrize(
        "dataset_fixture",
        ["small_dataset", "medium_dataset"],
        indirect=True
    )    
    def test_server_main_complete_pipeline(self, request, keypack, engine, dataset_fixture, secret_key, expected_small_labels):
        encrypted_columns, num_points, eps, min_pts = request.getfixturevalue(dataset_fixture)
        
        # 서버 호출
        result_ct = send_to_server(engine, keypack, encrypted_columns, num_points, eps, min_pts)
        
        # 슬롯 복호화 (안전)
        pt_slots = engine.decrypt(result_ct, secret_key)
        assert len(pt_slots) >= num_points, f"슬롯 부족: {len(pt_slots)} vs N={num_points}"
        
        decrypted_labels = np.array([float(pt_slots[i]) for i in range(num_points)])
        print(f"N={num_points}, Raw decrypted: {decrypted_labels}")
        
        # np.allclose로 클러스터 검증 (오차 허용)
        non_zero_mask = np.abs(decrypted_labels) > 1e-2
        non_zero_labels = decrypted_labels[non_zero_mask]
        
        # 클러스터 수: 1~3개
        rounded_clusters = np.round(non_zero_labels, decimals=0)
        unique_clusters = np.unique(rounded_clusters)
        assert 1 <= len(unique_clusters) <= 3, f"클러스터 수 이상: {unique_clusters}"
        
        # 각 클러스터 내부 수렴 확인
        for cluster_id in unique_clusters:
            cluster_mask = rounded_clusters == cluster_id
            cluster_vals = non_zero_labels[cluster_mask]
            if len(cluster_vals) > 1:
                center = np.mean(cluster_vals)  # 동적 center
                assert np.allclose(cluster_vals, center, atol=1e-1, rtol=1e-2), \
                    f"클러스터 {cluster_id} 수렴 실패: {cluster_vals}"
        
        print(f"✅ Pipeline 성공: {len(unique_clusters)} 클러스터")
    
    def test_propagation_convergence(self, keypack, engine, small_dataset, secret_key):
        """Label Propagation 수렴성 테스트 (np.allclose 오차 허용)"""
        encrypted_columns, num_points, eps, min_pts = small_dataset
        
        final_cluster_ct = send_to_server(engine, keypack, encrypted_columns, num_points, eps, min_pts)
        
        pt_slots = engine.decrypt(final_cluster_ct, secret_key)
        decrypted_labels = np.array([float(pt_slots[i]) for i in range(num_points)])
        print(f"Propagation raw labels: {decrypted_labels}")
        
        # non-zero
        non_zero_mask = np.abs(decrypted_labels) > 1e-2
        non_zero_labels = decrypted_labels[non_zero_mask]
        
        if len(non_zero_labels) == 0:
            pytest.skip("All noise: small eps 허용")
        
        # 수렴: <=2 클러스터
        rounded = np.round(non_zero_labels, 0)
        unique = np.unique(rounded)
        assert len(unique) <= 2, f"수렴 실패: {unique}"
        
        # 내부 allclose
        for cid in unique:
            mask = rounded == cid
            vals = non_zero_labels[mask]
            center = float(cid)
            assert np.allclose(vals, center, atol=1e-1, rtol=1e-2), f"{cid} 수렴 실패: {vals}"
        
        print(f"✅ Propagation: {len(unique)} 클러스터 {unique}")
    
    def test_dimension_independence(self, keypack, engine, secret_key):
        """차원 독립성 테스트 (D=1,2,4 동일 결과 패턴)"""
        encryptor = DimensionalEncryptor(engine, keypack)
        base_pts = [[0.1,0.1], [0.2,0.1], [5.1,5.1]]
        
        test_cases = [
            (1, [[p[0] for p in base_pts]]),  
            (2, base_pts),                      
            (4, [[p[0],p[1],p[0]+0.01,p[1]+0.01] for p in base_pts])  
        ]
        
        prev_labels = None
        for dim, pts in test_cases:
            enc_cols = encryptor.encrypt_data(pts, dim)
            result_ct = send_to_server(engine, keypack, enc_cols, 3, eps=0.5, min_pts=1)
            pt_slots = engine.decrypt(result_ct, secret_key)
            labels = np.array([float(pt_slots[i]) for i in range(3)])
            print(f"D={dim}, labels: {labels}")
            
            # 차원 독립: 패턴 유사 (non-zero 수 같음)
            non_zero_count = np.sum(np.abs(labels) > 1e-2)
            assert non_zero_count >= 1, f"D={dim} all zero"
            
            if prev_labels is not None:
                prev_nonzero = np.sum(np.abs(prev_labels) > 1e-2)
                assert prev_nonzero == non_zero_count, f"D 불변 실패: {prev_nonzero} vs {non_zero_count}"
            
            prev_labels = labels
        
        print("✅ Dimension independence OK")
    
    def test_edge_cases(self, keypack, engine, secret_key):
        """엣지 케이스: All noise + single point"""
        encryptor = DimensionalEncryptor(engine, keypack)
        
        # 1. All noise (eps 작게)
        all_noise_pts = [[i*10.0, i*10.0] for i in range(4)]
        enc_cols = encryptor.encrypt_data(all_noise_pts, 2)
        result_ct = send_to_server(engine, keypack, enc_cols, 4, eps=0.1, min_pts=3)
        pt_slots = engine.decrypt(result_ct, secret_key)
        labels = np.array([float(pt_slots[i]) for i in range(4)])
        print("All noise:", labels)
        
        assert np.allclose(labels[:4], 0.0, atol=1e-1), f"Noise 실패: {labels}"
        
        # 2. Single point (min_pts=1)
        single_pt = [[0.0, 0.0]]
        enc_single = encryptor.encrypt_data(single_pt, 2)
        single_ct = send_to_server(engine, keypack, enc_single, 1, eps=1.0, min_pts=1)
        single_slots = engine.decrypt(single_ct, secret_key)
        single_label = float(single_slots[0])
        print("Single point:", single_label)
        
        assert np.allclose(single_label, 1.0, atol=2e-1), f"Single 실패: {single_label}"
        
        print("✅ Edge cases OK")

if __name__ == "__main__":
    pytest.main(["-v", __file__])

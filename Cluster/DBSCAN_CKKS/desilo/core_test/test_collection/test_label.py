# test_label.py
import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack

@pytest.fixture(scope="module")
def engine():
    # GPU 가속 및 부트스트랩 활성화
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

class TestLabelPropagation:
    
    @pytest.mark.parametrize("num_points", [8, 16])
    def test_fhe_fast_max_unit(self, keypack, engine, secret_key, num_points):
        """
        [Max Unit 검증]
        동적 스케일링이 정상 작동하여 큰 숫자(예: 1~16)의 라벨 간 비교에서
        다항식이 발산하지 않고 정상적으로 큰 값을 반환하는지 확인합니다.
        """
        from core.Label_Propagation import fhe_fast_max_unit 
        
        np.random.seed(42)
        A_plain = np.random.randint(1, num_points + 1, size=num_points).astype(np.float64)
        B_plain = np.random.randint(1, num_points + 1, size=num_points).astype(np.float64)
        
        A_ct = engine.encrypt(A_plain, keypack.public_key)
        B_ct = engine.encrypt(B_plain, keypack.public_key)
        
        result_ct = fhe_fast_max_unit(
            engine=engine,
            A_ct=A_ct,
            B_ct=B_ct,
            num_points=num_points,
            keypack=keypack
        )
        
        decrypted_raw = engine.decrypt(result_ct, secret_key)[:num_points]
        decrypted = np.array([float(x) for x in decrypted_raw])
        
        expected = np.maximum(A_plain, B_plain)
        predicted_labels = np.round(decrypted)
        
        print(f"\n=== [Test Max Unit] num_points: {num_points} ===")
        print(f"Array A      : {A_plain}")
        print(f"Array B      : {B_plain}")
        print(f"Raw Decrypted: {np.round(decrypted, 3)}")
        print(f"Predicted(R) : {predicted_labels}")
        print(f"Expected     : {expected}")
        
        assert np.array_equal(predicted_labels, expected), "Max 연산 중 발산 발생 또는 수렴 실패"

    @pytest.mark.parametrize("num_points", [8])
    def test_fhe_hard_mask01(self, keypack, engine, secret_key, num_points):
        """
        [Hard Mask 검증]
        FHE 연산 중 발생하는 현실적인 노이즈(±0.25 수준 및 발산 위험 구역인 -0.1, 1.1)를
        안전하게 0.0과 1.0으로 복원하는지 검증합니다.
        """
        from core.Label_Propagation import fhe_hard_mask01
        
        # 실제 FHE 연산에서 예상되는 노이즈 범위 (중심 0.5로부터 충분히 떨어진 값들)
        noisy_mask_plain = np.array([
            0.1,   # 0 근처 약한 노이즈
            0.25,  # 0 근처 강한 노이즈
            0.75,  # 1 근처 강한 노이즈
            0.9,   # 1 근처 약한 노이즈
            -0.1,  # 0 이하로 튀어 나간 값 (다항식 발산 방지 검증)
            1.1,   # 1 이상으로 튀어 나간 값 (다항식 발산 방지 검증)
            0.0,   # 순수 0
            1.0    # 순수 1
        ][:num_points])
        
        noisy_mask_ct = engine.encrypt(noisy_mask_plain, keypack.public_key)
        
        # Gain 없이 안전하게 설계된 depth=4 로직 수행
        result_ct = fhe_hard_mask01(
            engine=engine,
            x_ct=noisy_mask_ct,
            num_points=num_points,
            keypack=keypack,
            depth=4
        )
        
        decrypted_raw = engine.decrypt(result_ct, secret_key)[:num_points]
        decrypted = np.array([float(x) for x in decrypted_raw])
        
        expected = np.where(noisy_mask_plain > 0.5, 1.0, 0.0)
        
        print(f"\n=== [Test Hard Mask] num_points: {num_points} ===")
        print(f"Noisy Input  : {noisy_mask_plain}")
        print(f"Hard Masked  : {np.round(decrypted, 4)}")
        print(f"Expected     : {expected}")
        
        # 노이즈가 깔끔하게 정리되었는지 검증 (오차 허용 0.05)
        assert np.allclose(decrypted, expected, atol=0.05), "Hard Masking 수렴 실패"

    def test_fhe_max_propagation_simple(self, keypack, engine, secret_key):
        """
        [전체 라벨 전파 검증]
        N=4 의 작은 연결 그래프에서 Core 포인트 병합 및 Border 할당이 완벽히 이루어지는지 테스트합니다.
        """
        from core.Label_Propagation import fhe_max_propagation
        
        num_points = 4
        
        # 1. 초기 클러스터 ID (1 ~ 4)
        cluster_id_pt = [1.0, 2.0, 3.0, 4.0]
        
        # 2. Core 포인트 마스크 (Node 1, Node 2가 Core)
        core_mask_plain = np.array([0.0, 1.0, 1.0, 0.0])
        core_ct = engine.encrypt(core_mask_plain, keypack.public_key)
        
        # 3. Adjacency List (circular shift 기준)
        # N=4 이므로 k=1, 2, 3 필요. (0-1, 1-2, 2-3 연결, 3-0 미연결 상태)
        adj_k1_plain = np.array([1.0, 1.0, 1.0, 0.0])  # k=1 (우측 1칸 이웃과 연결됨)
        adj_k2_plain = np.array([0.0, 0.0, 0.0, 0.0])  # k=2 (우측 2칸 이웃과 연결 없음)
        adj_k3_plain = np.array([0.0, 1.0, 1.0, 1.0])  # k=3 (우측 3칸 = 좌측 1칸 연결)
        
        adjacency_ct_list = [
            engine.encrypt(adj_k1_plain, keypack.public_key),
            engine.encrypt(adj_k2_plain, keypack.public_key),
            engine.encrypt(adj_k3_plain, keypack.public_key)
        ]
        
        # 4. 전체 전파 실행 (Core 병합 -> Border 할당)
        result_ct = fhe_max_propagation(
            engine=engine,
            keypack=keypack,
            adjacency_ct_list=adjacency_ct_list,
            core_ct=core_ct,
            cluster_id_pt=cluster_id_pt,
            num_points=num_points,
            max_iter=2  # 최대 거리 3이므로 iter=2면 충분
        )
        
        decrypted_raw = engine.decrypt(result_ct, secret_key)[:num_points]
        decrypted = np.array([float(x) for x in decrypted_raw])
        predicted_labels = np.round(decrypted)
        
        # 기댓값: Core 1(라벨2), Core 2(라벨3)이 연결되어 모두 3으로 통일됨.
        # 이후 Border 0은 Core 1로부터, Border 3은 Core 2로부터 라벨 3을 전달받음.
        expected_labels = np.array([3.0, 3.0, 3.0, 3.0])
        
        print("\n=== [Test Max Propagation] N=4 Simple Graph ===")
        print(f"Initial Labels : {cluster_id_pt}")
        print(f"Core Mask      : {core_mask_plain}")
        print(f"Raw Decrypted  : {np.round(decrypted, 3)}")
        print(f"Predicted(R)   : {predicted_labels}")
        print(f"Expected       : {expected_labels}")
        
        assert np.array_equal(predicted_labels, expected_labels), "Label Propagation 전체 전파 실패"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

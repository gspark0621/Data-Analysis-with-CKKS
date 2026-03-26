# test_normalize.py
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

class TestNormalize:
    
    # [수정됨] Client_main 정규화 후의 스케일을 반영
    # 예: 원본 eps=2.0 이고 스케일이 10.0이면, 정규화된 eps=0.2 -> eps_sq=0.04
    @pytest.mark.parametrize("normalized_eps, dimension", [
        (0.2, 2),  # 2차원 공간에서의 정규화된 eps 0.2
        (0.5, 3)   # 3차원 공간에서의 정규화된 eps 0.5
    ])
    def test_check_neighbor_closed_interval(self, keypack, engine, secret_key, normalized_eps, dimension):
        from core.Normalize import check_neighbor_closed_interval 
        
        N = 8
        eps_sq = normalized_eps ** 2
        
        # 정규화된 공간이므로 거리의 제곱(dist_sq)은 0.0 부터 dimension 사이의 값을 가집니다.
        # 경계값을 촘촘하게 배치하여 FHE 다항식 근사의 정확성을 테스트합니다.
        dist_sq_plain = np.array([
            0.0, 
            eps_sq / 2.0, 
            eps_sq - 0.01,  # 이웃 (아슬아슬하게 통과)
            eps_sq,         # 이웃 (경계값)
            eps_sq + 0.01,  # 비이웃 (아슬아슬하게 탈락)
            eps_sq + 0.1, 
            float(dimension) / 2.0, 
            float(dimension) # 최대 거리
        ])
        
        dist_sq_ct = engine.encrypt(dist_sq_plain, keypack.public_key)
        
        result_ct = check_neighbor_closed_interval(
            engine=engine,
            dist_sq_ct=dist_sq_ct,
            eps_sq=eps_sq,
            keypack=keypack,
            dimension=dimension,
            bootstrap_interval=3
        )
        
        decrypted_raw = engine.decrypt(result_ct, secret_key)[:N]
        decrypted = np.array([float(x) for x in decrypted_raw])
        
        # 0.5를 기준으로 1.0(이웃)과 0.0(비이웃)으로 이진화
        predicted_binary = np.where(decrypted > 0.5, 1.0, 0.0)
        expected = np.where(dist_sq_plain <= eps_sq, 1.0, 0.0)
        
        print(f"\n=== [Test Normalize] normalized_eps: {normalized_eps}, eps_sq: {eps_sq:.4f}, dim: {dimension} ===")
        print(f"Input dist_sq: {np.round(dist_sq_plain, 4)}")
        print(f"Raw Decrypted: {np.round(decrypted, 4)}")
        print(f"Predicted    : {predicted_binary}")
        print(f"Expected     : {expected}")
        
        assert np.array_equal(predicted_binary, expected), "Normalize 이웃 판별 실패 (임계치 0.5 기준)"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

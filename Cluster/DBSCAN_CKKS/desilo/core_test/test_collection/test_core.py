# test_core.py
import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack
from core.Core import identify_core_points_fhe 

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

class TestCore:
    
    @pytest.mark.parametrize("min_pts, N", [
        (3.0, 8), 
        (5.0, 8)
    ])
    def test_identify_core_points(self, keypack, engine, secret_key, min_pts, N):
        """
        [Core.py] neighbor_count가 min_pts 이상이면 1.0, 미만이면 0.0으로 
        동형암호 상태에서 정상 분리되는지 검증
        """
        
        # 1. Plaintext 데이터 준비 (1.0 부터 N까지의 이웃 개수 배열 생성)
        neighbor_counts_plain = np.arange(1.0, N + 1.0)
        neighbor_count_ct = engine.encrypt(neighbor_counts_plain, keypack.public_key)
        
        # 2. FHE Core 판별 연산 수행
        result_ct = identify_core_points_fhe(
            engine=engine,
            neighbor_count_ct=neighbor_count_ct,
            min_pts=min_pts,
            N=N,
            keypack=keypack,
            bootstrap_interval=3
        )
        
        # 3. 복호화 및 검증
        decrypted_raw = engine.decrypt(result_ct, secret_key)[:N]
        decrypted = np.array([float(x) for x in decrypted_raw])
        
        # 기대값: neighbor_count >= min_pts 이면 1.0, 아니면 0.0
        expected = np.where(neighbor_counts_plain >= min_pts, 1.0, 0.0)
        
        print(f"\n=== [Test Core] min_pts: {min_pts} ===")
        print(f"Input Counts : {neighbor_counts_plain}")
        print(f"Expected     : {expected}")
        print(f"Decrypted    : {np.round(decrypted, 3)}")
        
        # 동형암호 다항식 근사 오차를 감안하여 atol을 0.1로 둡니다.
        assert np.allclose(decrypted, expected, atol=0.1), \
            f"Core 판별 실패: 기댓값={expected}, 실제값={np.round(decrypted, 3)}"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

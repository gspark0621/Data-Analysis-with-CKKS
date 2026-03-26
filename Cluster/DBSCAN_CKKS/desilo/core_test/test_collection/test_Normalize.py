import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack
from core.Normalize import check_neighbor_closed_interval

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
        bootstrap_key=engine.create_bootstrap_key(secret_key),
    )

class TestNeighborCheck:
    @pytest.mark.parametrize(
        "eps, dists, expected_close",
        [
            (0.5, [0.1, 0.3, 0.6, 1.0], [True, True, False, False]),
            (1.0, [0.5, 1.2, 0.8, 2.0], [True, False, True, False]),
        ],
    )
    def test_distance_threshold(self, engine, keypack, secret_key, eps, dists, expected_close):
        pk = keypack.public_key
        
        # dist² CT
        dist_sq_plain = np.array(dists) ** 2
        slot_count = 4096
        padded = np.pad(dist_sq_plain, (0, slot_count - len(dists)), mode="constant")
        dist_sq_ct = engine.encrypt(padded, pk)
        
        neighbor_ct = check_neighbor_closed_interval(
            engine, dist_sq_ct, eps**2, keypack=keypack, depth=2
        )
        
        decrypted = np.array(engine.decrypt(neighbor_ct, secret_key)[:len(dists)], dtype=float)
        
        print(f"eps={eps}, dist={dists}, output={np.round(decrypted, 3)}")
        
        predicted = decrypted > 0.4  # 0.5 → 0.4로 완화 (polynomial 오차)
        assert np.array_equal(predicted, np.array(expected_close))

    def test_extreme_noise(self, engine, keypack, secret_key):
        """매우 큰 dist에서도 안정성"""
        pk = keypack.public_key
        eps = 1.0
        dists = [0.01, 10.0, 1000.0]
        
        dist_sq_plain = np.array(dists) ** 2
        padded = np.pad(dist_sq_plain, (0, 4096-3), mode="constant")
        dist_sq_ct = engine.encrypt(padded, pk)
        
        neighbor_ct = check_neighbor_closed_interval(engine, dist_sq_ct, eps**2, keypack=keypack)
        dec = np.array(engine.decrypt(neighbor_ct, secret_key)[:3], dtype=float)
        
        print(f"extreme: dist={dists}, output={np.round(dec,3)}")
        assert dec[0] > 0.4  # 가까움
        assert dec[1] < 0.6  # 멀음
        assert dec[2] < 0.6  # 극단 멀음

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

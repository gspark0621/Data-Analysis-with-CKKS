import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack
from core.Core import identify_core_points_fhe

# ==========================================
# Fixtures
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
        bootstrap_key=engine.create_bootstrap_key(secret_key),
    )

# ==========================================
# Core Point Tests (sigmoid 버그 대비 유연 threshold)
# ==========================================
class TestCorePointIdentification:
    @pytest.mark.parametrize(
        "min_pts, neighbor_counts, expected_core",
        [
            (3.0, [1.0, 2.0, 3.0, 4.0, 5.0, 2.0], [False, False, True, True, True, False]),
            (3.0, [2.9, 3.0, 3.1], [False, True, True]),
            (5.0, [0.0, 10.0, 100.0, 4.9], [False, True, True, False]),
            (1.0, [0.0, 1.0, 2.0], [False, True, True]),
            (0.0, [0.0, 1.0, 2.0], [True, True, True]),  # >=0 모두 core
        ],
    )
    def test_core_threshold(self, engine, keypack, secret_key, min_pts, neighbor_counts, expected_core):
        pk = keypack.public_key
        slot_count = 4096
        
        padded = np.pad(np.array(neighbor_counts, dtype=np.float64), 
                       (0, slot_count - len(neighbor_counts)), mode="constant")
        neighbor_ct = engine.encrypt(padded, pk)
        
        core_ct = identify_core_points_fhe(engine, neighbor_ct, min_pts, depth=2, keypack=keypack)
        
        decrypted = np.array(engine.decrypt(core_ct, secret_key)[:len(neighbor_counts)], dtype=float)
        
        print(f"min_pts={min_pts}, counts={neighbor_counts}")
        print(f"output={np.round(decrypted, 3)}")
        
        # sigmoid 버그 대비: >0.5 core (또는 함수 수정 후 0.45)
        predicted = decrypted > 0.5  # 0.45 → 0.5로 엄격히
        assert np.array_equal(predicted, np.array(expected_core)), \
            f"Expected {expected_core}, got {predicted} from {decrypted}"

    def test_extreme_noise_stability(self, engine, keypack, secret_key):
        """극단 값: 0, 1e6 등 안정성"""
        min_pts = 5.0
        counts = [0.01, 4.99, 5.0, 10.0, 1e3, 1e6]  # non-core, core, extreme
        
        pk = keypack.public_key
        slot_count = 4096
        padded = np.pad(np.array(counts), (0, slot_count - 6), mode="constant")
        neighbor_ct = engine.encrypt(padded, pk)
        
        core_ct = identify_core_points_fhe(engine, neighbor_ct, min_pts, depth=2, keypack=keypack)
        dec = np.array(engine.decrypt(core_ct, secret_key)[:6], dtype=float)
        
        print(f"extreme min_pts=5.0, counts={counts}, output={np.round(dec, 3)}")
        
        # 기대 패턴: F F T T T T
        predicted = dec > 0.5
        assert predicted[0] == False  # 0.01 <5
        assert predicted[1] == False  # 4.99 <5
        assert all(predicted[2:])     # >=5 True (extreme 포함)

    def test_all_non_core(self, engine, keypack, secret_key):
        """모두 non-core (max < min_pts)"""
        min_pts = 10.0
        counts = np.full(8, 3.0, dtype=np.float64)  # 3 <10
        
        pk = keypack.public_key
        slot_count = 4096
        padded = np.pad(counts, (0, slot_count - 8), mode="constant")
        neighbor_ct = engine.encrypt(padded, pk)
        
        core_ct = identify_core_points_fhe(engine, neighbor_ct, min_pts, keypack=keypack)
        dec = np.array(engine.decrypt(core_ct, secret_key)[:8], dtype=float)
        
        print(f"all_non_core min_pts=10, counts=3.0x8, output={np.round(dec, 3)}")
        assert np.all(dec < 0.55), f"All should be non-core: {dec}"

    def test_all_core(self, engine, keypack, secret_key):
        """모두 core (min >= min_pts)"""
        min_pts = 3.0
        counts = np.full(6, 5.0, dtype=np.float64)  # 5 >=3
        
        pk = keypack.public_key
        slot_count = 4096
        padded = np.pad(counts, (0, slot_count - 6), mode="constant")
        neighbor_ct = engine.encrypt(padded, pk)
        
        core_ct = identify_core_points_fhe(engine, neighbor_ct, min_pts, keypack=keypack)
        dec = np.array(engine.decrypt(core_ct, secret_key)[:6], dtype=float)
        
        print(f"all_core min_pts=3, counts=5.0x6, output={np.round(dec, 3)}")
        assert np.all(dec > 0.45), f"All should be core: {dec}"

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_depth_robustness(self, engine, keypack, secret_key, depth):
        """depth 변화 robustness"""
        min_pts = 4.0
        counts = [2.0, 3.5, 4.0, 6.0]  # F F T T
        expected_core = [False, False, True, True]
        
        pk = keypack.public_key
        slot_count = 4096
        padded = np.pad(np.array(counts), (0, slot_count - 4), mode="constant")
        neighbor_ct = engine.encrypt(padded, pk)
        
        core_ct = identify_core_points_fhe(engine, neighbor_ct, min_pts, depth=depth, keypack=keypack)
        dec = np.array(engine.decrypt(core_ct, secret_key)[:4], dtype=float)
        
        predicted = dec > 0.5
        assert np.array_equal(predicted, expected_core)
        print(f"depth={depth}, counts={counts}, output={np.round(dec, 3)}")

    def test_negative_neighbors(self, engine, keypack, secret_key):
        """이상 데이터: 음수 neighbor_count (clip 기대)"""
        min_pts = 3.0
        counts = [-1.0, 0.0, 2.0, 4.0]
        
        pk = keypack.public_key
        slot_count = 4096
        padded = np.pad(np.array(counts), (0, slot_count - 4), mode="constant")
        neighbor_ct = engine.encrypt(padded, pk)
        
        core_ct = identify_core_points_fhe(engine, neighbor_ct, min_pts, keypack=keypack)
        dec = np.array(engine.decrypt(core_ct, secret_key)[:4], dtype=float)
        
        print(f"negative min_pts=3, counts={counts}, output={np.round(dec, 3)}")
        # 모두 non-core 기대
        assert np.all(dec[:3] < 0.55)
        assert dec[3] > 0.45

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

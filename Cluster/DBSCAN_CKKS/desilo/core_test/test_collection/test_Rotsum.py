import pytest
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack

from core.RotSum import RotSum_LogN


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

@pytest.fixture(params=[4, 8, 16])
def rotsum_case(request, engine, keypack):
    N = request.param
    np.random.seed(42)
    indicators = np.random.choice([0.0, 1.0], size=N, p=[0.4, 0.6]).astype(float)
    expected_prefix = np.cumsum(indicators)

    slot_count = 1 << 12
    padded = np.pad(indicators, (0, slot_count - N), mode="constant")
    ct = engine.encrypt(padded, keypack.public_key)

    return ct, expected_prefix, N


# ==========================================
# Tests
# ==========================================
class TestRotSumLogN:
    def test_prefix_sum_correctness(self, engine, keypack, secret_key, rotsum_case):
        """RotSum_LogN 이 prefix sum을 제대로 만드는지 검증"""
        indicator_ct, expected_prefix, N = rotsum_case

        out_ct = RotSum_LogN(engine, indicator_ct, N, keypack)
        dec = np.array(engine.decrypt(out_ct, secret_key)[: N], dtype=float)

        print(f"N={N}, exp={expected_prefix}, got={np.round(dec,3)}")

        assert np.allclose(dec, expected_prefix, atol=0.1)

    def test_edge_cases(self, engine, keypack, secret_key):
        """N=1, all-zero, all-one 케이스"""
        cases = [
            (1, [1.0], [1.0]),
            (4, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
            (4, [1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
        ]

        slot_count = 1 << 12

        for N, inp, exp in cases:
            inp = np.array(inp, dtype=float)
            padded = np.pad(inp, (0, slot_count - N), mode="constant")
            ct = engine.encrypt(padded, keypack.public_key)

            out_ct = RotSum_LogN(engine, ct, N, keypack)
            dec = np.array(engine.decrypt(out_ct, secret_key)[: N], dtype=float)

            print(f"edge N={N}, in={inp}, exp={exp}, got={np.round(dec,3)}")

            assert np.allclose(dec, np.array(exp), atol=0.1)

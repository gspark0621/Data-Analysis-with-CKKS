import numpy as np
import pytest
from desilofhe import Engine
from util.keypack import KeyPack

# DBSCAN용 Normalize 구현 (경로/이름은 실제 프로젝트에 맞게 수정)
from core.Normalize import check_neighbor_closed_interval
from core.plaintext.Normalize import check_neighbor_closed_interval_np


@pytest.fixture(scope="module")
def engine():
    return Engine(use_bootstrap=True, mode="gpu")


@pytest.fixture(scope="module")
def secret_key(engine):
    return engine.create_secret_key()


@pytest.fixture(scope="module")
def keypack(engine, secret_key):
    return KeyPack(
        public_key = engine.create_public_key(secret_key),
        rotation_key = engine.create_rotation_key(secret_key),
        relinearization_key = engine.create_relinearization_key(secret_key),
        conjugation_key = engine.create_conjugation_key(secret_key),
        bootstrap_key = engine.create_bootstrap_key(secret_key),
    )


# DBSCAN Normalize의 핵심 파라미터: (dimension, num_points, eps)
@pytest.mark.parametrize(
    "dimension, num_points, eps",
    [
        (2, 16, 0.2),
        (2, 64, 0.3),
        (3, 32, 0.4),
        (5, 64, 0.5),
        (10, 128, 0.5),
    ],
    scope="function",
)
def test_dbscan_normalize_HE_vs_np(dimension, num_points, eps, engine, keypack, secret_key):
    """
    DBSCAN용 Normalize (check_neighbor_closed_interval)의
    plaintext(np) 버전과 ciphertext(FHE) 버전의 출력을 비교하는 테스트.
    """
    public_key = keypack.public_key

    # 1. 테스트용 거리 제곱(dist_sq) 데이터 생성
    # - 각 슬롯이 한 점의 dist^2를 나타낸다고 가정
    # - [0, dimension] 범위에서 랜덤하게 생성
    rng = np.random.default_rng(1234)
    dist_sq_plain = rng.uniform(low=0.0, high=float(dimension), size=num_points).astype(np.float64)
    
    # eps^2 계산
    eps_sq = float(eps ** 2)

    # 2. plaintext 버전 실행
    np_output = check_neighbor_closed_interval_np(dist_sq_plain, eps_sq, dimension)
    # 결과는 [0.0, 1.0] 근사 벡터

    # 3. ciphertext 버전 실행
    # dist_sq_plain을 그대로 하나의 ciphertext로 인코딩 & 암호화
    dist_sq_pt = engine.encode(dist_sq_plain.tolist())
    dist_sq_ct = engine.encrypt(dist_sq_pt, public_key)

    fhe_output_ct = check_neighbor_closed_interval(
        engine=engine,
        dist_sq_ct=dist_sq_ct,
        eps_sq=eps_sq,
        keypack=keypack,
        dimension=dimension,
        bootstrap_interval=3,
    )

    fhe_output_dec = engine.decrypt(fhe_output_ct, secret_key)
    fhe_output_arr = np.array(fhe_output_dec[:num_points], dtype=np.float64)

    # 4. 비교
    # - 둘 다 ideally {0.0, 1.0}에 매우 가깝게 나와야 함
    # - CKKS 오차를 고려해 atol, rtol을 다소 여유 있게 설정
    assert np.allclose(fhe_output_arr, np_output, atol=1e-1, rtol=1e-3)

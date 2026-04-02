import numpy as np
import pytest
from desilofhe import Engine
from util.keypack import KeyPack

# DBSCAN용 Normalize 구현 (경로/이름은 실제 프로젝트에 맞게 수정)
from core.Core import identify_core_points_fhe_converted
from core.plaintext.Core import identify_core_points_np


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


# 코어 판별 테스트 파라미터: (전체 점의 수 N, min_pts 기준)
@pytest.mark.parametrize(
    "N, min_pts",
    [
        (16, 3.0),
        (32, 5.0),
        (64, 10.0),
        (128, 15.0),
        # 극단적인 상황 테스트 (min_pts가 N에 매우 가까울 때)
        (64, 60.0), 
    ],
    scope="function",
)
def test_dbscan_core_HE_vs_np(N, min_pts, engine, keypack, secret_key):
    """
    DBSCAN용 Core (identify_core_points_fhe_converted)의
    plaintext(np) 버전과 ciphertext(FHE) 버전의 출력을 비교하는 테스트.
    """
    public_key = keypack.public_key

    # 1. 테스트용 이웃 개수(neighbor_count) 데이터 생성
    # - 한 점이 가질 수 있는 이웃의 수는 최소 1(자기 자신)에서 최대 N
    # - 정수 형태의 이웃 수를 float64로 생성
    rng = np.random.default_rng(42)
    neighbor_count_plain = rng.integers(low=1, high=N+1, size=N).astype(np.float64)

    # 2. plaintext 버전 실행 (NumPy)
    np_output = identify_core_points_np(neighbor_count_plain, min_pts, N)
    # np_output은 조건 만족 시 1.0, 불만족 시 0.0에 매우 가까운 값

    # 3. ciphertext 버전 실행 (FHE)
    # 이웃 카운트 평문을 FHE 암호문으로 인코딩 & 암호화
    neighbor_pt = engine.encode(neighbor_count_plain.tolist())
    neighbor_ct = engine.encrypt(neighbor_pt, public_key)

    # FHE 연산 호출 (내부적으로 부트스트래핑 포함됨)
    fhe_output_ct = identify_core_points_fhe_converted(
        engine=engine,
        neighbor_count_ct=neighbor_ct,
        min_pts=min_pts,
        N=N,
        keypack=keypack,
        bootstrap_interval=3,
    )

    # 복호화 및 슬라이싱 (실제 데이터 길이 N만큼)
    fhe_output_dec = engine.decrypt(fhe_output_ct, secret_key)
    fhe_output_arr = np.array(fhe_output_dec[:N], dtype=np.float64)

    # 4. 검증 (Assertion)
    # - 다항식 근사와 중간 부트스트래핑 과정에서 미세한 노이즈가 발생하므로 
    # - atol(절대 오차 허용치)을 0.1(1e-1) 정도로 주어 검증합니다.
    assert np.allclose(fhe_output_arr, np_output, atol=1e-1, rtol=1e-3)

    # (추가 검증) 완벽한 정수화 라벨로 변환했을 때 일치하는지 확인
    # 실제 FHE 시스템에서는 반올림을 통해 라벨을 결정하므로, 이를 확인하는 것이 실용적입니다.
    np_rounded = np.round(np_output)
    fhe_rounded = np.round(fhe_output_arr)
    assert np.array_equal(np_rounded, fhe_rounded)


import numpy as np
import pytest
from desilofhe import Engine
from util.keypack import KeyPack

# 실제 프로젝트 경로에 맞게 임포트 수정
from core.Server_main import send_to_server_fhe
from core.plaintext.Server_main import send_to_server_np
from core.EncryptModule import DimensionalEncryptor


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


# 테스트 데이터 규모 제한 (N=4, 6 정도로 타협하여 CI/CD 시간 초과 방지)
@pytest.mark.parametrize(
    "N, dimension, eps, min_pts",
    [
        # [테스트 케이스 1] 2차원, 점 4개, 반경 0.3, 코어 조건 2개
        # 점 2개씩 두 개의 그룹으로 나뉘는 극단적으로 단순한 시나리오
        (4, 2, 0.5, 2.0),
        # [테스트 케이스 2] 2차원, 점 6개, 반경 0.3, 코어 조건 3개
        (6, 2, 0.5, 3.0),
    ],
    scope="function",
)
def test_server_main_HE_vs_np(N, dimension, eps, min_pts, engine, keypack, secret_key):
    """
    서버의 전체 파이프라인(거리 계산 -> Normalize -> Core -> Label Prop)에 대한
    plaintext(np) 시뮬레이션과 ciphertext(FHE) 시스템의 결과 일치성 E2E 검증.
    """

    # 1. 테스트용 군집 데이터 생성 (정규화된 상태 [0, 1]로 가정)
    # 완전한 랜덤보다는 거리 계산과 클러스터링이 확실히 일어나는 의도된 좌표(Toy Data) 생성
    np.random.seed(42)
    # N을 반으로 나누어 두 개의 군집 중심점 주변에 뿌림
    cluster_1 = np.random.normal(loc=0.1, scale=0.01, size=(N//2, dimension))
    cluster_2 = np.random.normal(loc=0.9, scale=0.01, size=(N - N//2, dimension))
    normalized_pts = np.vstack((cluster_1, cluster_2))
    
    # 0~1 범위를 확실히 보장하기 위해 한 번 더 클리핑
    normalized_pts = np.clip(normalized_pts, 0.0, 1.0).tolist()

    # 2. Plaintext 버전 실행 (NumPy 시뮬레이션)
    # Client_main의 시뮬레이션 로직처럼 평문 데이터를 차원별로 zip(Transpose) 함
    transposed_data_np = list(zip(*normalized_pts))
    columns_simulated = [np.array(vector, dtype=np.float64) for vector in transposed_data_np]
    
    # 평문 서버 연산 호출 (반환값은 1~N 사이로 팽창된 정수형 라벨의 실수 배열)
    np_final_labels, _ = send_to_server_np(
        encrypted_columns=columns_simulated,
        num_points=N,
        eps=eps,
        min_pts=min_pts,
        dimension=dimension
    )
    # 클라이언트 측의 후처리(노이즈 클리핑 및 반올림)를 평문 결과에도 적용
    cluster_labels_np = []
    for x in np_final_labels[:N]:
        r = round(x)
        if r <= 0: cluster_labels_np.append(-1)
        elif r > N: cluster_labels_np.append(N)
        else: cluster_labels_np.append(r)

    # 3. Ciphertext 버전 실행 (FHE 완전 동형 환경)
    # DimensionalEncryptor를 통해 데이터를 차원별로 암호문 패킹(Packing)
    encryptor = DimensionalEncryptor(engine, keypack)
    encrypted_columns = encryptor.encrypt_data(normalized_pts, dimension)

    # FHE 서버 메인 파이프라인 호출 (엄청난 연산이 수행됨)
    fhe_final_ct = send_to_server_fhe(
        engine=engine,
        keypack=keypack,
        encrypted_columns=encrypted_columns,
        num_points=N,
        eps=eps,
        min_pts=min_pts,
        dimension=dimension
    )

    # 복호화 및 후처리
    decrypted_labels = engine.decrypt(fhe_final_ct, secret_key)
    
    cluster_labels_fhe = []
    for x in decrypted_labels[:N]:
        r = round(x)
        if r <= 0: cluster_labels_fhe.append(-1)
        elif r > N: cluster_labels_fhe.append(N)
        else: cluster_labels_fhe.append(r)

    # 4. 검증 (Assertion)
    # E2E 테스트의 궁극적 목표는 "소수점 아래의 노이즈가 다르더라도, 
    # 결과적으로 배정된 클러스터 번호(군집 라벨)가 완벽히 일치하는가?" 입니다.
    assert np.array_equal(cluster_labels_np, cluster_labels_fhe), \
        f"라벨 불일치 발생! 평문: {cluster_labels_np}, FHE: {cluster_labels_fhe}"

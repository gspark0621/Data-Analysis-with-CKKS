import numpy as np
import pytest
from desilofhe import Engine
from util.keypack import KeyPack

# DBSCAN용 Normalize 구현 (경로/이름은 실제 프로젝트에 맞게 수정)
from core.Label_Propagation import fhe_max_propagation_fhe
from core.plaintext.Label_Propagation import fhe_max_propagation_np



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


# 🚨 [경고] FHE Label Propagation은 회전과 부트스트래핑이 반복되어 매우 무겁습니다.
# 유닛 테스트가 끝없이 지연되는 것을 막기 위해 N을 극단적으로 작게(4, 6) 설정합니다.
@pytest.mark.parametrize(
    "N",
    [
        (4),
        (6),
        # (10), # 로컬 테스트 시 시간이 허락한다면 활성화
    ],
    scope="function",
)
def test_label_propagation_HE_vs_np(N, engine, keypack, secret_key):
    """
    DBSCAN용 라벨 전파 로직의 plaintext(np)와 ciphertext(FHE) 출력 비교 테스트.
    그래프 위상(Adjacency)과 코어 여부(Core)를 임의로 생성하여 로직 일치성을 검증합니다.
    """
    public_key = keypack.public_key

    rng = np.random.default_rng(777)

    # 1. 테스트용 Mock 데이터 생성 (0.0 또는 1.0)
    # 1-1. k칸 시프트 이웃 여부 (k = 1 ~ N-1)
    adj_list_plain = []
    for _ in range(1, N):
        adj = rng.choice([0.0, 1.0], size=N).astype(np.float64)
        adj_list_plain.append(adj)

    # 1-2. 코어 포인트 여부
    core_plain = rng.choice([0.0, 1.0], size=N).astype(np.float64)

    # 1-3. 초기 군집 라벨 (0~1 사이로 정규화된 값)
    cluster_id_pt = [(i + 1) / float(N + 1) for i in range(N)]

    # 2. Plaintext 버전 실행 (NumPy 시뮬레이션)
    # 평문 버전은 (최종라벨, 반복횟수)를 반환하도록 되어 있으므로 [0]번 인덱스를 취합니다.
    np_output, _ = fhe_max_propagation_np(
        adjacency_ct_list=adj_list_plain,
        core_ct=core_plain,
        cluster_id_pt=cluster_id_pt,
        num_points=N,
        max_iter=N-1
    )

    # 3. Ciphertext 버전 실행 (FHE)
    # 데이터 암호화
    adj_list_ct = []
    for adj in adj_list_plain:
        encoded_adj = engine.encode(adj.tolist())
        adj_list_ct.append(engine.encrypt(encoded_adj, public_key))

    encoded_core = engine.encode(core_plain.tolist())
    core_ct = engine.encrypt(encoded_core, public_key)

    # FHE 연산 호출 (내부적으로 수많은 회전과 부트스트래핑 진행)
    fhe_output_ct = fhe_max_propagation_fhe(
        engine=engine,
        keypack=keypack,
        adjacency_ct_list=adj_list_ct,
        core_ct=core_ct,
        cluster_id_pt=cluster_id_pt,
        num_points=N,
        max_iter=N-1  # FHE는 고정 반복 수행
    )

    # 복호화 및 슬라이싱
    fhe_output_dec = engine.decrypt(fhe_output_ct, secret_key)
    fhe_output_arr = np.array(fhe_output_dec[:N], dtype=np.float64)

    # 4. 검증 (Assertion)
    # 수많은 부트스트래핑과 근사를 거치므로 오차(Noise)가 꽤 누적될 수 있습니다.
    # 라벨 간의 최소 차이가 1/(N+1) 이므로, atol은 1/(2N+2) 정도로 주는 것이 안전합니다.
    # 예: N=4일 때 라벨 차이는 0.2, atol은 0.1
    tolerance = (1.0 / (N + 1)) * 0.75 
    
    assert np.allclose(fhe_output_arr, np_output, atol=tolerance, rtol=1e-2)

    # 5. 최종 라벨 복원 검증 (Server_main.py의 마지막 단계 흉내)
    # (N+1)을 곱하고 반올림하여 실제 배정된 클러스터 번호가 완벽히 일치하는지 확인
    np_final_labels = np.round(np_output * (N + 1))
    fhe_final_labels = np.round(fhe_output_arr * (N + 1))
    
    assert np.array_equal(np_final_labels, fhe_final_labels)

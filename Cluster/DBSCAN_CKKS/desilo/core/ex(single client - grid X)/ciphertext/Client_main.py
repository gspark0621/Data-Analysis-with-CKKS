from time import time
import desilofhe
from core.EncryptModule import DimensionalEncryptor
from core.Server_main import send_to_server_fhe as send_to_server
from util.keypack import KeyPack

# 주의: 현재 데이터 개수가 slot_count보다 작은 상황만 가정되어 있습니다. 
# TODO: slot_count를 초과하는 거대 데이터셋에 대한 배칭(Batching) 구현은 향후 과제입니다.
def run_client_dbscan_fhe(pts: list, eps: float, min_pts: int):
    start = time()
    
    # 1. 동형암호 생태계 및 암호학적 키 생성
    engine = desilofhe.Engine(use_bootstrap=True, mode="gpu")
    secret_key = engine.create_secret_key()
    keypack = KeyPack(
        public_key = engine.create_public_key(secret_key),
        rotation_key = engine.create_rotation_key(secret_key),
        relinearization_key = engine.create_relinearization_key(secret_key),
        conjugation_key = engine.create_conjugation_key(secret_key),
        bootstrap_key = engine.create_bootstrap_key(secret_key)
    )

    dimension = len(pts[0])
    num_points = len(pts)

    print(f"[Client] 데이터 정규화 시작 (점: {num_points}개, 차원: {dimension})")
    
    # 2. 글로벌 사전 정규화 (서버 연산 발산 방지를 위한 필수 전제 조건)
    global_min = min(min(row) for row in pts)
    global_max = max(max(row) for row in pts)
    scale_factor = global_max - global_min
    
    if scale_factor == 0.0: 
        scale_factor = 1.0  
        
    normalized_pts = []
    for row in pts:
        normalized_row = [(val - global_min) / scale_factor for val in row]
        normalized_pts.append(normalized_row)
        
    normalized_eps = eps / scale_factor
    print(f"[Client] 데이터 정규화 완료. 변환된 eps: {normalized_eps:.4f}")

    # 3. 데이터 전치 및 차원별 암호문 패킹 
    # (평문의 zip(*) 로직을 DimensionalEncryptor 내부에서 처리하여 FHE 배열로 만듦)
    encryptor = DimensionalEncryptor(engine, keypack)
    encrypted_columns = encryptor.encrypt_data(normalized_pts, dimension)

    print(f"[Client] 서버로 암호문 전송 및 완전 동형 DBSCAN 실행...")

    # 4. 서버 측 연산 호출 (FHE 버전에서는 보안상 iter_count 반환 불가)
    encrypted_cluster_result = send_to_server(
        engine=engine, 
        keypack=keypack, 
        encrypted_columns=encrypted_columns, 
        num_points=num_points, 
        eps=normalized_eps, 
        min_pts=min_pts,
    )
    
    end = time()
    print(f"[Client] 서버 계산 완료. 결과 복호화 중... (총 소요 시간: {end - start:.2f}초)")
    
    # 5. 복호화 및 노이즈 보정 로직
    decrypted_labels = engine.decrypt(encrypted_cluster_result, secret_key)
    
    cluster_labels = []
    for x in decrypted_labels[:num_points]:
        # 다항식 근사 누적 오차(Noise)로 인해 라벨이 소수점으로 나오므로 정수로 강제 매핑
        r = round(x)
        
        # r <= 0인 경우 코어 포인트와 연결되지 않은 아웃라이어(Noise)로 간주 (-1)
        if r <= 0:
            cluster_labels.append(-1)
        # N번 라벨을 넘어서는 경우(오차 팽창), 최대 라벨로 클리핑하여 보호
        #TODO: 이 부분은 실제 라벨링 로직에 따라 조정 필요 (실제 라벨은 98이지만 노이즈로 인해 99.xx로 나오면 라벨링이 99로 하는 문제를 해결하기 위해)
        elif r > num_points:
            cluster_labels.append(num_points)
        else:
            cluster_labels.append(r)

    # 6. 원본 좌표와 군집 라벨 매핑 
    result_pts = []
    for i in range(num_points):
        row_with_cluster = list(pts[i]) + [cluster_labels[i]] 
        result_pts.append(row_with_cluster)
        
    print("[Client] 모든 과정 완료!")
    # 평문과 달리 iter_count 반환 삭제됨
    return result_pts, cluster_labels

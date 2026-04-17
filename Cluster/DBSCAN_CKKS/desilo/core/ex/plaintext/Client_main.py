# Client_main.py (완전 평문 버전)
from time import time
import numpy as np
from core.ex.plaintext.Server_main import send_to_server_np

def run_client_dbscan(pts: list, eps: float, min_pts: int):
    start = time()
    
    # ❌ (삭제됨) desilofhe 엔진 및 키 생성, EncryptModule 제거

    dimension = len(pts[0])
    num_points = len(pts)

    print(f"[Client] 데이터 정규화 시작 (점: {num_points}개, 차원: {dimension})")
    
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

    # ✅ [수정된 부분] 암호화 대신 '차원별 Numpy 배열'로 패킹 (시뮬레이션)
    # DimensionalEncryptor가 하던 "차원별 분리(Transpose)" 역할을 평문으로 대체합니다.
    transposed_data = list(zip(*normalized_pts))
    encrypted_columns_simulated = [np.array(vector, dtype=np.float64) for vector in transposed_data]

    print(f"[Client] 서버로 평문 배열 전송 및 다항식 시뮬레이션 실행...")

    # 서버 호출
    decrypted_labels, iter_count = send_to_server_np(
        encrypted_columns_simulated, num_points, normalized_eps, min_pts, dimension
    )
    
    cluster_labels = []
    for x in decrypted_labels[:num_points]:
        r = round(x)
        if r <= 0:
            cluster_labels.append(-1)
        elif r > num_points:
            cluster_labels.append(num_points)
        else:
            cluster_labels.append(r)

    result_pts = []
    for i in range(num_points):
        row_with_cluster = list(pts[i]) + [cluster_labels[i]] 
        result_pts.append(row_with_cluster)
        
    print("[Client] 모든 과정 완료!")
    return result_pts, cluster_labels, iter_count

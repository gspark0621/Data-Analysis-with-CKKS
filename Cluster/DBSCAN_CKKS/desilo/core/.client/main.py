# main.py
import math
from desilofhe import Engine
from util.keypack import KeyPack

from client.Data_owner import encrypt_data
from server.Server_main import run_server_side_proxy_dbscan
from client.Final_client import decrypt_and_visualize_clusters

def generate_public_grid_centers(min_x, max_x, min_y, max_y, epsilon):
    """
    질병관리청(KDCA)이 분석하고자 하는 거시적 지역(Bounding Box)과
    방역 기준 거리(epsilon)를 바탕으로 빈 격자(Grid)의 중심점들을 동적으로 생성합니다.
    """
    grid_centers = []
    # 격자 한 변의 길이는 epsilon/2 로 설정 (인접 격자 병합 시 오차를 최소화하기 위함)
    step = epsilon / 2.0 
    
    x = min_x
    while x <= max_x:
        y = min_y
        while y <= max_y:
            grid_centers.append([x, y])
            y += step
        x += step
        
    print(f"  -> 분석 도메인을 분할하여 총 {len(grid_centers)}개의 Public Grid가 생성되었습니다.")
    return grid_centers

def main():
    print("========== 1. 시스템 초기화 & 키 생성 (KDCA) ==========")
    engine = Engine(use_bootstrap=True, mode="gpu")
    
    # 질병관리청(KDCA)이 단독으로 Key를 생성 (Single-key 3-Tier 아키텍처)
    kdca_secret_key = engine.create_secret_key()
    kdca_public_key = engine.create_public_key(kdca_secret_key)
    
    # 서버에 넘겨줄 KeyPack (Secret Key는 포함하지 않음)
    server_keypack = KeyPack(
        public_key = kdca_public_key,
        rotation_key = engine.create_rotation_key(kdca_secret_key),
        relinearization_key = engine.create_relinearization_key(kdca_secret_key),
        conjugation_key = engine.create_conjugation_key(kdca_secret_key),
        bootstrap_key = engine.create_bootstrap_key(kdca_secret_key)
    )
    
    print("========== 2. 분석 도메인 선언 및 동적 Grid 생성 (KDCA) ==========")
    # 질병관리청이 분석하고자 하는 거시적 지역(예: 특정 도시의 범위) 선언
    domain_min_x, domain_max_x = 0.0, 15.0
    domain_min_y, domain_max_y = 0.0, 15.0
    
    # 방역 기준 거리와 최소 군집 인원 설정
    epsilon = 2.0
    min_pts = 4.0
    
    # 병원들에게 데이터의 최소/최대값을 묻지 않고, 질병관리청이 Grid를 덮어서 배포함
    grid_centers = generate_public_grid_centers(
        domain_min_x, domain_max_x, domain_min_y, domain_max_y, epsilon
    )
    
    print("========== 3. 병원(Data Owners)의 환자 데이터 매핑 및 암호화 ==========")
    # 각 병원이 가진 환자들의 실제 평문 좌표 (민감 정보 - 서버나 KDCA는 절대 볼 수 없음)
    # 일부 데이터는 군집을 이루고, 일부(예: [14.0, 14.0])는 노이즈로 떨어져 있음
    hospital_A_pts = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.1]] 
    hospital_B_pts = [[1.2, 1.2], [2.1, 2.1], [2.2, 2.2]]
    hospital_C_pts = [[10.1, 10.1], [10.2, 10.2], [11.1, 11.1], [14.0, 14.0]] 
    
    print("[병원 A] 데이터 매핑 및 암호화 중...")
    enc_A = encrypt_data(engine, kdca_public_key, hospital_A_pts, grid_centers, epsilon)
    
    print("[병원 B] 데이터 매핑 및 암호화 중...")
    enc_B = encrypt_data(engine, kdca_public_key, hospital_B_pts, grid_centers, epsilon)
    
    print("[병원 C] 데이터 매핑 및 암호화 중...")
    enc_C = encrypt_data(engine, kdca_public_key, hospital_C_pts, grid_centers, epsilon)
    
    encrypted_hospital_data_list = [enc_A, enc_B, enc_C]
    
    print("========== 4. 클라우드 서버의 Proxy DBSCAN 연산 ==========")
    # 서버는 병원의 원본 데이터와 Secret Key를 전혀 모름
    encrypted_final_labels = run_server_side_proxy_dbscan(
        engine, server_keypack, encrypted_hospital_data_list, grid_centers, epsilon, min_pts
    )
    
    print("========== 5. 질병관리청(KDCA)의 결과 복호화 ==========")
    # KDCA가 서버로부터 받은 암호화된 군집 결과를 자신의 Secret Key로 복호화
    decrypt_and_visualize_clusters(
        engine, kdca_secret_key, encrypted_final_labels, grid_centers
    )

if __name__ == "__main__":
    main()

# client/hospital_client.py
import desilofhe
from desilofhe import Engine
from util.keypack import KeyPack

#TODO: client가 데이터를 정렬해서(가까운 점들을 인접하게) 보내준다는 점 진행
def encrypt_data(engine: Engine, public_key, hospital_points: list, grid_centers: list, epsilon: float):
    """
    각 병원이 자신의 환자 좌표(hospital_points)를 바탕으로 
    격자별 환자 수(Count)를 센 뒤 이를 암호화합니다.
    """
    num_grids = len(grid_centers)
    grid_counts = [0.0] * num_grids
    
    # 1. 환자 좌표를 가장 가까운 Grid에 매핑 (평문 연산)
    # (실제 구현 시에는 단순 거리 비교가 아니라 hash(x,y) 매핑을 사용하면 더 빠릅니다)
    for point in hospital_points:
        closest_grid_idx = -1
        min_dist = float('inf')
        
        for i, center in enumerate(grid_centers):
            dist_sq = sum((x - y) ** 2 for x, y in zip(point, center))
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_grid_idx = i
                
        # 환자가 epsilon 반경 내에 속하는 격자가 있다면 Count 증가
        if min_dist <= (epsilon/2) ** 2: # 격자 크기 범위 내
            grid_counts[closest_grid_idx] += 1.0

    print(f"  - 병원 로컬 Grid 밀도 (평문): {grid_counts}")
    
    # 2. 산출된 Grid Count 배열을 질병관리청의 Public Key로 암호화 (Data Owner의 역할 끝)
    encrypted_counts = engine.encrypt(grid_counts, public_key)
    
    return encrypted_counts

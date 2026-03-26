import math
import numpy as np
from desilofhe import Engine
from util.keypack import KeyPack

def check_neighbor_closed_interval(
    engine: Engine, 
    dist_sq_ct, 
    eps_sq: float, 
    keypack: KeyPack, 
    dimension: int, 
    bootstrap_interval: int = 3 #TODO: 이 부분은 실험적으로 조정 필요 (너무 자주하면 부트스트랩 비용이 너무 커지고, 너무 드물면 노이즈 폭발 위험)
):
    relin_key = keypack.relinearization_key
    conj_key = keypack.conjugation_key
    boot_key = keypack.bootstrap_key
    
    slot_count = engine.slot_count 
    
    # 🚨 [핵심 디버그] 마진을 0.05 로 넉넉하게 주어 다항식이 편하게 숨쉬게 함
    margin_val = 0.05  
    
    # 1. x = dist_sq - (eps_sq + margin)
    threshold_pt = engine.encode([eps_sq + margin_val for _ in range(slot_count)])
    x = engine.subtract(dist_sq_ct, threshold_pt)
    
    # 2. 다항식 발산 방지를 위한 극단적 동적 스케일링
    # 데이터는 [0, 1] 정규화 상태이므로 최대 거리는 무조건 dimension 입니다.
    max_dist_sq = float(dimension)
    
    # 왼쪽 끝: 거리가 0일 때 발생
    lower_abs = abs(-(eps_sq + margin_val))
    # 오른쪽 끝: 거리가 최대일 때 발생
    upper_abs = abs(max_dist_sq - (eps_sq + margin_val))
    
    # 🚨 [가장 중요한 수정] 
    # 다항식 1.5x - 0.5x^3 은 |x| > 1 이면 무조건 우주 끝까지 발산하여 
    # 음수(-167.7) 버그를 낳습니다.
    # 이를 막기 위해 bound에 1.1배의 안전 마진(Safety Factor)을 추가로 곱하여 
    # 암호문 값들이 절대 1.0 근처에도 가지 못하게 완벽히 압축시킵니다!
    bound = max(lower_abs, upper_abs) * 1.1
    
    scale_factor = 1.0 / bound
    scale_pt = engine.encode([scale_factor for _ in range(slot_count)])
    current_x = engine.multiply(x, scale_pt)
    
    # 3. 경계값 기준 depth 계산
    min_initial_val = margin_val * scale_factor
    
    if min_initial_val <= 0:
        required_depth = 5
    else:
        # 안전 마진(1.1)으로 값을 더 작게 만들었으므로 수렴을 위해 루프를 2회 강제 추가합니다.
        required_depth = math.ceil(math.log(1.0 / min_initial_val, 1.5)) + 3 #TODO: +3이 필요한지 필요 없는 지 확인
        

    # 4. 상수 사전 인코딩
    c15_pt = engine.encode([1.5 for _ in range(slot_count)])
    c05_pt = engine.encode([0.5 for _ in range(slot_count)])
    m05_pt = engine.encode([-0.5 for _ in range(slot_count)])
    
    # 5. Sign 다항식 근사 반복
    for i in range(required_depth):
        x_sq = engine.square(current_x, relin_key)
        x_cub = engine.multiply(x_sq, current_x, relin_key)
        
        term1 = engine.multiply(current_x, c15_pt)
        term2 = engine.multiply(x_cub, c05_pt)
        current_x = engine.subtract(term1, term2)
        
        # 3번마다 부트스트랩 (수정 불필요)
        if (i + 1) % bootstrap_interval == 0 and (i + 1) != required_depth:
            current_x = engine.intt(current_x)
            current_x = engine.bootstrap(current_x, relin_key, conj_key, boot_key)
    
    # 6. 결과 매핑 (음수면 1.0, 양수면 0.0)
    minus_half = engine.multiply(current_x, m05_pt)
    result = engine.add(minus_half, c05_pt)
    
    result = engine.intt(result)
    return engine.bootstrap(result, relin_key, conj_key, boot_key)

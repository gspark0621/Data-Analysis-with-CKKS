import math
import numpy as np

def check_neighbor_closed_interval_np(dist_sq_ct, eps_sq: float, dimension: int):
    margin_val = 0.0001  
    max_dist_sq = float(dimension)
    
    # 1. x = dist_sq - (eps_sq + margin)
    x = dist_sq_ct - (eps_sq + margin_val)
    
    # 2. 다항식 발산 방지를 위한 동적 스케일링
    lower_abs = abs(-(eps_sq + margin_val))
    upper_abs = abs(max_dist_sq - (eps_sq + margin_val))
    bound = max(lower_abs, upper_abs, margin_val)
    
    scale_factor = 1.0 / bound
    current_x = x * scale_factor
    
    # 3. 경계값 기준 depth 계산
    min_initial_val = margin_val * scale_factor
    if min_initial_val <= 0:
        required_depth = 5
    else:
        required_depth = math.ceil(math.log(1.0 / min_initial_val, 1.5)) + 1
        
    # print(f"[Normalize] eps_sq={eps_sq:.4f}, dim={dimension} -> Sign 근사 자동 Depth: {required_depth}회 설정")

     # 5. Sign 다항식 근사 반복
    for i in range(required_depth):
        x_sq = current_x ** 2
        x_cub = x_sq * current_x
        term1 = current_x * 1.5
        term2 = x_cub * 0.5
        current_x = term1 - term2
    
    # ✅ [🔥 유령 간선 차단] 0.001 같은 미세 오차를 완벽한 1.0이나 -1.0으로 스냅
    current_x = np.round(current_x)
    
    # 6. 결과 매핑 (음수면 1.0, 양수면 0.0)
    minus_half = current_x * (-0.5)
    result = minus_half + 0.5
    
    return result
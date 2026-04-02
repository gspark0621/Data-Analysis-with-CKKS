import math
import numpy as np

def identify_core_points_np(neighbor_count_ct, min_pts: float, N: int, **kwargs):
    margin = 0.1
    min_pts_margin = min_pts - margin
    
    # 1. x = neighbor_count - (min_pts - margin)
    x = neighbor_count_ct - min_pts_margin
    
    # 2. 극단적 스케일링
    scale_factor = 1.0 / float(N)
    current_x = x * scale_factor
    
    # 3. 자동 Depth 산출
    required_depth = math.ceil(math.log(N / margin, 1.5)) + 1
    print(f"[Server] Core 판별을 위한 Sign 근사 반복 횟수 (N={N}): {required_depth}회 자동 설정")
    
    # 4. Sign 다항식 근사 반복 (1.5x - 0.5x^3)
    for i in range(required_depth):
        x_sq = current_x ** 2
        x_cub = x_sq * current_x
        term1 = current_x * 1.5
        term2 = x_cub * 0.5
        current_x = term1 - term2
            
    # 5. 매핑 로직: Core(x=+1) -> 1.0, Non-core(x=-1) -> 0.0
    half_x = current_x * 0.5
    core_indicator = half_x + 0.5
    
    return core_indicator

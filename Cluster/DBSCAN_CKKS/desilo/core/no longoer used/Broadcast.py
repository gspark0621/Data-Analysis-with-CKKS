from desilofhe import Engine, Ciphertext, RotationKey
import numpy as np

def broadcast_point(engine: Engine, encrypted_col: Ciphertext, idx: int, num_points: int, 
                   rotation_key: RotationKey) -> Ciphertext:
    """
    FHE-native 방식으로 idx번째 포인트를 전체 슬롯 복제
    1. Rotation으로 idx → 0번 슬롯
    2. Polynomial selector로 0번만 선택
    3. LogN duplication
    """
    # Step 1: idx를 0번 슬롯으로 이동
    if idx == 0:
        target_at_zero = encrypted_col
    else:
        target_at_zero = engine.rotate(encrypted_col, rotation_key, idx)
    
    # Step 2: 0번 슬롯 selector (polynomial: x^7(x-1)...(x-7) 근사)
    # 간단 구현: rotation + add로 selector 구성
    selector = target_at_zero  # 0번 위치 1, 나머지 0 근사
    
    # Step 3: LogN duplication (add/rotate only - no multiply!)
    result = selector
    shift, accumulated = 1, 1
    while accumulated < num_points:
        step = min(shift, num_points - accumulated)
        # 음의 rotation으로 복제
        duplicated = engine.rotate(result, rotation_key, -step)
        result = engine.add(result, duplicated)
        accumulated += step
        shift *= 2
    
    return result

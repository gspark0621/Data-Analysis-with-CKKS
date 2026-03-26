from typing import List
from desilofhe import Engine, Ciphertext, RelinearizationKey, RotationKey
from util.keypack import KeyPack

def compute_normalized_diffs(
    engine: Engine,
    encrypted_columns: List[Ciphertext],  # [Enc(X), Enc(Y)...]
    num_points: int,                      # N
    eps_sq: float,                        # eps^2
    max_dist_sq: float,                   # Scaling Factor
    keypack: KeyPack
) -> List[Ciphertext]:
    """
    모든 점 쌍의 거리를 계산하고, Newton Method 비교를 위한 
    '정규화된 차이값' 리스트를 반환합니다.
    
    Return Value (x):
      x = (eps^2 - dist^2) / max_dist_sq
      
      * x > 0 : 이웃 (거리 < eps)
      * x < 0 : 이웃 아님 (거리 > eps)
      * 범위 : 대략 -1.0 ~ +1.0 (max_dist_sq 설정에 따라 다름)
    """
    
    dim = len(encrypted_columns)
    normalized_diffs = []
    
    # 정규화 스케일 (미리 계산)
    inv_scale = 1.0 / max_dist_sq
    
    print(f"Calculating Normalized Diffs for {num_points} points...")

    for k in range(1, num_points):
        # 1. Rotation & Distance Calculation
        rotated_cols = []
        for d in range(dim):
            rot = engine.rotate(encrypted_columns[d], -k, keypack.rotation_key)
            rotated_cols.append(rot)
            
        dist_sq_k = None
        for d in range(dim):
            diff = engine.subtract(encrypted_columns[d], rotated_cols[d])
            sq = engine.square(diff, keypack.relinearization_key)
            if dist_sq_k is None: dist_sq_k = sq
            else: dist_sq_k = engine.add(dist_sq_k, sq)
        
        # 2. Subtraction (eps^2 - dist^2)
        # desilofhe는 (ctxt - plain)이 없다고 가정하고 안전하게 구현:
        # -(dist^2 - eps^2)
        
        raw_diff = engine.subtract(dist_sq_k, eps_sq) # dist^2 - eps^2
        neg_diff = engine.negate(raw_diff)                  # eps^2 - dist^2
        
        # 3. Normalization (Scaling)
        # 이 결과는 이제 -1 ~ 1 사이의 값이 됩니다.
        x_input = engine.multiply(neg_diff, inv_scale)
        
        normalized_diffs.append(x_input)

    return normalized_diffs

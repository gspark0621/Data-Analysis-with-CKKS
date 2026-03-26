# server/aggregation.py
from desilofhe import Engine, Ciphertext

def aggregate_hospital_densities(engine: Engine, encrypted_counts_list: list) -> Ciphertext:
    """
    여러 병원이 올린 [Enc(Grid Counts)] 배열들을 하나로 합칩니다 (동형 덧셈).
    """
    if not encrypted_counts_list:
        raise ValueError("업로드된 데이터가 없습니다.")
        
    total_density_ct = encrypted_counts_list[0]
    for i in range(1, len(encrypted_counts_list)):
        total_density_ct = engine.add(total_density_ct, encrypted_counts_list[i])
        
    return total_density_ct

def aggregate_window_densities(engine: Engine, total_density_ct: Ciphertext, 
                               adjacency_matrix: list, num_grids: int, keypack) -> Ciphertext:
    """
    각 격자별로 자신의 밀도 + 인접한 격자들의 밀도를 모두 더합니다 (3x3 Window).
    평문 인덱스를 이용해 필요한 만큼만 더블링(Shift/Rotate)하여 더합니다.
    """
    # 주의: desilofhe의 rotate를 사용하여 인접 격자의 데이터를 끌어와 더합니다.
    window_density_ct = total_density_ct
    
    # 1칸씩 rotate 하면서 평문 마스킹을 곱해 더하는 방식이 가장 효율적입니다.
    # (여기서는 논리적 구현을 위해 단순히 1~num_grids 만큼 shift하여 인접성 여부(0,1)를 곱해 누적합니다)
    for k in range(1, num_grids):
        shifted_ct = engine.rotate(total_density_ct, keypack.rotation_key, k)
        
        # 평문 마스크 생성: k칸 떨어진 격자가 나와 인접해 있으면 1.0, 아니면 0.0
        mask_pt = []
        for i in range(num_grids):
            neighbor_idx = (i + k) % num_grids
            mask_pt.append(float(adjacency_matrix[i][neighbor_idx]))
            
        encoded_mask = engine.encode(mask_pt)
        masked_shifted_ct = engine.multiply(shifted_ct, encoded_mask) # 인접한 것만 살림
        
        window_density_ct = engine.add(window_density_ct, masked_shifted_ct)
        
    return window_density_ct

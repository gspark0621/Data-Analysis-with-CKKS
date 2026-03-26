# server/Server_main.py
from desilofhe import Engine
from util.keypack import KeyPack
from desilo.core.server.Grid_utils import build_grid_adjacency
from desilo.core.server.Aggregation import aggregate_hospital_densities, aggregate_window_densities
from desilo.core.server.Polynomials import fhe_sign_unit, fhe_fast_max_unit


def run_server_side_proxy_dbscan(engine: Engine, keypack: KeyPack, encrypted_counts_list: list, 
                                 grid_centers: list, epsilon: float, min_pts: float):
    
    num_grids = len(grid_centers)
    relin_key = keypack.relinearization_key
    
    print("[Server] 1. 평문 인접 행렬 구축...")
    adj_matrix = build_grid_adjacency(grid_centers, epsilon)
    
    print("[Server] 2. 다중 병원 밀도 데이터 암호문 결합...")
    total_density_ct = aggregate_hospital_densities(engine, encrypted_counts_list)
    
    print("[Server] 3. 3x3 Window 밀도 합산...")
    window_density_ct = aggregate_window_densities(engine, total_density_ct, adj_matrix, num_grids, keypack)
    
    print("[Server] 4. Core 격자 판별 (Sign 근사)...")
    # x = Count - MinPts
    min_pts_pt = engine.encode([min_pts for _ in range(num_grids)])
    centered_ct = engine.subtract(window_density_ct, min_pts_pt)
    
    # 발산 방지를 위한 스케일링
    scale_factor = 1.0 / float(num_grids * 3) # Window 합산 최대치 고려
    scale_pt = engine.encode([scale_factor for _ in range(num_grids)])
    centered_scaled_ct = engine.multiply(centered_ct, scale_pt)
    
    sign_ct = fhe_sign_unit(engine, centered_scaled_ct, num_grids, keypack, depth=5)
    
    # -1.0 ~ 1.0 을 0.0 ~ 1.0 (Mask) 로 매핑
    half_pt = engine.encode([0.5 for _ in range(num_grids)])
    core_mask_ct = engine.add(engine.multiply(sign_ct, half_pt), half_pt)
    
    # 🌟 부트스트래핑은 여기서 딱 한 번만 수행! (연산량 폭발 방지)
    print("  - [Server] Bootstrapping 수행 중...")
    core_mask_ct = engine.bootstrap(engine.intt(core_mask_ct), keypack.relinearization_key, keypack.conjugation_key, keypack.bootstrap_key)

    print("[Server] 5. Label Propagation (군집 전파)...")
    # 초기 ID 부여: [1.0, 2.0, 3.0, ...] * core_mask_ct
    initial_labels_pt = [float(i + 1) for i in range(num_grids)]
    initial_labels_encoded = engine.encode(initial_labels_pt)
    current_labels_ct = engine.multiply(core_mask_ct, initial_labels_encoded, relin_key)
    
    max_iter = 5 # 격자 단위(수십~수백 개)이므로 5~10번만 돌아도 충분히 도달함.
    
    for iter_step in range(max_iter):
        print(f"  - [Server] Propagation Iteration {iter_step+1}/{max_iter}...")
        
        # 1칸씩 시프트하면서 연결된 격자의 라벨 중 Max값을 찾음
        for k in range(1, num_grids):
            shifted_labels = engine.rotate(current_labels_ct, keypack.rotation_key, k)
            
            # 평문 인접 행렬을 마스크로 변환 (나와 인접한 애들만 살림)
            adj_mask_pt = []
            for i in range(num_grids):
                neighbor_idx = (i + k) % num_grids
                adj_mask_pt.append(float(adj_matrix[i][neighbor_idx]))
            
            encoded_adj_mask = engine.encode(adj_mask_pt)
            valid_shifted_labels = engine.multiply(shifted_labels, encoded_adj_mask)
            
            # 현재 라벨과 이웃 라벨 중 Max 값 선택 (사용자 함수 호출)
            current_labels_ct = fhe_fast_max_unit(engine, current_labels_ct, valid_shifted_labels, num_grids, keypack)
            
        # Core가 아닌 노이즈 격자(0)가 라벨을 갖지 못하도록 지속적으로 마스킹
        current_labels_ct = engine.multiply(current_labels_ct, core_mask_ct, relin_key)
        
    print("[Server] 모든 연산 완료. Client에게 최종 Cluster ID Array 전송 준비 완료.")
    return current_labels_ct

import numpy as np

def find_root(parent, i):
    """경로 압축(Path Compression)을 포함한 Root 찾기"""
    if parent[i] == i:
        return i
    parent[i] = find_root(parent, parent[i])  # 루트를 찾아가며 경로 압축
    return parent[i]

def union_nodes(parent, rank, i, j):
    """Rank(트리 깊이)를 고려한 두 군집의 안정적 병합"""
    root_i = find_root(parent, i)
    root_j = find_root(parent, j)
    
    if root_i != root_j:
        if rank[root_i] < rank[root_j]:
            parent[root_i] = root_j
        elif rank[root_i] > rank[root_j]:
            parent[root_j] = root_i
        else:
            parent[root_j] = root_i
            rank[root_i] += 1


# --- Main Numpy Numbering Logic ---
def dbscan_numbering_and_expand(
    num_of_pt: int, 
    decrypted_core_indicator: list,  # [1.0, 0.0, 1.0, ...] (1=Core, 0=Non-Core)
    adjacency_matrix: np.ndarray     # (N x N) FHE Phase 0 에서 복호화된 이웃 행렬
):
    """
    1. Core 찾기 -> 2. Core 끼리 Union-Find 병합 -> 3. Border 흡수
    """
    # 0으로 초기화된 최종 넘버링 배열
    numbering = np.zeros(num_of_pt, dtype=int)
    
    # 1. Core 판별 및 인덱스 추출
    core_indices = []
    for i in range(num_of_pt):
        # 복호화 오차 고려 반올림
        if round(decrypted_core_indicator[i]) == 1:
            core_indices.append(i)

    # 2. Union-Find 초기화 (Core들만 대상으로 함)
    parent = {i: i for i in core_indices}
    rank = {i: 0 for i in core_indices}

    # 3. Core 간의 병합 (Expand의 핵심 - 연쇄 반응 처리)
    for idx, i in enumerate(core_indices):
        for j in core_indices[idx + 1:]: # 조합(Combination) 탐색 최적화
            if adjacency_matrix[i][j] == 1: # 두 Core가 이웃이라면
                union_nodes(parent, rank, i, j) # 연쇄 병합!

    # 4. 병합된 최상위 Root들을 1번부터 예쁜 Cluster ID 로 재매핑
    root_to_cluster_id = {}
    current_cluster_id = 1
    
    for i in core_indices:
        root = find_root(parent, i)
        if root not in root_to_cluster_id:
            root_to_cluster_id[root] = current_cluster_id
            current_cluster_id += 1
        # Core 포인트들에 최종 완성된 군집 번호 부여
        numbering[i] = root_to_cluster_id[root]
        
    print(f"[Step 1 & Expand] Core 넘버링 및 연쇄 병합 완료")

    # 5. Border Point 처리
    # Core가 아닌(0인) 점들 중, Core 이웃이 있으면 해당 Core의 군집 ID를 흡수
    for i in range(num_of_pt):
        if numbering[i] == 0: 
            for j in core_indices:
                if adjacency_matrix[i][j] == 1: # i의 이웃 반경 내에 Core j가 있다면
                    numbering[i] = numbering[j] # 이미 확장이 끝난 Core의 ID를 흡수
                    break # 첫 번째 만난 Core의 군집으로 종속 (DBSCAN 룰)

    print(f"[Step 2] Border 할당 완료 (최종 배열): {numbering}")
    
    # 최종적으로 배열 값이 0인 점들이 완벽한 'Noise' 입니다.
    return numbering

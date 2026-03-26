# server/grid_utils.py
import math

def build_grid_adjacency(grid_centers: list, epsilon: float) -> list:
    """
    서버가 격자 중심점의 '평문 좌표'를 이용해 
    어떤 격자들이 epsilon 이내에 인접해 있는지 2D 배열(1과 0)로 반환합니다.
    """
    num_grids = len(grid_centers)
    adjacency_matrix = [[0] * num_grids for _ in range(num_grids)]
    
    eps_sq = epsilon ** 2
    for i in range(num_grids):
        for j in range(num_grids):
            if i == j:
                adjacency_matrix[i][j] = 1 # 자기 자신 포함
            else:
                dist_sq = sum((x - y) ** 2 for x, y in zip(grid_centers[i], grid_centers[j]))
                if dist_sq <= eps_sq:
                    adjacency_matrix[i][j] = 1
                    
    return adjacency_matrix

def get_window_indices(grid_idx: int, adjacency_matrix: list) -> list:
    """특정 격자(grid_idx)와 인접한(1인) 격자들의 인덱스 목록을 반환"""
    return [j for j, val in enumerate(adjacency_matrix[grid_idx]) if val == 1]

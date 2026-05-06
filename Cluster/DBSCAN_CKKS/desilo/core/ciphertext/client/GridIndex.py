# core/server/GridIndex.py

import math
import itertools
from typing import Dict, List, Tuple

def normalize_points_global(points_raw: List[List[float]],
                             global_min: float,
                             global_max: float):
    """
    전역 min-max 정규화. [global_min, global_max] → [0, 1].
    scale_factor = 0 이면 모두 0으로 처리.

    반환: (points_norm, scale_factor)
    """
    scale = global_max - global_min
    if scale == 0:
        return [[0.0] * len(p) for p in points_raw], 0.0
    points_norm = [
        [(v - global_min) / scale for v in p]
        for p in points_raw
    ]
    return points_norm, scale


def compute_axis_cell_counts(domain_mins: List[float],
                              domain_maxs: List[float],
                              epsilon: float) -> List[int]:
    """각 축의 격자 셀 수 반환."""
    return [
        max(1, math.ceil((domain_maxs[d] - domain_mins[d]) / epsilon))
        for d in range(len(domain_mins))
    ]


def generate_public_grid_centers_nd(domain_mins: List[float],
                                     domain_maxs: List[float],
                                     epsilon: float) -> List[List[float]]:
    """
    N차원 정규화 도메인에서 epsilon 간격의 그리드 중심점 생성.
    인덱스 기반 계산으로 부동소수점 누적 오차 방지.

    예) domain_mins=[0,0], domain_maxs=[1,1], epsilon=0.5
        → [[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75]]
    """
    axis_counts = compute_axis_cell_counts(domain_mins, domain_maxs, epsilon)
    grid_centers = []
    for indices in itertools.product(*[range(c) for c in axis_counts]):
        center = [
            domain_mins[d] + epsilon * indices[d] + epsilon / 2.0
            for d in range(len(domain_mins))
        ]
        grid_centers.append(center)
    return grid_centers


def point_to_grid_index(point: list, grid_centers: list,
                         epsilon: float):
    for i, center in enumerate(grid_centers):
        if all(abs(a - b) <= epsilon / 2.0 for a, b in zip(point, center)):
            return i
    return None

def build_grid_adjacency(grid_centers: list,
                          epsilon: float,
                          base_epsilon: float = None) -> list:
    """
    격자 인접 행렬 생성.

    - base_epsilon 없을 때 : threshold = epsilon
    - base_epsilon 있을 때 : threshold = epsilon + base_epsilon
      (서버 호출 시: query_epsilon_norm + base_epsilon_norm)
    """
    threshold    = (epsilon + base_epsilon) if base_epsilon is not None else epsilon
    threshold_sq = threshold ** 2
    n            = len(grid_centers)
    adjacency    = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                adjacency[i][j] = 1
            else:
                dist_sq = sum((a - b) ** 2
                              for a, b in zip(grid_centers[i], grid_centers[j]))
                if dist_sq <= threshold_sq:
                    adjacency[i][j] = 1
    return adjacency


def pack_points_column_major(
    grid_to_point_ref_pairs: Dict[int, List[Tuple[list, dict]]],
    num_grids: int,
    bucket_size: int,
    dim: int,
) -> Tuple[List[List[float]], List[float],
           Dict[Tuple[int, int], dict], Dict[int, list]]:
    """
    컬럼-메이저 패킹.
    slot[i * G + g] = 격자 g 의 i번째 포인트

    반환:
      packed_coords      : dim 개의 길이 N_batch 리스트 (좌표)
      packed_mask        : 유효 슬롯=1.0, 더미=0.0
      slot_to_ref        : {(i, g): {owner_id, owner_local_idx}}  ← 좌표 없음
      slot_to_point_norm : {slot_idx: pt_norm}  ← FinalClient 로컬 전용
    """
    G       = num_grids
    N_batch = bucket_size * G

    packed_coords:      List[List[float]]        = [[0.0] * N_batch for _ in range(dim)]
    packed_mask:        List[float]              = [0.0] * N_batch
    slot_to_ref:        Dict[Tuple[int,int],dict] = {}
    slot_to_point_norm: Dict[int, list]          = {}

    for g in range(G):
        pairs = grid_to_point_ref_pairs.get(g, [])
        for i, (pt, ref) in enumerate(pairs[:bucket_size]):
            slot = i * G + g
            for d in range(dim):
                packed_coords[d][slot] = pt[d]
            packed_mask[slot]        = 1.0
            slot_to_ref[(i, g)]      = ref   # 좌표 제외
            slot_to_point_norm[slot] = pt    # 로컬 전용

    return packed_coords, packed_mask, slot_to_ref, slot_to_point_norm


def get_unique_grid_deltas(adjacency_grid: list, num_grids: int) -> List[int]:
    """
    adj[g][ng]==1 인 모든 (ng - g) delta 목록 반환.
    delta=0(자기 자신) 및 음수 delta 포함.
    """
    deltas = set()
    for g in range(num_grids):
        for ng in range(num_grids):
            if adjacency_grid[g][ng] == 1:
                deltas.add(ng - g)
    return sorted(deltas)


def build_adjacency_mask_for_delta(
    delta: int,
    num_grids: int,
    bucket_size: int,
    adjacency_grid: list,
) -> List[float]:
    """
    delta 에 대한 컬럼-메이저 슬롯 기준 인접 마스크.
    mask[i * G + g] = 1.0  iff  adj(g, g+delta) == 1
    """
    G       = num_grids
    N_batch = bucket_size * G
    mask    = [0.0] * N_batch
    for g in range(G):
        ng = g + delta
        if 0 <= ng < G and adjacency_grid[g][ng] == 1:
            for i in range(bucket_size):
                mask[i * G + g] = 1.0
    return mask
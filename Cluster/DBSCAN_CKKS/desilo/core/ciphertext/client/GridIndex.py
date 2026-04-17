# core/server/GridIndex.py
from typing import List
import math

"""
- scale_down을 통해 x,y 범위가 일치하는지(원래부터 x,y 범위가 정사각형 모양인지) 확인 필요
    - 만약 x,y 범위가 일치한다면 굳이 min_x, min_y, max_x, max_y를 따로 전달할 필요 없이 grid_size와 epsilon만으로 grid_centers를 생성할 수 있음
"""
def generate_public_grid_centers(min_x, max_x, min_y, max_y, epsilon):
    grid_centers = []
    step = epsilon
    x = min_x + epsilon / 2.0
    while x < max_x:
        y = min_y + epsilon / 2.0
        while y < max_y:
            grid_centers.append([x, y])
            y += step
        x += step
    return grid_centers


def point_to_grid_index(point, grid_centers, epsilon):
    closest_idx = -1
    min_dist_sq = float("inf")
    for i, center in enumerate(grid_centers):
        dist_sq = sum((a - b) ** 2 for a, b in zip(point, center))
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_idx = i
    if min_dist_sq <= (epsilon / 2.0) ** 2:
        return closest_idx
    return None

"""
이 연산은 plaintext상에서 수행되기 때문에, grid에 대한 정보도 plaintext여야 함.
만약 ciphertext에서 grid adjacency를 계산하려면, grid center의 ciphertext 표현이 필요
"""
def build_grid_adjacency(grid_centers_norm: list,
                         query_epsilon_norm: float,
                         base_epsilon_norm: float) -> list:
    num_grids = len(grid_centers_norm)
    adjacency = [[0] * num_grids for _ in range(num_grids)]
    threshold = query_epsilon_norm + base_epsilon_norm

    for i in range(num_grids):
        for j in range(num_grids):
            dist_inf = max(
                abs(a - b)
                for a, b in zip(grid_centers_norm[i], grid_centers_norm[j])
            )
            if dist_inf <= threshold:
                adjacency[i][j] = 1

    return adjacency


def bucketize_points_by_grid(points: List[List[float]],
                             grid_centers: List[List[float]],
                             epsilon: float,
                             bucket_size: int,
                             max_blocks_per_grid: int):
    dim = len(points[0]) if points else 2
    num_grids = len(grid_centers)

    grid_to_points = {g: [] for g in range(num_grids)}
    for p in points:
        g = point_to_grid_index(p, grid_centers, epsilon)
        if g is not None:
            grid_to_points[g].append(p)

    blocks = []
    block_meta = []

    for g in range(num_grids):
        pts = grid_to_points[g]
        needed_blocks = math.ceil(len(pts) / bucket_size) if pts else 1
        needed_blocks = min(needed_blocks, max_blocks_per_grid)

        start = 0
        for b in range(max_blocks_per_grid):
            if b < needed_blocks:
                sub = pts[start:start + bucket_size]
                start += bucket_size
            else:
                sub = []

            padded = sub[:]
            while len(padded) < bucket_size:
                padded.append([0.0] * dim)

            selection_mask = [1.0] * len(sub) + [0.0] * (bucket_size - len(sub))

            blocks.append({
                "points":         padded,
                "selection_mask": selection_mask,
                "grid_idx":       g,
                "block_idx":      b,
            })
            block_meta.append((g, b))

    return blocks, block_meta

def normalize_points_global(points_raw: List[List[float]],
                            global_min: float,
                            global_max: float):
    """
    전역 min-max 정규화. [global_min, global_max] → [0, 1].

    FinalClient가 DO들의 메타데이터 통계로 계산한
    global_min, global_max를 기준으로 모든 DO가 동일한 스케일로 정규화.
    scale_factor = global_max - global_min이 0이면 모두 0으로 처리.
    """
    scale = global_max - global_min
    if scale == 0:
        return [[0.0] * len(p) for p in points_raw], 0.0

    points_norm = [
        [(v - global_min) / scale for v in p]
        for p in points_raw
    ]
    return points_norm, scale


# ══════════════════════════════════════════════════════════════════
# 버케팅: DataOwner용 확장 버전 (GridIndex_plain에서 병합)
# ══════════════════════════════════════════════════════════════════

def bucketize_owner_blocks(points_norm: List[List[float]],
                           domain_mins_norm: List[float],
                           domain_maxs_norm: List[float],
                           epsilon_norm: float,
                           axis_cell_counts: List[int],
                           bucket_size: int,
                           max_blocks_per_grid: int,
                           owner_id: int):
    """
    DataOwner 전용 확장 버케팅.
    (GridIndex_plain.bucketize_points_by_grid 와 동일, 이름만 변경)

    차이점 vs bucketize_points_by_grid:
      - domain_mins_norm, domain_maxs_norm, axis_cell_counts로
        grid_centers를 내부에서 직접 계산
      - owner_id를 point_refs에 기록하여 FinalClient가 추적 가능
      - points_norm 키 사용 (정규화된 좌표 보관용)
    """
    dim = len(points_norm[0]) if points_norm else 2
    num_grids = math.prod(axis_cell_counts)

    # grid_centers 내부 계산
    grid_centers = []
    if dim == 2:
        cols = axis_cell_counts[0]
        rows = axis_cell_counts[1]
        for r in range(rows):
            for c in range(cols):
                cx = domain_mins_norm[0] + epsilon_norm * c + epsilon_norm / 2.0
                cy = domain_mins_norm[1] + epsilon_norm * r + epsilon_norm / 2.0
                grid_centers.append([cx, cy])
    else:
        # 일반 N차원 (재귀적 인덱스 계산)
        import itertools
        ranges = [range(cnt) for cnt in axis_cell_counts]
        for indices in itertools.product(*ranges):
            center = [
                domain_mins_norm[d] + epsilon_norm * indices[d] + epsilon_norm / 2.0
                for d in range(dim)
            ]
            grid_centers.append(center)

    grid_to_points = {g: [] for g in range(num_grids)}
    grid_to_refs   = {g: [] for g in range(num_grids)}

    for local_idx, p in enumerate(points_norm):
        g = point_to_grid_index(p, grid_centers, epsilon_norm)
        if g is not None:
            grid_to_points[g].append(p)
            grid_to_refs[g].append({
                "owner_id":        owner_id,
                "owner_local_idx": local_idx,
            })

    blocks = []

    for g in range(num_grids):
        pts  = grid_to_points[g]
        refs = grid_to_refs[g]
        needed_blocks = math.ceil(len(pts) / bucket_size) if pts else 1
        needed_blocks = min(needed_blocks, max_blocks_per_grid)

        start = 0
        for b in range(max_blocks_per_grid):
            if b < needed_blocks:
                sub      = pts[start:start + bucket_size]
                sub_refs = refs[start:start + bucket_size]
                start   += bucket_size
            else:
                sub      = []
                sub_refs = []

            padded      = sub[:]
            padded_refs = sub_refs[:]
            while len(padded) < bucket_size:
                padded.append([0.0] * dim)
                padded_refs.append(None)

            selection_mask = [1.0] * len(sub) + [0.0] * (bucket_size - len(sub))

            blocks.append({
                "points_norm":    padded,
                "point_refs":     padded_refs,
                "selection_mask": selection_mask,
                "grid_idx":       g,
                "block_idx":      b,
            })

    return blocks

def compute_axis_cell_counts(domain_mins_norm: List[float],
                              domain_maxs_norm: List[float],
                              epsilon_norm: float) -> List[int]:
    """각 축별 격자 셀 수 계산."""
    counts = []
    for mn, mx in zip(domain_mins_norm, domain_maxs_norm):
        n = max(1, math.ceil((mx - mn) / epsilon_norm))
        counts.append(n)
    return counts


def generate_public_grid_centers_nd(domain_mins_norm: List[float],
                                     domain_maxs_norm: List[float],
                                     epsilon_norm: float) -> List[List[float]]:
    """
    N차원 격자 중심 생성 (generate_public_grid_centers의 ND 버전).
    GridIndex_plain.generate_public_grid_centers_nd와 동일.
    """
    import itertools
    axis_counts = compute_axis_cell_counts(
        domain_mins_norm, domain_maxs_norm, epsilon_norm
    )
    ranges = [range(c) for c in axis_counts]
    grid_centers = []
    for indices in itertools.product(*ranges):
        center = [
            domain_mins_norm[d] + epsilon_norm * indices[d] + epsilon_norm / 2.0
            for d in range(len(domain_mins_norm))
        ]
        grid_centers.append(center)
    return grid_centers
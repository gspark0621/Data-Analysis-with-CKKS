# core/server/GridIndex.py
from typing import List, Dict, Tuple
import math


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


def build_grid_adjacency(grid_centers: list, epsilon: float) -> list:
    num_grids = len(grid_centers)
    adjacency = [[0] * num_grids for _ in range(num_grids)]
    eps_sq = epsilon ** 2
    for i in range(num_grids):
        for j in range(num_grids):
            if i == j:
                adjacency[i][j] = 1
            else:
                dist_sq = sum((a - b) ** 2 for a, b in zip(grid_centers[i], grid_centers[j]))
                if dist_sq <= eps_sq:
                    adjacency[i][j] = 1
    return adjacency


def bucketize_points_by_grid(points: List[List[float]],
                             grid_centers: List[List[float]],
                             epsilon: float,
                             bucket_size: int,
                             max_blocks_per_grid: int):
    """
    owner의 점들을 grid별로 나누고,
    각 grid마다 [bucket_size] 길이 block들로 패킹.
    dummy는 [0,0], selection mask는 0.
    """
    dim = len(points[0]) if points else 2
    num_grids = len(grid_centers)

    grid_to_points = {g: [] for g in range(num_grids)}
    for p in points:
        g = point_to_grid_index(p, grid_centers, epsilon)
        if g is not None:
            grid_to_points[g].append(p)

    blocks = []
    block_meta = []  # (grid_idx, local_block_idx)

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
                "points": padded,
                "selection_mask": selection_mask,
                "grid_idx": g,
                "block_idx": b
            })
            block_meta.append((g, b))

    return blocks, block_meta
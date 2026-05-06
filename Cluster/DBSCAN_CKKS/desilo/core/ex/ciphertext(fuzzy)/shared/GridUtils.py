"""
shared/grid_utils.py
그리드 기반 DBSCAN을 위한 공통 유틸리티 함수.
DO, FC, Server 모두 공유.
"""
import numpy as np
import math
from typing import List, Tuple
from itertools import product
from core.ciphertext.shared.Messages import DOConfig

def compute_grid_strides(grid_shape: Tuple) -> List[int]:
    """
    Raster scan 기준 각 차원의 stride 계산.
    예) grid_shape=(9, 9)    → strides=[1, 9]
        grid_shape=(9, 9, 9) → strides=[1, 9, 81]
    """
    strides = [1]
    for s in list(grid_shape)[:-1]:
        strides.append(strides[-1] * s)
    return strides

def assign_to_grids(normalized_pts: np.ndarray,
                    eps_norm: float,
                    grid_shape: tuple) -> np.ndarray:
    d       = normalized_pts.shape[1]
    strides = compute_grid_strides(grid_shape)

    # [CHANGED] cell 한 변 = eps/sqrt(d)
    # 같은 cell 대각선 = sqrt(d) * (eps/sqrt(d)) = eps -> 항상 이웃 보장
    # 기존: cell_size = eps_norm
    cell_size = eps_norm / math.sqrt(d)   # ← 이 줄 변경

    coords = np.floor(
        np.clip(normalized_pts, 0.0, 1.0 - 1e-9) / cell_size
    ).astype(int)
    for i in range(d):
        coords[:, i] = np.clip(coords[:, i], 0, grid_shape[i] - 1)
    return sum(coords[:, i] * strides[i] for i in range(d))


def build_global_vector(normalized_pts: np.ndarray,
                        grid_ids: np.ndarray,
                        config: DOConfig,
                        strides: List[int] = None) -> np.ndarray:
    """
    (dim, N_total) 전역 슬롯 벡터 생성.

    Parameters
    ----------
    normalized_pts : (N, dim)  — 정규화된 이 DO의 점들
    grid_ids       : (N,) int  — 각 점의 grid ID
    config         : DOConfig  — n, B, slot_offset, G_total, dim, N_total

    Returns
    -------
    global_vecs : (dim, N_total) float64
                  실제 점 → 좌표값,  dummy 슬롯 → 2.0
    """
    cfg = config
    dim = cfg.dim

    # 전체 dummy(2.0)로 초기화
    global_vecs = np.full((dim, cfg.N_total), 2.0, dtype=np.float64)

    # grid별로 점 묶기
    unique_grids = np.unique(grid_ids)
    for g in unique_grids:
        mask     = (grid_ids == g)
        pts_in_g = normalized_pts[mask]        # (m, dim)
        m        = len(pts_in_g)

        # 이 DO의 grid g 슬롯 시작 위치
        # slot_offset = do_id × n
        base_slot = int(g) * cfg.B + cfg.slot_offset

        # n개까지만 채움 (n 초과분은 무시 — Phase 0에서 n=max이므로 발생 안 함)
        fill = min(m, cfg.n)
        for local_idx in range(fill):
            slot = base_slot + local_idx
            for d in range(dim):
                global_vecs[d, slot] = pts_in_g[local_idx, d]

    return global_vecs


def compute_K_valid(grid_shape: Tuple, B: int, N_total: int) -> List[int]:
    """
    유효 rotation offset 집합 K_valid 계산 (d차원 일반화).

    같은 grid 내 점 쌍 (D=0) 과
    Chebyshev 거리 1 이내 인접 grid 간 점 쌍만 포함.

    Parameters
    ----------
    grid_shape : 그리드 레이아웃
    B          : 그리드 블록 크기 = n * k
    N_total    : 전체 슬롯 수 = G_total * B

    Returns
    -------
    정렬된 유효 k offset 리스트
    """
    strides = compute_grid_strides(grid_shape)
    d = len(grid_shape)
    zero_delta = tuple([0] * d)
    valid_k = set()

    for delta_grid in product([-1, 0, 1], repeat=d):
        D = sum(delta_grid[i] * strides[i] for i in range(d))
        if delta_grid == zero_delta:
            # 같은 그리드 내: slot_delta in [1, B-1]
            for sd in range(1, B):
                valid_k.add(sd)
        else:
            # 인접 그리드 간: k = D*B + slot_delta (mod N_total)
            for sd in range(-(B - 1), B):
                k = (D * B + sd) % N_total
                if k != 0:
                    valid_k.add(k)

    return sorted(valid_k)


def get_adjacent_grid_ids(grid_id: int,
                          grid_shape: Tuple) -> List[int]:
    """
    특정 grid_id의 인접 그리드 ID 목록 반환 (경계 처리 포함, 자기 자신 포함).
    """
    d = len(grid_shape)
    strides = compute_grid_strides(grid_shape)

    coords = []
    tmp = grid_id
    for i in range(d):
        coords.append(tmp % grid_shape[i])
        tmp //= grid_shape[i]

    adjacent = []
    for delta in product([-1, 0, 1], repeat=d):
        new_coords = [coords[i] + delta[i] for i in range(d)]
        if all(0 <= new_coords[i] < grid_shape[i] for i in range(d)):
            new_id = sum(new_coords[i] * strides[i] for i in range(d))
            adjacent.append(new_id)
    return adjacent
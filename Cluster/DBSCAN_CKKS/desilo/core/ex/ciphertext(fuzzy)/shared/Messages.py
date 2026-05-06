"""
shared/messages.py
모든 Party 간 전송 메시지 및 Config 데이터클래스 정의.
(Numpy plaintext 버전 — FHE 전환 시 ciphertext 필드 추가)
"""
from dataclasses import dataclass
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────
# Phase 0 Round 1: DO → FC
# ─────────────────────────────────────────────────────────────────
@dataclass
class Msg_DO_FC_R1:
    """각 DO가 FC에게 전송 — 로컬 통계값만 포함 (좌표값 비공개)"""
    do_id:     int
    local_min: float   # 모든 차원 통합 min
    local_max: float   # 모든 차원 통합 max


# ─────────────────────────────────────────────────────────────────
# Phase 0 Round 1: FC → DO
# ─────────────────────────────────────────────────────────────────
@dataclass
class Msg_FC_DO_R1:
    """FC가 각 DO에게 전송 — 전역 정규화 파라미터 + 그리드 레이아웃"""
    global_min:   float
    global_max:   float
    scale_factor: float
    eps_norm:     float       # ε / scale_factor
    grid_shape:   Tuple       # (cols, rows) for 2D; (cols, rows, depths) for 3D
    G_total:      int         # 전체 그리드 셀 수
    dim:          int


# ─────────────────────────────────────────────────────────────────
# Phase 0 Round 2: DO → FC
# ─────────────────────────────────────────────────────────────────
@dataclass
class Msg_DO_FC_R2:
    """각 DO가 FC에게 전송 — 그리드 내 최대 점 수 (밀도 상한값)"""
    do_id: int
    n_do:  int    # 이 DO가 임의의 그리드 셀에 가진 최대 점 수


# ─────────────────────────────────────────────────────────────────
# Phase 0 Final: FC → DO (최종 슬롯 할당)
# ─────────────────────────────────────────────────────────────────
@dataclass
class DOConfig:
    """FC → DO: 슬롯 할당 및 암호화에 필요한 모든 파라미터"""
    do_id:        int
    scale_factor: float
    global_min:   float
    eps_norm:     float
    grid_shape:   Tuple
    G_total:      int
    dim:          int
    n:            int         # 전체 DO 중 최대 n_do
    k:            int         # DO 수
    B:            int         # block size = n * k
    N_total:      int         # 전체 슬롯 수 = G_total * B
    slot_offset:  int         # 이 DO의 그리드 블록 내 시작 슬롯 = do_id * n


# ─────────────────────────────────────────────────────────────────
# Phase 0 Final: FC → Server
# ─────────────────────────────────────────────────────────────────
@dataclass
class ServerConfig:
    """FC → Server: FHE 연산 파라미터 일체"""
    N_total:     int
    B:           int
    G_total:     int
    grid_shape:  Tuple
    K_valid:     List[int]    # 유효 rotation offset 집합
    eps_norm_sq: float        # ε_norm² (거리 임계값)
    min_pts:     int
    dim:         int
    n:           int
    k:           int
    strides:     List[int]    # raster scan stride (grid_id 계산용)
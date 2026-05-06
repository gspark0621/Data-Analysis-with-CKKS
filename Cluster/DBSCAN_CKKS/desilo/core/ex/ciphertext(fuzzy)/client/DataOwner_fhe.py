"""
DataOwnerFHE
  받는 것: public_key, DOConfig
  하는 것: 데이터 패킹 → 암호화 → Server로 전달
"""
from desilofhe import Engine
from core.ciphertext.shared.keypack import KeyPack
from core.ciphertext.shared.Messages import DOConfig 
from core.ciphertext.shared.GridUtils import compute_grid_strides, assign_to_grids, build_global_vector
from typing import List, Any
import numpy as np


class DataOwnerFHE:
    def __init__(self, do_id: int, raw_data: np.ndarray):
        self.do_id    = do_id
        self.raw_data = raw_data
        self.config: DOConfig = None
        self.normalized: np.ndarray = None

    def receive_config(self, config: DOConfig):
        self.config = config
        self.normalized = np.clip(
            (self.raw_data - config.global_min) / config.scale_factor,
            0.0, 1.0 - 1e-9,
        )

    def encrypt_global_vector(self, keypack: KeyPack) -> List[Any]:
        """
        global_vecs 구성 후 차원별 암호화.
        Returns: List[Ciphertext] 길이 = dim
        """

        cfg     = self.config
        e       = keypack.engine
        strides = compute_grid_strides(cfg.grid_shape)
        grid_ids = assign_to_grids(self.normalized, cfg.eps_norm, cfg.grid_shape)

        global_vecs = build_global_vector(
            self.normalized, grid_ids, cfg, strides=strides
        )  # (dim, N_total) numpy — dummy=2.0

        return [
            e.encrypt(global_vecs[d].tolist(), keypack.public_key)
            for d in range(cfg.dim)
        ]
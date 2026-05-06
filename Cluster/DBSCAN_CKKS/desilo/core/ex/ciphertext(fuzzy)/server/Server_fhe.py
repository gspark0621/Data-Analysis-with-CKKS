"""
ServerFHE — FHE DBSCAN 서버.
plaintext server_main.py 의 모든 np. 연산을 fhe_ops 로 교체.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from time import time
from typing import Dict, List, Tuple, Any
from core.ciphertext.shared.Messages import ServerConfig
from core.ciphertext.shared.keypack import KeyPack
from core.ciphertext.server.Operation_fhe import (
    fhe_rotate, fhe_check_neighbor, fhe_valid_mask,
    fhe_fuzzy_neighbor_same_cell, fhe_fuzzy_neighbor_intra, fhe_core_mask, fhe_max,
)


class ServerFHE:
    def __init__(self, config: ServerConfig, keypack: KeyPack):
        self.cfg = config
        self.kp  = keypack
        self._do_data: Dict[int, List[Any]] = {}  # {do_id: [ct_d0, ct_d1, ...]}

    def receive_do_data(self, do_id: int, encrypted_vecs: List[Any]) -> None:
        self._do_data[do_id] = encrypted_vecs

    def combine_do_data(self) -> List[Any]:
        """차원별 FHE add: combined[d] = Σ_i ct_DOi[d]"""
        e = self.kp.engine
        combined = list(self._do_data[0])          # deep copy 첫 DO
        for do_id, vecs in self._do_data.items():
            if do_id == 0:
                continue
            for d in range(self.cfg.dim):
                combined[d] = e.add(combined[d], vecs[d])
        return combined   # List[Ciphertext] len=dim

    def _split_k_valid(self):
        """
        K_valid를 같은 cell 내부 / 인접 cell 오프셋으로 분리.

        compute_K_valid() 구현상:
        같은 cell (delta_grid=0): k in {1, 2, ..., B-1}
        인접 cell (delta_grid!=0): 나머지 k
        """
        B = self.cfg.B
        same_cell_ks = set(range(1, B))
        adj_cell_ks  = set(self.cfg.K_valid) - same_cell_ks
        return same_cell_ks, adj_cell_ks

    def run_dbscan(self) -> Tuple[Any, Dict[str, float]]:
        """
        Returns
        -------
        ct_final      : Ciphertext — 정규화 라벨
        step_timings  : 단계별 소요 시간
        """
        cfg = self.cfg
        kp  = self.kp
        e   = kp.engine

        combined = self.combine_do_data()   # List[Ciphertext] len=dim

        # ── Step 1: 이웃 판별 ────────────────────────────────────
        print(f"[ServerFHE] Step 1: 이웃 판별 ({len(cfg.K_valid)}개 rotation)")
        t1 = time()

        same_cell_ks, adj_cell_ks = self._split_k_valid()

        adj_k_list: List[Any] = []           # crisp 0/1 — label propagation 전용
        total_neighbors = e.encrypt([0.0] * cfg.N_total, kp.public_key)

        for k in cfg.K_valid:
            # 차원별 dist^2 계산 (공통)
            dist_sq = None
            for d in range(cfg.dim):
                rotated = fhe_rotate(e, combined[d], kp.rotation_key, k)
                diff    = e.subtract(combined[d], rotated)
                sq      = e.multiply(diff, diff, kp.relinearization_key)
                dist_sq = sq if dist_sq is None else e.add(dist_sq, sq)

            if k in same_cell_ks:
                # 같은 cell 내 점 쌍
                # cell 한 변 = eps/sqrt(d) 조건 하에서 무조건 d(x,y) <= eps 보장
                # -> sign_bootstrap 불필요
                fuzzy_k = fhe_fuzzy_neighbor_same_cell(e, dist_sq, cfg.eps_norm_sq)
                adj_k   = e.encrypt([1.0] * cfg.N_total, kp.public_key)  # 항상 이웃

            else:
                # 인접 cell 점 쌍
                # crisp 판별 (sign_bootstrap 1회)
                adj_k   = fhe_check_neighbor(e, dist_sq, cfg.eps_norm_sq, kp)
                # adj_k를 clamp로 재활용 -> 추가 sign_bootstrap 불필요
                fuzzy_k = fhe_fuzzy_neighbor_intra(e, dist_sq, cfg.eps_norm_sq, adj_k, kp)

            adj_k_list.append(adj_k)
            total_neighbors = e.add(total_neighbors, fuzzy_k)

        # self-loop 보정: 자기 자신의 fuzzy 기여 = 1 (기존과 동일)
        total_neighbors = e.add(total_neighbors, 1.0)

        # dummy 슬롯 제거
        valid_mask      = fhe_valid_mask(e, combined[0], kp)          # sign_bootstrap
        total_neighbors = e.multiply(total_neighbors, valid_mask, kp.relinearization_key)

        t1_sec = time() - t1
        print(f"[ServerFHE] Step 1 완료: {t1_sec:.2f}초")

        # ── Step 2: Core point 판별 ──────────────────────────────
        print(f"[ServerFHE] Step 2: Core 판별 (min_pts={cfg.min_pts})")
        t2 = time()

        N_real    = cfg.n * cfg.k * (3 ** cfg.dim)
        core_mask = fhe_core_mask(e, total_neighbors, cfg.min_pts, N_real, kp)  # sign_bootstrap

        t2_sec = time() - t2
        print(f"[ServerFHE] Step 2 완료: {t2_sec:.2f}초")

        # ── Step 3: 라벨 전파 ────────────────────────────────────
        print(f"[ServerFHE] Step 3: 라벨 전파")
        t3 = time()

        cluster_id_vals = [(i + 1) / float(cfg.N_total + 1) for i in range(cfg.N_total)]
        cluster_id = e.encrypt(cluster_id_vals, kp.public_key)

        ct_final = self._propagate_labels_fhe(
            adj_k_list, core_mask, cluster_id, max_iter=5
        )

        t3_sec = time() - t3
        print(f"[ServerFHE] Step 3 완료: {t3_sec:.2f}초")

        # 스케일 복원: × (N_total + 1)
        ct_final = e.multiply(ct_final, float(cfg.N_total + 1))

        step_timings = {
            "step1_neighbor_sec":    t1_sec,
            "step2_core_sec":        t2_sec,
            "step3_propagation_sec": t3_sec,
        }
        return ct_final, step_timings

    def _propagate_labels_fhe(self,
                               adj_k_list: List[Any],
                               core_mask: Any,
                               cluster_id: Any,
                               max_iter: int = 5) -> Any:
        """
        Core-Core 전파 + Border 할당 (FHE 버전).
        Numpy 대응: propagate_labels()
        """
        cfg = self.cfg
        kp  = self.kp
        e   = kp.engine

        non_core_mask = e.add(e.multiply(core_mask, -1.0), 1.0)  # 1 - core_mask
        core_labels   = e.multiply(core_mask, cluster_id, kp.relinearization_key)

        # ── Core-Core 전파 ────────────────────────────────────────
        for it in range(max_iter):
            for adj_k, k in zip(adj_k_list, cfg.K_valid):
                shifted_core_labels = fhe_rotate(e, core_labels, kp.rotation_key, k)
                shifted_core_mask   = fhe_rotate(e, core_mask,   kp.rotation_key, k)

                # edge_mask = adj_k * core_mask * shifted_core_mask
                edge_mask  = e.multiply(adj_k, core_mask, kp.relinearization_key)
                edge_mask  = e.multiply(edge_mask, shifted_core_mask, kp.relinearization_key)

                # candidate = edge_mask * shifted_core_labels
                candidate  = e.multiply(edge_mask, shifted_core_labels, kp.relinearization_key)

                # core_labels = max(core_labels, candidate) * core_mask
                core_labels = fhe_max(e, core_labels, candidate, kp)   # sign_bootstrap
                core_labels = e.multiply(core_labels, core_mask, kp.relinearization_key)

        # ── Border 할당 ───────────────────────────────────────────
        border_labels = e.encrypt([0.0] * cfg.N_total, kp.public_key)
        assigned_mask = e.encrypt([0.0] * cfg.N_total, kp.public_key)

        for adj_k, k in zip(adj_k_list, cfg.K_valid):
            shifted_core_labels = fhe_rotate(e, core_labels, kp.rotation_key, k)
            shifted_core_mask   = fhe_rotate(e, core_mask,   kp.rotation_key, k)

            # cand_mask = adj_k * shifted_core_mask * non_core_mask
            cand_mask   = e.multiply(adj_k, shifted_core_mask, kp.relinearization_key)
            cand_mask   = e.multiply(cand_mask, non_core_mask, kp.relinearization_key)

            # empty_mask = 1 - min(assigned_mask, 1) ≈ 1 - assigned_mask (0/1이므로)
            empty_mask  = e.add(e.multiply(assigned_mask, -1.0), 1.0)

            # accept_mask = min(cand_mask * empty_mask, 1) ≈ cand_mask * empty_mask (0/1)
            accept_mask = e.multiply(cand_mask, empty_mask, kp.relinearization_key)

            border_labels = e.add(
                border_labels,
                e.multiply(accept_mask, shifted_core_labels, kp.relinearization_key)
            )
            assigned_mask = fhe_max(e, assigned_mask, accept_mask, kp)  # sign_bootstrap

        return e.add(core_labels, border_labels)
"""
main.py — Phase 0~4 오케스트레이션
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from core.plaintext.client.FinalClient import FinalClient
from core.plaintext.client.MultiPartyDataOwner import DataOwner
from core.plaintext.server.ServerMain import Server


def generate_cluster_data(n_pts: int, dim: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    centers = np.random.uniform(1, 9, (3, dim))
    pts = [np.random.randn(n_pts // 3, dim) * 0.4 + c for c in centers]
    return np.vstack(pts)


def print_phase(n: int, title: str):
    print(f"\n{'='*62}\n  Phase {n}: {title}\n{'='*62}")


# ── Phase 0: Pre-negotiation ─────────────────────────────────────
def phase0(do_list, fc, dim):
    print_phase(0, "Pre-negotiation (DO ↔ FC)")

    # Round 1: DO → FC
    for do in do_list:
        fc.receive_round1(do.compute_round1_msg())

    r1_resp = fc.process_round1(dim=dim)
    print(f"  scale={r1_resp.scale_factor:.3f}, eps_norm={r1_resp.eps_norm:.4f}, "
          f"grid={r1_resp.grid_shape}, G_total={r1_resp.G_total}")
    for do in do_list:
        do.receive_round1_response(r1_resp)

    # Round 2: DO → FC
    for do in do_list:
        fc.receive_round2(do.compute_round2_msg())

    gp = fc.process_round2()
    print(f"  n={gp['n']}, k={gp['k']}, B={gp['B']}, "
          f"N_total={gp['N_total']}, |K_valid|={len(gp['K_valid'])}")

    for do in do_list:
        do.receive_final_config(fc.get_do_config(do.do_id))


# ── Phase 1: Key Generation (FHE TODO) ──────────────────────────
def phase1(fc, do_list):
    print_phase(1, "Key Generation (Numpy: skip)")
    fc.generate_keys()


# ── Phase 2: Data Packing & Encryption ──────────────────────────
def phase2(do_list, server):
    print_phase(2, "Data Encryption & Packing (DO → Server)")
    for do in do_list:
        vecs = do.pack_global_vector()
        server.receive_do_data(do.do_id, do.encrypt_global_vector(vecs))
        print(f"  DO_{do.do_id}: 패킹 완료 (shape={vecs.shape})")


# ── Phase 3: Server FHE-DBSCAN ───────────────────────────────────
def phase3(server):
    print_phase(3, "Server FHE-DBSCAN")
    return server.run_dbscan()


# ── Phase 4: Decryption ──────────────────────────────────────────
def phase4(fc, encrypted_result, do_list):
    print_phase(4, "Decryption (FC)")
    cfg = fc.get_do_config(0)
    do_slot_map = {}
    for do in do_list:
        slots = []
        for g_id, cnt in do.get_grid_point_count().items():
            base = g_id * cfg.B + cfg.slot_offset
            slots.extend(range(base, base + cnt))
        do_slot_map[do.do_id] = slots

    N_real = sum(do.N_pts for do in do_list)
    result = fc.decrypt_result(encrypted_result, N_real, do_slot_map)

    for do_id, labels in result.items():
        n_clusters = len(set(l for l in labels if l > 0))
        print(f"  DO_{do_id}: 점={len(labels)}, 클러스터={n_clusters}, "
              f"noise={labels.count(-1)}")
    return result


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    EPS, MIN_PTS, DIM = 1.0, 3, 2
    do_list = [DataOwner(i, generate_cluster_data(90, DIM, 42 + i * 7))
               for i in range(3)]
    fc = FinalClient(eps=EPS, min_pts=MIN_PTS)

    phase0(do_list, fc, DIM)
    phase1(fc, do_list)

    server = Server(fc.get_server_config())
    phase2(do_list, server)
    encrypted_result = phase3(server)
    phase4(fc, encrypted_result, do_list)
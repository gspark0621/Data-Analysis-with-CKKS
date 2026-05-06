"""
FHE Multiparty DBSCAN 전체 파이프라인 테스트.
"""
import sys, os
import numpy as np
from time import time

# ── plaintext 재사용 ──────────────────────────────────────────────
from core.plaintext.client.FinalClient import FinalClient
from core.plaintext.client.MultiPartyDataOwner import DataOwner
from core.plaintext.shared.GridUtils import compute_grid_strides

# ── FHE 전용 ─────────────────────────────────────────────────────
from core.ciphertext.client.FinalClient_fhe import FinalClientFHE
from core.ciphertext.client.DataOwner_fhe import DataOwnerFHE
from core.ciphertext.server.Server_fhe import ServerFHE
from core.ciphertext.shared.Messages import ServerConfig

DATASET_PATH = (
    "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS"
    "/desilo/dataset/Other_cluster/hepta.arff"
)


def load_arff(path):
    from scipy.io import arff
    data, meta = arff.loadarff(path)
    X = np.array(
        [[row[i] for i in range(len(meta.names()) - 1)] for row in data],
        dtype=float,
    )
    y = np.array([int(row[-1]) for row in data])
    return X, y


def remap_labels(labels_raw: np.ndarray, N_total: int) -> np.ndarray:
    """
    복호화된 정규화 라벨 → 순차 정수 클러스터 ID.
    0 이하 → noise(-1)
    """
    raw = np.round(labels_raw[:N_total]).astype(int)
    mapping = {}
    out = np.zeros(N_total, dtype=int)
    next_id = 1
    for i, v in enumerate(raw):
        if v <= 0:
            out[i] = -1
        else:
            if v not in mapping:
                mapping[v] = next_id
                next_id += 1
            out[i] = mapping[v]
    return out


def main():
    eps     = float(input("eps 값을 입력하세요 (예: 1) > "))
    min_pts = int(float(input("min_pts 값을 입력하세요 (예: 4) > ")))
    n_dos   = int(input("DO 수를 입력하세요 (예: 3) > "))

    X, y_true = load_arff(DATASET_PATH)
    N, dim    = X.shape
    chunks    = np.array_split(np.random.permutation(N), n_dos)

    # ── Phase 0: Pre-negotiation ──────────────────────────────────
    print("\n=== Phase 0: Pre-negotiation ===")
    t0 = time()

    fc_plain  = FinalClient(eps=eps, min_pts=min_pts)
    dos_plain = [DataOwner(i, X[chunks[i]]) for i in range(n_dos)]

    for do in dos_plain:
        fc_plain.receive_round1(do.compute_round1_msg())
    r1_resp = fc_plain.process_round1(dim)
    for do in dos_plain:
        do.receive_round1_response(r1_resp)
        fc_plain.receive_round2(do.compute_round2_msg())
    grid_params = fc_plain.process_round2()

    t_phase0 = time() - t0
    print(f"  완료: {t_phase0:.4f}초")
    print(f"  n={grid_params['n']}, k={n_dos}, B={grid_params['B']}, "
          f"N_total={grid_params['N_total']}, |K_valid|={len(grid_params['K_valid'])}")

    # ── ServerConfig 구성 ─────────────────────────────────────────
    server_config = ServerConfig(
        N_total     = grid_params['N_total'],
        B           = grid_params['B'],
        G_total     = grid_params['G_total'],
        grid_shape  = grid_params['grid_shape'],
        K_valid     = grid_params['K_valid'],
        eps_norm_sq = grid_params['eps_norm'] ** 2,
        min_pts     = min_pts,
        dim         = dim,
        n           = grid_params['n'],
        k           = n_dos,
        strides     = compute_grid_strides(grid_params['grid_shape']),
    )

    # ── Phase 1: Key Generation ───────────────────────────────────
    print("\n=== Phase 1: Key Generation ===")
    t1 = time()

    fc_fhe  = FinalClientFHE()
    keypack = fc_fhe.generate_keys()   # FC 혼자 키 생성

    t_phase1 = time() - t1
    print(f"  완료: {t_phase1:.2f}초")

    # ── Phase 2: Data Packing & Encryption ───────────────────────
    print("\n=== Phase 2: Data Packing ===")
    t2 = time()

    server  = ServerFHE(server_config, keypack)          # server 선언
    dos_fhe = [DataOwnerFHE(i, X[chunks[i]]) for i in range(n_dos)]

    for i, do_fhe in enumerate(dos_fhe):
        do_fhe.receive_config(fc_plain.get_do_config(i)) # config 1회만 전달
        enc_vecs = do_fhe.encrypt_global_vector(keypack) # keypack 통째로
        server.receive_do_data(do_fhe.do_id, enc_vecs)
        print(f"  DO_{i}: 암호화 완료")

    t_phase2 = time() - t2
    print(f"  완료: {t_phase2:.2f}초")

    # ── Phase 3: FHE DBSCAN ──────────────────────────────────────
    print("\n=== Phase 3: FHE DBSCAN ===")
    ct_final, step_timings = server.run_dbscan()
    step_timings["phase3_total_sec"] = sum(step_timings.values())

    # ── Phase 4: Decryption ───────────────────────────────────────
    print("\n=== Phase 4: Decryption ===")
    t4 = time()

    labels_raw = fc_fhe.decrypt(ct_final)               # FC secret_key로 직접 복호화
    labels_all = remap_labels(labels_raw, server_config.N_total)

    # 각 점의 slot 위치에서 라벨 추출
    result_labels = np.full(N, -1, dtype=int)
    for i, do_fhe in enumerate(dos_fhe):
        cfg = do_fhe.config
        for local_idx in range(len(do_fhe.normalized)):
            from core.ciphertext.shared.GridUtils import assign_to_grids
            grid_ids = assign_to_grids(
                do_fhe.normalized, cfg.eps_norm, cfg.grid_shape
            )
            g    = int(grid_ids[local_idx])
            slot = g * cfg.B + cfg.slot_offset + local_idx
            result_labels[chunks[i][local_idx]] = labels_all[slot]

    t_phase4 = time() - t4
    print(f"  완료: {t_phase4:.2f}초")

    # ── ARI 평가 ─────────────────────────────────────────────────
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y_true, result_labels) * 100
    n_clusters = len(set(result_labels) - {-1})
    n_noise    = int((result_labels == -1).sum())
    print(f"\n  클러스터 수={n_clusters}, noise={n_noise}/{N}")
    print(f"  ARI: {ari:.2f} / 100점")

    # ── 타이밍 출력 ──────────────────────────────────────────────
    timings = {
        "phase0_sec":                     t_phase0,
        "phase1_keygen_sec":              t_phase1,
        "phase2_encrypt_sec":             t_phase2,
        "phase3_step1_neighbor_sec":      step_timings["step1_neighbor_sec"],
        "phase3_step2_core_sec":          step_timings["step2_core_sec"],
        "phase3_step3_propagation_sec":   step_timings["step3_propagation_sec"],
        "phase3_total_sec":               step_timings["phase3_total_sec"],
        "phase4_decrypt_sec":             t_phase4,
    }
    print("\n타이밍")
    for k, v in timings.items():
        print(f"  {k:<40s}: {v:.4f}초")


if __name__ == "__main__":
    main()
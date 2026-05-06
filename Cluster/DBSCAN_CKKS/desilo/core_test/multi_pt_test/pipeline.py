"""
test_pipeline.py
Grid-based Multiparty Plaintext DBSCAN 전체 파이프라인 테스트.
"""
from http import server
import os, sys

from core.plaintext.main import print_phase
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from time import time
from typing import List, Dict, Tuple
from sklearn.metrics import adjusted_rand_score

from core.plaintext.client.FinalClient import FinalClient
from core.plaintext.client.MultiPartyDataOwner import DataOwner
from core.plaintext.server.ServerMain import Server

DATASET_PATH = (
    "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/lsun.arff"
)

# ─────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────

def load_arff_to_pts_with_labels(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    pts, true_labels = [], []
    data_section = False
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@data"):
                data_section = True
                continue
            if data_section:
                line = line.replace("\t", " ").replace(",", " ")
                values = line.split()
                if len(values) < 2:
                    continue
                pts.append([float(v) for v in values[:-1]])
                true_labels.append(int(float(values[-1])))
    if not pts:
        raise ValueError("데이터를 찾을 수 없습니다.")
    return np.array(pts, dtype=np.float64), np.array(true_labels, dtype=int)


def remap_labels_to_sequential(labels: List[int]) -> List[int]:
    unique_valid = sorted(x for x in set(labels) if x != -1)
    mapping = {old: idx + 1 for idx, old in enumerate(unique_valid)}
    return [-1 if x == -1 else mapping[x] for x in labels]


def save_timings_txt(filename: str, timings_dict: dict) -> None:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for k, v in timings_dict.items():
                f.write(f"{k}: {float(v):.6f}\n")
        print(f"✅ 타이밍 파일 저장 완료: {filename}")
    except Exception as e:
        print(f"❌ 타이밍 파일 저장 실패: {e}")


def split_data_for_owners(pts: np.ndarray, n_dos: int) -> List[np.ndarray]:
    """데이터를 n_dos개 DO에 균등 분배."""
    chunk = len(pts) // n_dos
    return [
        pts[i * chunk : (i + 1) * chunk if i < n_dos - 1 else len(pts)]
        for i in range(n_dos)
    ]


# ─────────────────────────────────────────────────────────────────
# Phase 래퍼
# ─────────────────────────────────────────────────────────────────

def run_phase0(do_list: List[DataOwner], fc: FinalClient, dim: int) -> None:
    for do in do_list:
        fc.receive_round1(do.compute_round1_msg())

    r1_resp = fc.process_round1(dim=dim)
    print(f"  scale={r1_resp.scale_factor:.4f}, eps_norm={r1_resp.eps_norm:.6f}, "
          f"grid={r1_resp.grid_shape}, G_total={r1_resp.G_total}")
    for do in do_list:
        do.receive_round1_response(r1_resp)

    for do in do_list:
        msg = do.compute_round2_msg()
        fc.receive_round2(msg)
        print(f"    DO_{do.do_id}: n_do={msg.n_do}")

    gp = fc.process_round2()
    slot_ok = "✓" if gp["N_total"] <= 32768 else "✗ slot_count 초과!"
    print(f"  n={gp['n']}, k={gp['k']}, B={gp['B']}, "
          f"N_total={gp['N_total']} {slot_ok}, |K_valid|={len(gp['K_valid'])}")
    for do in do_list:
        do.receive_final_config(fc.get_do_config(do.do_id))


def run_phase2(do_list: List[DataOwner], server: Server) -> None:
    for do in do_list:
        vecs = do.pack_global_vector()
        server.receive_do_data(do.do_id, do.encrypt_global_vector(vecs))
        real_slots = int((vecs[0] > 0).sum() - (vecs[0] >= 2.0).sum())
        print(f"  DO_{do.do_id}: 유효 슬롯={real_slots}/{vecs.shape[1]}")

def run_phase3(server: Server) -> Tuple[np.ndarray, Dict[str, float]]:
    print_phase(3, "Server FHE-DBSCAN")
    t3 = time()
    final_labels, step_timings = server.run_dbscan()  # ← 튜플 언패킹
    step_timings["phase3_total_sec"] = time() - t3    # ← phase3 전체도 기록
    return final_labels, step_timings

def extract_labels_from_result(
    final_raw: np.ndarray,
    do_list: List[DataOwner],
    N_total: int,
) -> Tuple[List[int], List[Dict]]:
    """
    point_slot_map 기반으로 원본 점 순서대로 라벨 추출.
    slot이 없는 점(n 초과 누락) → noise(-1) 처리.
    """
    all_labels, debug_rows = [], []
    global_idx = 0
    for do in sorted(do_list, key=lambda d: d.do_id):
        for local_idx in range(do.N_pts):
            slot = do.point_slot_map.get(local_idx)
            label_raw = float(final_raw[slot]) if slot is not None else 0.0
            label_int = round(label_raw)
            label_out = -1 if label_int <= 0 else label_int
            all_labels.append(label_out)
            debug_rows.append({
                "global_idx": global_idx, "do_id": do.do_id,
                "local_idx": local_idx,   "slot": slot,
                "label_raw": label_raw,   "label": label_out,
            })
            global_idx += 1
    return all_labels, debug_rows


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Multiparty Plaintext Grid-DBSCAN 파이프라인 테스트")
    print("=" * 58)
    print(f"\n▶ 데이터셋: {DATASET_PATH}\n")

    eps_val     = float(input("eps 값을 입력하세요 (예: 0.5) > "))
    min_pts_val = int(float(input("min_pts 값을 입력하세요 (예: 3) > ")))
    n_dos       = int(input("DO 수를 입력하세요 (예: 3) > "))

    print(f"\n  eps={eps_val}, min_pts={min_pts_val}, n_dos={n_dos}")

    print("\n데이터 로딩 중...")
    pts, true_labels = load_arff_to_pts_with_labels(DATASET_PATH)
    N, DIM = pts.shape
    print(f"  완료: {N}개 점, {DIM}차원")

    splits  = split_data_for_owners(pts, n_dos)
    do_list = [DataOwner(i, splits[i]) for i in range(n_dos)]
    for do in do_list:
        print(f"  DO_{do.do_id}: {do.N_pts}개 점")

    fc = FinalClient(eps=eps_val, min_pts=min_pts_val)
    total_start = time()

    # Phase 0
    print("\n" + "=" * 58 + "\n  Phase 0: Pre-negotiation\n" + "=" * 58)
    t0 = time(); run_phase0(do_list, fc, DIM); t_phase0 = time() - t0

    # Phase 1 (Numpy: skip)
    print("\n" + "=" * 58 + "\n  Phase 1: Key Generation (Numpy: skip)\n" + "=" * 58)
    fc.generate_keys()

    server = Server(fc.get_server_config())

    # Phase 2
    print("\n" + "=" * 58 + "\n  Phase 2: Data Packing (DO → Server)\n" + "=" * 58)
    t2 = time(); run_phase2(do_list, server); t_phase2 = time() - t2

    # Phase 3
    print("\n" + "=" * 58 + "\n  Phase 3: Server DBSCAN\n" + "=" * 58)
    t3 = time()
    final_raw, step_timings = run_phase3(server)              # ← step_timings 수신
    t_phase3 = step_timings["phase3_total_sec"]

    # Phase 4
    print("\n" + "=" * 58 + "\n  Phase 4: Decryption & Label Extraction\n" + "=" * 58)
    t4 = time()
    all_labels_raw, debug_rows = extract_labels_from_result(
        final_raw, do_list, fc.get_server_config().N_total
    )
    all_labels = remap_labels_to_sequential(all_labels_raw)
    t_phase4 = time() - t4
    t_total  = time() - total_start

    # 결과 샘플
    print("\n결과 샘플 (상위 20개)")
    for row in debug_rows[:min(20, len(debug_rows))]:
        print(f"  gid={row['global_idx']:4d} | do={row['do_id']} | "
              f"local={row['local_idx']:4d} | slot={str(row['slot']):6s} | "
              f"label_raw={row['label_raw']:8.2f} | label={row['label']}")

    # 검증
    true_labels_seq = remap_labels_to_sequential(true_labels.tolist())
    n_clusters = len([c for c in set(all_labels) if c != -1])
    n_noise    = all_labels.count(-1)
    ari_score  = adjusted_rand_score(true_labels_seq, all_labels)

    print(f"\n  클러스터 수={n_clusters}, noise={n_noise}/{N}")
    print(f"\n📊 ARI: {ari_score * 100:.2f} / 100점")
    if ari_score > 0.99:
        print("  => 🎉 대성공!")
    elif ari_score > 0.80:
        print("  => 👍 대부분 동일, 경계 일부 차이")
    else:
        print("  => ❌ eps/min_pts 재점검 필요")

    # 타이밍
    timings = {
        "phase0_sec":                t_phase0,
        "phase2_sec":                t_phase2,
        # ── Phase 3 세부 타이밍 ──────────────────────────────
        "phase3_step1_neighbor_sec": step_timings["step1_neighbor_sec"],
        "phase3_step2_core_sec":     step_timings["step2_core_sec"],
        "phase3_step3_propagation_sec": step_timings["step3_propagation_sec"],
        "phase3_total_sec":          t_phase3,
        # ────────────────────────────────────────────────────
        "phase4_sec":                t_phase4,
        "total_sec":                 t_total,
    }
    print("\n타이밍")
    for k, v in timings.items():
        print(f"  {k:<35s}: {v:.4f}초")
    save_timings_txt(
        f"timings_eps{eps_val}_min{min_pts_val}_dos{n_dos}.txt", timings
    )

    # # 디버그 CSV
    # debug_file = f"debug_eps{eps_val}_min{min_pts_val}_dos{n_dos}.csv"
    # try:
    #     with open(debug_file, "w", encoding="utf-8") as f:
    #         f.write("Point_ID,DO_ID,Local_IDX,Global_Slot,"
    #                 "Label_Raw,Pred_Label,True_Label\n")
    #         for i, row in enumerate(debug_rows):
    #             f.write(f"{row['global_idx']},{row['do_id']},{row['local_idx']},"
    #                     f"{row['slot']},{row['label_raw']:.4f},"
    #                     f"{all_labels[i]},{true_labels_seq[i]}\n")
    #     print(f"✅ 디버그 CSV: {debug_file}")
    # except Exception as e:
    #     print(f"❌ 디버그 CSV 저장 실패: {e}")

    # # 결과 CSV
    # result_file = f"result_eps{eps_val}_min{min_pts_val}_dos{n_dos}.csv"
    # try:
    #     with open(result_file, "w", encoding="utf-8") as f:
    #         f.write(",".join([f"x{i+1}" for i in range(DIM)]) +
    #                 ",Pred_Cluster,True_Class\n")
    #         for i, row in enumerate(debug_rows):
    #             do  = do_list[row["do_id"]]
    #             pt  = do.raw_data[row["local_idx"]]
    #             coords = ",".join([f"{v:.6f}" for v in pt])
    #             f.write(f"{coords},{all_labels[i]},{true_labels[i]}\n")
    #     print(f"✅ 결과 CSV: {result_file}")
    # except Exception as e:
    #     print(f"❌ 결과 CSV 저장 실패: {e}")


if __name__ == "__main__":
    main()
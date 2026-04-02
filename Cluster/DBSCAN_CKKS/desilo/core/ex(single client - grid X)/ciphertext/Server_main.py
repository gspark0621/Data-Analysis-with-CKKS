import os
from time import time
import desilofhe
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
import numpy as np
from core.Normalize import check_neighbor_closed_interval
from core.Core import identify_core_points_fhe_converted as identify_core_points_fhe
from core.Label_Propagation import fhe_max_propagation_fhe, fhe_circular_shift


def save_vector_csv(filename, values, header):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(header + "\n")
            for i, v in enumerate(values):
                f.write(f"{i},{float(v):.6f}\n")
        print(f"✅ 저장 완료: {filename}")
    except Exception as e:
        print(f"❌ 저장 실패 ({filename}): {e}")


def send_to_server_fhe(
    engine, keypack, secret_key,
    encrypted_columns, num_points, eps, min_pts
):
    dim = len(encrypted_columns)
    N = num_points
    adj_k_list = []
    total_neighbors_ct = None

    debug_fhe = {}
    timings = {}

    print(f"\n[DEBUG] 1. 이웃 판별(Normalize) 단계 시작... eps^2 = {eps**2:.4f}")
    normalize_start = time()

    for k in range(1, N):
        dist_sq_k = None
        for d in range(dim):
            base_col = encrypted_columns[d]
            rotated_col = fhe_circular_shift(engine, base_col, k, N, keypack)
            diff_ct = engine.subtract(base_col, rotated_col)
            sq_ct = engine.square(diff_ct, keypack.relinearization_key)

            if dist_sq_k is None:
                dist_sq_k = sq_ct
            else:
                dist_sq_k = engine.add(dist_sq_k, sq_ct)

        adj_k = check_neighbor_closed_interval(engine, dist_sq_k, eps**2, keypack, dim)
        adj_k_list.append(adj_k)

        if total_neighbors_ct is None:
            total_neighbors_ct = adj_k
        else:
            total_neighbors_ct = engine.add(total_neighbors_ct, adj_k)

        if k == N - 1:
            dec_adj = engine.decrypt(adj_k, secret_key)
            valid_adj = dec_adj[:N]
            print(f"  -> k={k} 일 때 이웃 배열 (하위 10개): {np.round(valid_adj[-10:], 4)}")

    ones_plaintext = engine.encode([1.0 for _ in range(N)])
    total_neighbors_ct = engine.add(total_neighbors_ct, ones_plaintext)

    timings["normalize_sec"] = time() - normalize_start
    print(f"[TIME] Normalize 단계 소요 시간: {timings['normalize_sec']:.2f}초")

    dec_total = engine.decrypt(total_neighbors_ct, secret_key)[:N]
    debug_fhe["total_neighbors"] = np.array(dec_total)

    print(f"\n[DEBUG] 2. 총 이웃 수 누적 결과 (상위 10개): {np.round(dec_total[:10], 2)}")
    print(f"  -> 코어 포인트 조건 (min_pts) : {min_pts}")

    save_vector_csv(
        filename=f"debug_normalize_eps{eps:.4f}_min{int(min_pts)}.csv",
        values=dec_total,
        header="Point_ID,Total_Neighbors"
    )

    core_start = time()
    core_ct = identify_core_points_fhe(engine, total_neighbors_ct, min_pts, N, keypack=keypack)
    timings["core_sec"] = time() - core_start
    print(f"[TIME] Core 단계 소요 시간: {timings['core_sec']:.2f}초")

    dec_core = engine.decrypt(core_ct, secret_key)[:N]
    debug_fhe["core_mask"] = np.array(dec_core)

    print(f"\n[DEBUG] 3. Core 판별 마스크 결과 (상위 10개): {np.round(dec_core[:10], 4)}")

    if np.max(dec_core[:N]) < 0.5:
        print("  ❌ [FATAL] 코어 포인트가 단 한 개도 검출되지 않았습니다! Core.py의 스케일링/마진을 점검하세요.")

    save_vector_csv(
        filename=f"debug_core_eps{eps:.4f}_min{int(min_pts)}.csv",
        values=dec_core,
        header="Point_ID,Core_Mask"
    )

    print("\n[DEBUG] 4. Label Propagation 전파 시작...")
    cluster_id_pt = [(i + 1) / float(N + 1) for i in range(N)]

    lp_start = time()
    final_norm_ct = fhe_max_propagation_fhe(
        engine, keypack, adj_k_list, core_ct, cluster_id_pt, N,
        max_iter=5,
    )
    timings["label_propagation_sec"] = time() - lp_start
    print(f"[TIME] Label_Propagation 단계 소요 시간: {timings['label_propagation_sec']:.2f}초")

    dec_final_norm = engine.decrypt(final_norm_ct, secret_key)[:N]
    debug_fhe["final_norm_labels"] = np.array(dec_final_norm)

    print(f"\n[DEBUG] 5. 전파 완료 후 정규화 라벨 결과 (상위 10개): {np.round(dec_final_norm[:10], 4)}")

    save_vector_csv(
        filename=f"debug_labelprop_norm_eps{eps:.4f}_min{int(min_pts)}.csv",
        values=dec_final_norm,
        header="Point_ID,Final_Norm_Label"
    )

    post_start = time()
    scale_back_pt = engine.encode([float(N + 1) for _ in range(N)])
    final_ct = engine.multiply(final_norm_ct, scale_back_pt)

    final_ct = engine.bootstrap(
        engine.intt(final_ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key
    )
    timings["postprocess_sec"] = time() - post_start
    print(f"[TIME] Final scale-back + bootstrap 소요 시간: {timings['postprocess_sec']:.2f}초")

    dec_final = engine.decrypt(final_ct, secret_key)[:N]
    debug_fhe["final_labels"] = np.array(dec_final)

    print(f"\n[DEBUG] 6. 최종 복원된 라벨 결과 (상위 10개): {np.round(dec_final[:10], 2)}")

    save_vector_csv(
        filename=f"debug_labelprop_final_eps{eps:.4f}_min{int(min_pts)}.csv",
        values=dec_final,
        header="Point_ID,Final_Label"
    )

    debug_fhe["timings"] = timings

    print("\n[DEBUG] 7. 디버깅용 중간 상태값 추출 완료")
    return final_ct, debug_fhe

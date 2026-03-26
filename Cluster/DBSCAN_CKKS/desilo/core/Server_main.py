import desilofhe
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
import numpy as np
from core.Normalize import check_neighbor_closed_interval
from core.Core import identify_core_points_fhe_converted as identify_core_points_fhe
from core.Label_Propagation import fhe_max_propagation_fhe, fhe_circular_shift


def send_to_server_fhe(
    engine, keypack, secret_key, # 🚨 디버깅을 위해 secret_key 추가
    encrypted_columns, num_points, eps, min_pts
):
    dim = len(encrypted_columns) # dim = 데이터 차원 수(encryted_columns = [(x차원), (y차원), ...] 형태로 들어옴)
    N = num_points
    adj_k_list = []
    total_neighbors_ct = None

    print(f"\n[DEBUG] 1. 이웃 판별(Normalize) 단계 시작... eps^2 = {eps**2:.4f}")
    
    for k in range(1, N):
        dist_sq_k = None
        for d in range(dim):
            base_col = encrypted_columns[d]
            # 순환 시프트
            rotated_col = fhe_circular_shift(engine, base_col, k, N, keypack)
            diff_ct = engine.subtract(base_col, rotated_col)
            sq_ct = engine.square(diff_ct, keypack.relinearization_key)

            if dist_sq_k is None:
                dist_sq_k = sq_ct
            else:
                dist_sq_k = engine.add(dist_sq_k, sq_ct)

        # Normalize 모듈
        adj_k = check_neighbor_closed_interval(engine, dist_sq_k, eps**2, keypack, dim)
        adj_k_list.append(adj_k)

        if total_neighbors_ct is None:
            total_neighbors_ct = adj_k
        else:
            total_neighbors_ct = engine.add(total_neighbors_ct, adj_k)

        # 🚨 [Probe 1] k = N - 1 일 때 이웃 여부가 0.0과 1.0으로 잘 나오는지 검사
        if k == N - 1:  # 마지막 k에 대해서 검사
            dec_adj = engine.decrypt(adj_k, secret_key)
            
            # 동형암호 쓰레기값(패딩)을 제외하고, 유효한 N개의 데이터만 먼저 추출
            valid_adj = dec_adj[:N] 
            
            # 유효 데이터의 하위(마지막) 10개 출력
            print(f"  -> k={k} 일 때 이웃 배열 (하위 10개): {np.round(valid_adj[-10:], 4)}")


    # 자기 자신 포함
    ones_plaintext = engine.encode([1.0 for _ in range(N)])
    total_neighbors_ct = engine.add(total_neighbors_ct, ones_plaintext)


    # 🚨 [Probe 2] 총 이웃 수(total_neighbors)가 정상적으로 합산되었는지 검사
    dec_total = engine.decrypt(total_neighbors_ct, secret_key)
    print(f"\n[DEBUG] 2. 총 이웃 수 누적 결과 (상위 10개): {np.round(dec_total[:10], 2)}")
    print(f"  -> 코어 포인트 조건 (min_pts) : {min_pts}")

    # Core 판별
    core_ct = identify_core_points_fhe(engine, total_neighbors_ct, min_pts, N, keypack=keypack)

    # 🚨 [Probe 3] 코어 포인트 판별이 1.0 (True) 과 0.0 (False) 으로 잘 나뉘었는지 검사
    dec_core = engine.decrypt(core_ct, secret_key)
    print(f"\n[DEBUG] 3. Core 판별 마스크 결과 (상위 10개): {np.round(dec_core[:10], 4)}")
    
    # 만약 코어 포인트가 모두 0이라면 여기서 더 이상 진행할 필요가 없습니다.
    if np.max(dec_core[:N]) < 0.5:
        print("  ❌ [FATAL] 코어 포인트가 단 한 개도 검출되지 않았습니다! Core.py의 스케일링/마진을 점검하세요.")

    print("\n[DEBUG] 4. Label Propagation 전파 시작...")
    cluster_id_pt = [(i + 1) / float(N + 1) for i in range(N)]

    # 🚨 라벨 전파 모듈 내부로 진입 (이곳에서 값이 뭉개진다면 max_iter를 조절해야 함)
    final_norm_ct = fhe_max_propagation_fhe(
        engine, keypack, adj_k_list, core_ct, cluster_id_pt, N, 
        max_iter = 5,
    )

    # 🚨 [Probe 4] 라벨 전파 직후의 정규화된 라벨 값 (0~1 사이여야 함)
    dec_final_norm = engine.decrypt(final_norm_ct, secret_key)
    print(f"\n[DEBUG] 5. 전파 완료 후 정규화 라벨 결과 (상위 10개): {np.round(dec_final_norm[:10], 4)}")

    scale_back_pt = engine.encode([float(N + 1) for _ in range(N)])
    final_ct = engine.multiply(final_norm_ct, scale_back_pt)

    final_ct = engine.bootstrap(engine.intt(final_ct), keypack.relinearization_key, keypack.conjugation_key, keypack.bootstrap_key)

    # 🚨 [Probe 5] 최종 팽창된 정수 라벨
    dec_final = engine.decrypt(final_ct, secret_key)
    print(f"\n[DEBUG] 6. 최종 복원된 라벨 결과 (상위 10개): {np.round(dec_final[:10], 2)}")


    print("\n[DEBUG] 7. 디버깅용 중간 상태값 추출 및 복호화 시작...")
    # 🚨 수정됨: 디버깅용 중간 상태값 복호화 및 추출
    debug_fhe = {}
    
    # 1. 누적된 이웃 수 복호화
    dec_total = engine.decrypt(total_neighbors_ct, secret_key)[:N]
    debug_fhe['total_neighbors'] = np.array(dec_total)
    
    # 2. 코어 포인트 마스크 복호화
    dec_core = engine.decrypt(core_ct, secret_key)[:N]
    debug_fhe['core_mask'] = np.array(dec_core)
    
    # 3. 최종 라벨 복호화
    dec_final = engine.decrypt(final_ct, secret_key)[:N]
    debug_fhe['final_labels'] = np.array(dec_final)

    return final_ct, debug_fhe # debug_fhe 추가 반환

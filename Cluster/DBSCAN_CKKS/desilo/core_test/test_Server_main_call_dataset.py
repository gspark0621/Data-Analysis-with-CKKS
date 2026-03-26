import os
import numpy as np
from time import time
from desilofhe import Engine
from util.keypack import KeyPack
from sklearn.metrics import adjusted_rand_score

# 실제 프로젝트 경로에 맞게 임포트 
from core.Server_main import send_to_server_fhe
from core.plaintext.Server_main import send_to_server_np
from core.EncryptModule import DimensionalEncryptor

# -------------------------------------------------------------------
# 1. 데이터 경로 및 로더
# -------------------------------------------------------------------
DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/hepta.arff"

def load_arff_to_pts_with_labels(filepath: str):
    """
    ARFF 파일에서 좌표 데이터와 원본 정답(Class)을 함께 읽어옵니다.
    반환값: (좌표 배열 pts, 원본 정답 배열 true_labels)
    """
    pts = []
    true_labels = []
    data_section = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'): continue
            if line.lower().startswith('@data'):
                data_section = True
                continue
            if data_section:
                line = line.replace('\t', ' ').replace(',', ' ')
                values = line.split()
                if len(values) < 2: continue
                # 마지막 요소는 정답 클래스, 나머지는 좌표
                row = [float(v) for v in values[:-1]]
                label = int(float(values[-1]))
                pts.append(row)
                true_labels.append(label)
                    
    if not pts:
        raise ValueError("데이터를 찾을 수 없습니다. 파일 포맷을 확인해주세요.")
    return np.array(pts, dtype=np.float64), np.array(true_labels, dtype=int)


def main():
    print("==================================================")
    print("      대화형 FHE DBSCAN 전체 데이터 E2E 테스트    ")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # 2. 사용자 입력 (대화형 파라미터 세팅)
    # ---------------------------------------------------------
    print(f"▶ 데이터셋 경로: {DATASET_PATH}\n")

    eps_val = float(input("eps 값을 입력하세요 (예: 0.5) > "))
    min_pts_val = float(input("min_pts 값을 입력하세요 (예: 3) > "))

    print("\n--------------------------------------------------")
    print(f"▶ 실행 파라미터: eps = {eps_val}, min_pts = {min_pts_val}")
    print("--------------------------------------------------\n")

    # ---------------------------------------------------------
    # 3. 데이터 전체 로드 및 정규화
    # ---------------------------------------------------------
    print("데이터 전체 로딩 및 정규화 중...")
    
    # 원본 클래스 라벨도 함께 불러옵니다
    pts, true_labels = load_arff_to_pts_with_labels(DATASET_PATH)
    N = len(pts)
    dimension = pts.shape[1]

    global_min = np.min(pts)
    global_max = np.max(pts)
    scale_factor = global_max - global_min if (global_max - global_min) != 0.0 else 1.0
    
    normalized_pts = ((pts - global_min) / scale_factor).tolist()
    normalized_eps = eps_val / scale_factor
    print(f"완료! (총 {N}개의 점, {dimension}차원)\n")
    
    if N > 100:
        print(f"⚠️ 경고: 데이터 개수({N}개)가 많아 동형암호 연산에 매우 긴 시간이 소요될 수 있습니다.\n")

    # ---------------------------------------------------------
    # 4. Plaintext (NumPy) 시뮬레이션
    # ---------------------------------------------------------
    print("================ Plaintext 시뮬레이션 ===================")
    pt_start_time = time()
    transposed_data_np = list(zip(*normalized_pts))
    columns_simulated = [np.array(vector, dtype=np.float64) for vector in transposed_data_np]
    
    # 🚨 수정됨: debug_np 반환값을 추가로 받습니다.
    np_final_labels, _, debug_np = send_to_server_np(
        encrypted_columns=columns_simulated,
        num_points=N,
        eps=normalized_eps,
        min_pts=min_pts_val,
        dimension=dimension
    )
    
    cluster_labels_np = []
    for x in np_final_labels[:N]:
        r = round(x)
        if r <= 0: cluster_labels_np.append(-1)
        elif r > N: cluster_labels_np.append(N)
        else: cluster_labels_np.append(r)
        
    print(f"▶ Plaintext 소요 시간: {time() - pt_start_time:.2f}초\n")

    # ---------------------------------------------------------
    # 5. FHE 엔진 초기화 및 암호문 연산
    # ---------------------------------------------------------
    print("================ FHE 암호문 연산 ========================")
    print("FHE 엔진 및 키 생성 중 (GPU 모드)...")
    engine = Engine(use_bootstrap=True, mode="gpu")
    secret_key = engine.create_secret_key()
    keypack = KeyPack(
        public_key = engine.create_public_key(secret_key),
        rotation_key = engine.create_rotation_key(secret_key),
        relinearization_key = engine.create_relinearization_key(secret_key),
        conjugation_key = engine.create_conjugation_key(secret_key),
        bootstrap_key = engine.create_bootstrap_key(secret_key),
    )

    fhe_start_time = time()
    print("데이터 전체 암호화 및 서버 전송 중...")
    encryptor = DimensionalEncryptor(engine, keypack)
    encrypted_columns = encryptor.encrypt_data(normalized_pts, dimension)

    print("서버 동형암호 연산 중 (이 과정은 매우 오래 걸릴 수 있습니다)...")
    
    # 🚨 수정됨: debug_fhe 반환값을 추가로 받습니다.
    fhe_final_ct, debug_fhe = send_to_server_fhe(
        engine=engine,
        keypack=keypack,
        secret_key=secret_key, # 디버깅용 복호화를 위해 secret_key 전달
        encrypted_columns=encrypted_columns,
        num_points=N,
        eps=normalized_eps,
        min_pts=min_pts_val,
    )

    print("서버 연산 완료. 복호화 및 후처리 중...")
    decrypted_labels = engine.decrypt(fhe_final_ct, secret_key)
    
    cluster_labels_fhe = []
    for x in decrypted_labels[:N]:
        r = round(x)
        if r <= 0: cluster_labels_fhe.append(-1)
        elif r > N: cluster_labels_fhe.append(N)
        else: cluster_labels_fhe.append(r)
        
    print(f"▶ FHE 소요 시간: {time() - fhe_start_time:.2f}초\n")

    # ---------------------------------------------------------
    # 6. 중간 단계 값 차이 비교용 디버그 CSV 생성 (새로 추가된 부분)
    # ---------------------------------------------------------
    debug_filename = f"debug_fhe_vs_np_eps{eps_val}_min{min_pts_val}.csv"
    print("================ 중간값 오차 추적 파일 저장 ====================")
    print(f"단계별 근사 오차를 '{debug_filename}' 파일에 병합하여 저장 중...")
    
    try:
        with open(debug_filename, 'w', encoding='utf-8') as f:
            # CSV 헤더 작성 (어느 단계에서 오차가 커지는지 확인)
            f.write("Point_ID,NP_Neighbors,FHE_Neighbors,Diff_Neighbors,NP_Core,FHE_Core,Diff_Core,NP_Label,FHE_Label,Diff_Label\n")
            
            for i in range(N):
                # 1. 이웃 수 비교
                n_np = debug_np['total_neighbors'][i]
                n_fhe = debug_fhe['total_neighbors'][i]
                d_n = abs(n_np - n_fhe)

                # 2. 코어 포인트 여부 비교
                c_np = debug_np['core_mask'][i]
                c_fhe = debug_fhe['core_mask'][i]
                d_c = abs(c_np - c_fhe)

                # 3. 최종 라벨 비교
                l_np = debug_np['final_labels'][i]
                l_fhe = debug_fhe['final_labels'][i]
                d_l = abs(l_np - l_fhe)

                # 소수점 4자리까지 포맷팅하여 작성
                f.write(f"{i},{n_np:.4f},{n_fhe:.4f},{d_n:.4f},{c_np:.4f},{c_fhe:.4f},{d_c:.4f},{l_np:.4f},{l_fhe:.4f},{d_l:.4f}\n")
        print("✅ 디버깅 비교 파일 저장 완료!")
        print("👉 힌트: 저장된 CSV 파일에서 'Diff_Neighbors', 'Diff_Core', 'Diff_Label' 열 중 어느 지점부터 값이 크게 튀는지 확인하세요.\n")
    except Exception as e:
        print(f"❌ 디버깅 파일 저장 실패: {e}\n")

    # ---------------------------------------------------------
    # 7. 최종 결과 검증 및 저장
    # ---------------------------------------------------------
    print("================ 검증 결과 ==============================")
    
    # 7-1. 고유 클러스터 추출 및 요약
    pt_unique_clusters = sorted(list(set(cluster_labels_np)))
    pt_valid_clusters = [c for c in pt_unique_clusters if c != -1]
    
    fhe_unique_clusters = sorted(list(set(cluster_labels_fhe)))
    fhe_valid_clusters = [c for c in fhe_unique_clusters if c != -1]

    print("[Plaintext 시뮬레이션 결과]")
    print(f"  - 최종 발견된 클러스터 목록 (고유 ID): {pt_unique_clusters}")
    print(f"  - 총 {len(pt_valid_clusters)}개의 유효 군집과 노이즈(-1)가 도출되었습니다.\n")

    print("[FHE 연산 결과]")
    print(f"  - 최종 발견된 클러스터 목록 (고유 ID): {fhe_unique_clusters}")
    print(f"  - 총 {len(fhe_valid_clusters)}개의 유효 군집과 노이즈(-1)가 도출되었습니다.\n")

    # 7-2. ARI 검증
    ari_score = adjusted_rand_score(cluster_labels_np, cluster_labels_fhe)
    print("📊 [클러스터링 구조적 일치도 평가 (ARI)] 📊")
    print(f"  => 평문 시뮬레이션 구조 vs FHE 연산 구조 일치율: 【 {ari_score * 100:.2f} 점 / 100점 】")
    
    if ari_score > 0.99:
        print("  => 결론: 🎉 대성공! 라벨 번호만 부식되었을 뿐, 묶인 군집의 형태(Topology)는 평문과 100% 동일합니다!")
    elif ari_score > 0.80:
        print("  => 결론: 👍 대부분 동일하게 묶였으나, 경계선에서 소수의 오차가 발생했습니다.")
    else:
        print("  => 결론: ❌ 오차 누적으로 인해 군집 구조가 붕괴되었습니다 (라벨 융합 또는 파편화 발생).")

    # ---------------------------------------------------------
    # 8. CSV 파일 저장 (좌표, 원본 클래스, 예측 클래스 결합)
    # ---------------------------------------------------------
    output_filename = f"hepta_fhe_result_eps{eps_val}_min{min_pts_val}.csv"
    print(f"\n================ 파일 저장 ==============================")
    print(f"데이터셋의 모든 좌표와 클러스터 정보를 '{output_filename}' 에 병합 저장 중...")
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # 동적으로 헤더 생성 (x, y, z 등 차원에 맞게)
            axis_headers = [f"x{i+1}" for i in range(dimension)]
            header_str = ",".join(axis_headers) + ",True_Class,PT_Cluster,FHE_Cluster\n"
            f.write(header_str)
            
            # 데이터 작성
            for i in range(N):
                coords = ",".join([f"{val:.4f}" for val in pts[i]])
                f.write(f"{coords},{true_labels[i]},{cluster_labels_np[i]},{cluster_labels_fhe[i]}\n")
        print("✅ 파일 저장 완료! (추후 시각화 스크립트에서 이 파일을 직접 읽어 그리시면 됩니다.)")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")

if __name__ == '__main__':
    main()

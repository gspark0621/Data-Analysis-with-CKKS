# test_hepta.py (기존 test_random_blobs.py 대체)
import os
from time import time
import numpy as np 
from sklearn.metrics import adjusted_rand_score  # 클러스터링 정답률 평가 도구

# 작성해둔 평문 시뮬레이션 모듈 임포트
from core.plaintext.Client_main import run_client_dbscan

def load_fcps_dataset(file_path):
    """
    ARFF 파일을 읽어와서 좌표(pts)와 정답(true_labels)으로 분리합니다.
    차원(2D, 3D 등)에 상관없이 마지막 열을 클래스 라벨로 처리합니다.
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")

    pts = []
    true_labels = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # @data 태그가 나타난 이후부터가 실제 데이터입니다.
    is_data_section = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue  
            
        if line.lower() == '@data':
            is_data_section = True
            continue
            
        if is_data_section:
            parts = line.split(',')
            # 최소한 좌표 1개와 라벨 1개는 있어야 함 (길이 2 이상)
            if len(parts) >= 2:
                try:
                    # 마지막 값은 정답 라벨
                    label = float(parts[-1])
                    # 처음부터 마지막 직전까지의 값은 좌표 리스트로 변환
                    coordinates = [float(p) for p in parts[:-1]]
                    
                    pts.append(coordinates)
                    true_labels.append(label)
                except ValueError:
                    # 숫자로 변환할 수 없는 이상한 줄이 있으면 무시
                    continue
                
    return pts, true_labels


def main():
    print("==================================================")
    print("      Plaintext DBSCAN 시뮬레이션 (Hepta Dataset)    ")
    print("==================================================\n")

    # 1. 파일에서 데이터 불러오기
    # 사용자가 제공한 실제 경로
    file_path = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/hepta.arff"
    
    print(f"[{file_path}] 경로에서 데이터를 읽어옵니다...")
    pts, true_labels = load_fcps_dataset(file_path)
    
    num_points = len(pts)
    dimension = len(pts[0]) if num_points > 0 else 0
    print(f"데이터 로드 완료! (총 {num_points}개의 점, {dimension}차원)\n")

    # 2. 파라미터 설정
    # Hepta 데이터셋의 스케일에 맞춘 추천 파라미터 
    # (필요시 eps를 줄이거나 늘리며 max_iter 변화를 관찰해보세요)
    eps_value = 1
    min_pts_value = 4

    # 3. Client Plain 모듈 호출 
    print("================ 평문 시뮬레이션 연산 시작 ===============")
    start_time = time()
    
    # run_client_dbscan 호출 (3차원 데이터도 내부에서 자동 처리됨)
    result_pts, cluster_labels, required_iter = run_client_dbscan(pts, eps=eps_value, min_pts=min_pts_value)
    
    end_time = time()
    print("================ 평문 시뮬레이션 연산 종료 ===============\n")

    # 4. 결과 분석 및 ARI 점수(정확도) 계산
    ttime = end_time - start_time
    print(f"총 소요 시간: {ttime:.4f} 초")
    
    print("🔥 [핵심 결과] FHE 환경 적용을 위한 최적화 파라미터 도출 🔥")
    print(f"  => Hepta 데이터셋에서 라벨 전파가 완전히 수렴하기 위해 필요한")
    print(f"  => 정확한 max_iter 횟수는: 【 {required_iter}회 】 입니다.\n")
    
    unique_clusters = set(cluster_labels)
    valid_clusters = [c for c in unique_clusters if c != -1]
    
    print(f"최종 발견된 클러스터 목록 (고유 ID): {unique_clusters}")
    print(f"총 {len(valid_clusters)}개의 유효 군집과 노이즈가 도출되었습니다.\n")
    
    # 🔥 Adjusted Rand Index(ARI) 계산
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    
    print("📊 [클러스터링 일치율 평가 (Adjusted Rand Index)] 📊")
    print(f"  => 원본 데이터의 의도된 7개 그룹과 예측된 군집의 일치 점수: 【 {ari_score * 100:.2f} 점 / 100점 】")
    
    if ari_score > 0.95:
        print("  => 결과: 🌟 완벽에 가깝게 원본 그룹별로 잘 묶였습니다!")
    elif ari_score > 0.70:
        print("  => 결과: 👍 대부분 의도대로 묶였으나 일부 오차가 있습니다.")
    else:
        print("  => 결과: ⚠️ 군집이 섞이거나 제대로 묶이지 못했습니다. 파라미터(eps) 조절이 필요합니다.")


if __name__ == '__main__':
    main()

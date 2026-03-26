# test_lsun_original.py
import numpy as np
from time import time
# 원본 dbscan 코드가 담긴 파일에서 함수와 변수를 임포트합니다.
from Original_DBSCAN.dbscan import dbscan, NOISE

DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/hepta.arff"

def load_arff_to_pts(filepath: str, ignore_last_column: bool = True):
    """
    ARFF 파일을 읽어서 2차원 파이썬 리스트(pts)로 변환합니다.
    """
    pts = []
    data_section = False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            if line.lower().startswith('@data'):
                data_section = True
                continue
            
            if data_section:
                values = line.split(',')
                if ignore_last_column:
                    # 마지막 정답 클래스 라벨 제외하고 실수(float) 변환
                    row = [float(v) for v in values[:-1]]
                else:
                    row = [float(v) for v in values]
                pts.append(row)
                
    return pts

def main():
    print("==================================================")
    print("      Original DBSCAN 대화형 테스트 스크립트      ")
    print("         HEPTA (추천 eps=0.4, min_pts=4)         ")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 1. 사용자로부터 파라미터 직접 입력 받기 (Interactive)
    # ---------------------------------------------------------
    dataset_path = DATASET_PATH
    
    # eps 파라미터 입력 (필수)
    while True:
        try:
            eps_input = input("eps 값을 입력하세요 > ")
            eps_value = float(eps_input)
            break
        except ValueError:
            print("[오류] 올바른 숫자(실수)를 입력해주세요.\n")

    # min_pts 파라미터 입력 (필수)
    while True:
        try:
            min_pts_input = input("min_pts 값을 입력하세요 > ")
            min_pts_value = int(min_pts_input)
            break
        except ValueError:
            print("[오류] 올바른 정수를 입력해주세요.\n")

    print("\n--------------------------------------------------")
    print(f"▶ 데이터셋: {dataset_path}")
    print(f"▶ 파라미터: eps = {eps_value}, min_pts = {min_pts_value}")
    print("--------------------------------------------------\n")

    # ---------------------------------------------------------
    # 2. 데이터셋 로드
    # ---------------------------------------------------------
    print("데이터 로딩 중...")
    start_time = time()
    
    try:
        pts = load_arff_to_pts(dataset_path, ignore_last_column=True)
        print(f"데이터셋 로딩 완료! (총 {len(pts)}개의 점, {len(pts[0])}차원)\n")
    except FileNotFoundError:
        print(f"[치명적 오류] 파일을 찾을 수 없습니다: {dataset_path}")
        return

    # ---------------------------------------------------------
    # 3. Original DBSCAN 연산 실행
    # ---------------------------------------------------------
    print("================ Original DBSCAN 연산 시작 ================")
    dbscan_start_time = time()
    
    # 원본 dbscan 알고리즘은 m을 [Feature 차원 x Point 수] 형태로 요구하므로,
    # Numpy를 이용해 전치(Transpose)해줍니다.
    m_matrix = np.array(pts).T 
    
    # 원본 함수 호출
    cluster_labels = dbscan(m_matrix, eps_value, min_pts_value)
    
    # 출력 양식을 맞추기 위해 [X, Y, Cluster_ID] 형태로 조합
    result_pts = []
    for i in range(len(pts)):
        label = -1 if cluster_labels[i] is NOISE else cluster_labels[i]
        row = pts[i] + [label]
        result_pts.append(row)
        
    dbscan_end_time = time()
    print("================ Original DBSCAN 연산 종료 ================\n")

    # ---------------------------------------------------------
    # 4. 결과 출력 및 분석
    # ---------------------------------------------------------
    ttime = dbscan_end_time - start_time
    print(f"총 소요 시간: {ttime:.2f} 초 ({ttime / 60:.2f} 분)\n")
    
    print("\n--- 클러스터링 결과 샘플 (상위 10개) ---")
    print("[ X, Y, Z, ..., Cluster_ID ] (노이즈는 -1)")
    for row in result_pts[:10]:
        print(row)
        
    unique_clusters = set([lbl for lbl in cluster_labels if lbl is not NOISE])
    noise_count = cluster_labels.count(NOISE)
    
    print(f"\n최종 발견된 클러스터 목록 (고유 ID): {unique_clusters}")
    print(f"총 {len(unique_clusters)}개의 정상 군집과 {noise_count}개의 노이즈가 도출되었습니다.\n")

if __name__ == '__main__':
    main()

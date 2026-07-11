# test_target_original.py
import numpy as np
from time import time
from sklearn.metrics import adjusted_rand_score
from Original_DBSCAN.dbscan import dbscan, NOISE

DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/DBSCAN/target.arff"

# Ground Truth에서 small cluster → noise 처리할 임계값
NOISE_THRESHOLD = 10


def load_arff_to_pts(filepath: str, ignore_last_column: bool = True):
    pts = []
    labels = []
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
                    row = [float(v) for v in values[:-1]]
                    try:
                        labels.append(int(values[-1].strip()))
                    except ValueError:
                        labels.append(str(values[-1].strip()))
                else:
                    row = [float(v) for v in values]
                pts.append(row)

    return pts, labels


def convert_gt_labels(raw_labels, noise_threshold=NOISE_THRESHOLD):
    """
    포인트 수가 noise_threshold 이하인 클래스를 -1(noise)로 변환.
    Target 데이터셋의 outlier 그룹(4pts씩) 처리용.
    """
    raw = np.array(raw_labels)
    unique, counts = np.unique(raw, return_counts=True)
    converted = raw.copy()
    for u, c in zip(unique, counts):
        if c <= noise_threshold:
            converted[converted == u] = -1
    return converted


def main():
    print("==================================================")
    print("      Original DBSCAN 대화형 테스트 스크립트      ")
    print("           TARGET (eps, min_pts 입력)            ")
    print("==================================================\n")

    dataset_path = DATASET_PATH

    while True:
        try:
            eps_value = float(input("eps 값을 입력하세요 > "))
            break
        except ValueError:
            print("[오류] 올바른 숫자(실수)를 입력해주세요.\n")

    while True:
        try:
            min_pts_value = int(input("min_pts 값을 입력하세요 > "))
            break
        except ValueError:
            print("[오류] 올바른 정수를 입력해주세요.\n")

    print("\n--------------------------------------------------")
    print(f"▶ 데이터셋: {dataset_path}")
    print(f"▶ 파라미터: eps = {eps_value}, min_pts = {min_pts_value}")
    print("--------------------------------------------------\n")

    # ── 1. 데이터 로드 (좌표 + GT 레이블 동시에) ──────────────
    print("데이터 로딩 중...")
    start_time = time()

    try:
        pts, raw_gt = load_arff_to_pts(dataset_path, ignore_last_column=True)
        print(f"데이터셋 로딩 완료! (총 {len(pts)}개의 점, {len(pts[0])}차원)\n")
    except FileNotFoundError:
        print(f"[치명적 오류] 파일을 찾을 수 없습니다: {dataset_path}")
        return

    # GT 레이블 변환 (outlier 그룹 → -1)
    gt_labels = convert_gt_labels(raw_gt, NOISE_THRESHOLD)
    unique_gt, counts_gt = np.unique(gt_labels, return_counts=True)
    print("▶ Ground Truth 레이블 분포 (변환 후):")
    for u, c in zip(unique_gt, counts_gt):
        tag = " ← noise" if u == -1 else ""
        print(f"   Class {u}: {c}pts{tag}")
    print()

    # ── 2. Original DBSCAN 실행 ──────────────────────────────
    print("================ Original DBSCAN 연산 시작 ================")
    dbscan_start_time = time()

    m_matrix = np.array(pts).T
    cluster_labels = dbscan(m_matrix, eps_value, min_pts_value)

    result_pts = []
    pred_labels = []
    for i in range(len(pts)):
        label = -1 if cluster_labels[i] is NOISE else cluster_labels[i]
        pred_labels.append(label)
        result_pts.append(pts[i] + [label])

    dbscan_end_time = time()
    print("================ Original DBSCAN 연산 종료 ================\n")

    ttime = dbscan_end_time - start_time
    print(f"총 소요 시간: {ttime:.2f} 초 ({ttime / 60:.2f} 분)\n")

    # ── 3. 결과 출력 ─────────────────────────────────────────
    print("--- 클러스터링 결과 샘플 (상위 10개) ---")
    print("[ X, Y, ..., Cluster_ID ] (노이즈는 -1)")
    for row in result_pts[:10]:
        print(row)

    pred_arr = np.array(pred_labels)
    unique_clusters = sorted(set(pred_labels) - {-1})
    noise_count = pred_labels.count(-1)

    print(f"\n최종 발견된 클러스터 목록 (고유 ID): {unique_clusters}")
    print(f"총 {len(unique_clusters)}개의 정상 군집과 {noise_count}개의 노이즈가 도출되었습니다.\n")

    # ── 4. ARI 계산 ──────────────────────────────────────────
    print("============================================================")
    print("                    ARI (Adjusted Rand Index)               ")
    print("============================================================")

    # [방법 1] 전체 포인트 기준 ARI (노이즈 포함)
    ari_all = adjusted_rand_score(gt_labels, pred_arr)
    print(f"  ARI (전체, 노이즈 포함)          : {ari_all:.4f}")

    # [방법 2] GT 기준 노이즈(-1) 제외한 포인트만 비교
    mask_gt = gt_labels != -1
    ari_no_gt_noise = adjusted_rand_score(gt_labels[mask_gt], pred_arr[mask_gt])
    print(f"  ARI (GT 노이즈 제외)             : {ari_no_gt_noise:.4f}")

    # [방법 3] Pred 기준 노이즈(-1) 제외한 포인트만 비교
    mask_pred = pred_arr != -1
    ari_no_pred_noise = adjusted_rand_score(gt_labels[mask_pred], pred_arr[mask_pred])
    print(f"  ARI (Pred 노이즈 제외)           : {ari_no_pred_noise:.4f}")

    # [방법 4] GT + Pred 양쪽 모두 노이즈 아닌 포인트만 비교 (가장 엄격)
    mask_both = mask_gt & mask_pred
    ari_strict = adjusted_rand_score(gt_labels[mask_both], pred_arr[mask_both])
    print(f"  ARI (GT+Pred 노이즈 모두 제외)   : {ari_strict:.4f}")

    print("============================================================")
    print(f"\n  ★ 주 참고 지표: ARI (GT 노이즈 제외) = {ari_no_gt_noise:.4f}")
    if ari_no_gt_noise >= 0.9:
        grade = "🏆 Excellent"
    elif ari_no_gt_noise >= 0.7:
        grade = "✅ Good"
    elif ari_no_gt_noise >= 0.5:
        grade = "⚠️  Fair"
    else:
        grade = "❌ Poor"
    print(f"  ★ 평가: {grade}")
    print("============================================================\n")


if __name__ == '__main__':
    main()
    
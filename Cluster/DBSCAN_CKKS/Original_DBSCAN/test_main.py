# test_main.py
# ─────────────────────────────────────────────────────────────────────────────
# KD-LP 평문 미러 DBSCAN — "수렴까지 몇 round 가 필요했는가" 를 측정한다.
#
# 원본(Ester et al.) BFS DBSCAN 대신, FHE 파이프라인(Client_main / Server_main /
# Label_Propagation.fhe_kd_dense_propagation)과 **같은 순서·같은 규칙**으로
# 평문에서 라벨 전파를 수행한다.
#
#   1. 정규화      : scale = max(좌표범위, PC1 span)            (Client_main 과 동일)
#   2. 슬롯 정렬   : PC1 사영 argsort (SVD 부호 고정 포함)        (build_pca_order)
#   3. k_max       : 1-Lipschitz ε-window two-pointer 상한       (compute_kmax_from_pca_window)
#   4. adj_k       : slot i ↔ slot (i+k) mod N,  dist² ≤ ε_n²   (Normalize: 닫힌 구간)
#   5. core        : 1 + Σ_k (adj_k + adj_{N−k}) ≥ min_pts       (Server_main / Core)
#   6. Phase 1     : Core-Core max 전파
#                    · stride 1..k_max 를 크기 m 그룹으로 분할   (_slot_group_size, _stride_groups)
#                    · 그룹 내부 = 같은 스냅샷에서 후보 계산 후 max (패킹 트리-max, Jacobi)
#                    · 그룹 사이 = 순차 (Gauss-Seidel)
#                    · 1 round = forward(그룹 오름차순) + backward(그룹 내림차순)
#                    · ★ 고정 n_rounds 대신 "라벨이 더 이상 안 바뀔 때까지" 반복하고 round 수를 센다
#   7. Phase 2     : Core→Border  forward+backward 1회 (source=core 라벨 고정 → round 무관)
#   8. 후처리      : 평문은 라벨이 정확한 정수 → 라벨 그대로 클러스터 ID, 0 → 노이즈(-1)
#
# [round 의 정의]
#   R_needed = 라벨을 1개 이상 바꾼 round 의 수
#            = FHE 에서 n_rounds ≥ R_needed 이면 고정점(=정답 연결성분)에 도달하는 최소값.
#   rounds_executed = R_needed + 1  (마지막 1 round 는 "변화 없음" 확인용)
# ─────────────────────────────────────────────────────────────────────────────
import csv
import math
import os
from time import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


DATASET_PATH = "/home/junhyung/study/Data-Analysis-with-CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/swiss_banknote.arff"

# ── FHE 파이프라인과 맞춰야 하는 설정 ─────────────────────────────────────────
SLOT_COUNT        = 32768   # engine.slot_count (logN=16). 엔진 설정이 바뀌면 같이 바꿀 것 → m 이 바뀜
FORCE_GROUP_SIZE  = None    # Label_Propagation._FORCE_GROUP_SIZE 와 동일 의미 (None=자동)
MAX_ROUNDS_CAP    = None    # None 이면 N (고정점은 D_hop ≤ N−1 round 안에 반드시 도달)
PC1_SIGN_FIX      = True    # True  = Client_main.run_client_dbscan_fhe (절댓값 최대 성분 양수 고정)
                            # False = test_Server_main_call_dataset.py (부호 고정 없음, LAPACK 의존)
                            # ★ 부호가 뒤집히면 슬롯 순서가 역전 → round 수·border 소속이 달라질 수 있음

# ── 부가 기능 ────────────────────────────────────────────────────────────────
VERBOSE_ROUNDS    = True    # round 별 변경 슬롯 수 출력
SAVE_ROUNDS_CSV   = "kdlp_rounds_result.csv"   # None 이면 저장 안 함 (있으면 행 추가)
COMPARE_ORIGINAL  = False   # True 면 원본 dbscan(Original_DBSCAN) 도 돌려 ARI 비교 (O(N²) 파이썬 루프, 느림)


# ═════════════════════════════════════════════════════════════════════════════
# 데이터 로드 (기존 그대로)
# ═════════════════════════════════════════════════════════════════════════════

def load_arff_to_pts(filepath: str, ignore_last_column: bool = True):
    """ARFF 파일을 읽어서 2차원 파이썬 리스트(pts)로 변환합니다."""
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
                    row = [float(v) for v in values[:-1]]
                else:
                    row = [float(v) for v in values]
                pts.append(row)
    return pts


def load_arff_true_labels(filepath: str):
    """ARFF 파일의 마지막 컬럼(정답 클래스 라벨)만 추출합니다 (ARI 의 ground truth)."""
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
                labels.append(values[-1].strip())
    return labels


# ═════════════════════════════════════════════════════════════════════════════
# Client 측 (Client_main.py 미러)
# ═════════════════════════════════════════════════════════════════════════════

def normalize_like_client(pts_arr: np.ndarray, eps: float):
    """Client_main.run_client_dbscan_fhe 의 정규화 블록과 동일."""
    g_min   = pts_arr.min()
    g_max   = pts_arr.max()
    coord_r = (g_max - g_min) or 1.0

    Xc = pts_arr - pts_arr.mean(axis=0)
    ax = np.linalg.svd(Xc, full_matrices=False)[2][0]
    if PC1_SIGN_FIX and ax[np.argmax(np.abs(ax))] < 0:   # SVD 부호 고정 (슬롯 순서 결정론화)
        ax = -ax
    pc1      = Xc @ ax
    pc1_span = float(pc1.max() - pc1.min())

    scale  = max(coord_r, pc1_span) or 1.0
    norm   = (pts_arr - g_min) / scale
    eps_n  = eps / scale
    proj_n = pc1 / scale
    return norm, eps_n, proj_n, scale, coord_r, pc1_span


def build_pca_order(proj: np.ndarray):
    """Client_main.build_pca_order (proj 전달 경로)."""
    N = len(proj)
    order = np.argsort(proj, kind="stable")
    inv_perm = np.empty(N, dtype=np.int64)
    inv_perm[order] = np.arange(N, dtype=np.int64)
    return order, inv_perm, proj[order]


def compute_kmax_from_pca_window(proj_sorted: np.ndarray, eps_norm: float, N: int) -> int:
    """Client_main.compute_kmax_from_pca_window 와 동일."""
    if N <= 1:
        return 1
    j = 0
    W = 0
    for i in range(N):
        if j < i:
            j = i
        while j + 1 < N and proj_sorted[j + 1] - proj_sorted[i] <= eps_norm:
            j += 1
        W = max(W, j - i)
    return min(max(W, 1), N // 2)


# ═════════════════════════════════════════════════════════════════════════════
# Server 측 (Server_main.py / Label_Propagation.py 미러)
# ═════════════════════════════════════════════════════════════════════════════

def n_rounds_from_public(eps_norm: float, N: int, min_pts: int) -> int:
    """Server_main.n_rounds_from_public 와 동일 — 비교 출력용."""
    geo = math.ceil(1.0 / eps_norm)
    cap = math.ceil(3.0 * N / max(min_pts, 1)) - 1
    return max(1, min(geo, cap))


def _max_n_region_no_wrap(N: int, slot_count: int) -> int:
    r = 2
    while (r + 1 + math.ceil((r + 1) / 2.0)) * N <= slot_count:
        r += 1
    return max(2, r)


def _slot_group_size(N: int, k_max: int, slot_count: int) -> int:
    cap = max(1, _max_n_region_no_wrap(N, slot_count) - 1)
    if FORCE_GROUP_SIZE is not None:
        return max(1, min(FORCE_GROUP_SIZE, k_max, cap))
    return max(1, min(k_max, cap))


def _stride_groups(k_max: int, m: int):
    return [list(range(s, min(s + m - 1, k_max) + 1)) for s in range(1, k_max + 1, m)]


def _shift(v: np.ndarray, k: int) -> np.ndarray:
    """fhe_circular_shift: result[i] = v[(i+k) mod N]."""
    return np.roll(v, -k, axis=0)


def build_adj_k_list(sorted_pts: np.ndarray, eps_norm: float, k_max: int):
    """adj_k[i] = 1  iff  ‖p_i − p_{(i+k) mod N}‖² ≤ ε_n²   (k = 1..k_max, 슬롯 순서).
    Normalize.check_neighbor_closed_interval 의 닫힌 구간 판정에 대응
    (FHE 쪽 margin 2^-α 는 평문에서 0 으로 둔다)."""
    eps_sq = eps_norm ** 2
    adj = []
    for k in range(1, k_max + 1):
        d2 = np.sum((sorted_pts - _shift(sorted_pts, k)) ** 2, axis=1)
        adj.append(d2 <= eps_sq)
    return adj


def count_neighbors(adj_k_list, N: int) -> np.ndarray:
    """Server_main Step 1 의 total_neighbors 누적 (자기 자신 포함)."""
    total = np.ones(N, dtype=np.int64)
    for k, a in enumerate(adj_k_list, start=1):
        a = a.astype(np.int64)
        if 2 * k < N:
            total += a + _shift(a, N - k)       # adj_{N-k} = shift(adj_k, N−k)
        else:                                    # k == N/2: 중복 방지
            total += a
    return total


def kdlp_label_propagation(adj_k_list, core: np.ndarray, N: int, k_max: int,
                           slot_count: int, max_rounds: int, verbose: bool = True):
    """fhe_kd_dense_propagation 의 평문 미러.  n_rounds 를 고정하지 않고 수렴까지 돌린다.

    반환: final(slot 순서 정수 라벨, 0=노이즈), core_labels, info(dict)
    """
    core_i  = core.astype(np.int64)
    noncore = 1 - core_i

    m      = _slot_group_size(N, k_max, slot_count)
    groups = _stride_groups(k_max, m)
    assert (m + 1) * N <= slot_count, "패킹 슬롯 초과 — _pack_candidates 가 후보를 버리게 됨"

    # adjm_fwd = adj_k ⊙ shift(core,k) ⊙ core(dest),  adjm_bwd = shift(adjm_fwd, N−k)
    adjm_fwd = [a.astype(np.int64) * _shift(core_i, k) * core_i
                for k, a in enumerate(adj_k_list, start=1)]
    adjm_bwd = [(_shift(adjm_fwd[k - 1], N - k) if 2 * k < N else None)
                for k in range(1, k_max + 1)]

    labels = core_i * np.arange(1, N + 1, dtype=np.int64)     # 초기값 = core·(슬롯인덱스+1)

    # ── Phase 1: Core-Core, 변화가 없을 때까지 ──
    history = []
    rounds_executed = 0
    R_needed = 0
    converged = False
    for rnd in range(1, max_rounds + 1):
        before = labels.copy()
        rounds_executed = rnd
        for pass_name in ("forward", "backward"):
            g_order = groups if pass_name == "forward" else list(reversed(groups))
            for g in g_order:
                base = labels                               # 그룹 시작 스냅샷 (packed 영역0)
                acc  = base.copy()
                for k in g:                                  # 그룹 내부: 모두 같은 base 에서 후보 생성
                    if pass_name == "forward":
                        cand = adjm_fwd[k - 1] * _shift(base, k)
                    else:
                        if adjm_bwd[k - 1] is None:          # 2k ≥ N → backward 없음
                            continue
                        cand = adjm_bwd[k - 1] * _shift(base, N - k)
                    np.maximum(acc, cand, out=acc)
                labels = acc                                 # 그룹 사이: 순차
        n_changed = int(np.count_nonzero(labels != before))
        history.append(n_changed)
        if verbose:
            n_lbl = len(np.unique(labels[core_i == 1])) if core_i.any() else 0
            print(f"  [Round {rnd:3d}] 변경 슬롯 {n_changed:5d}개 | core 라벨 종류 {n_lbl}")
        if n_changed == 0:
            converged = True
            break
        R_needed = rnd

    # ── Phase 2: Core→Border (source = 고정된 core 라벨 → 1회 fwd+bwd 로 충분) ──
    border = np.zeros(N, dtype=np.int64)
    for pass_name in ("forward", "backward"):
        g_order = groups if pass_name == "forward" else list(reversed(groups))
        for g in g_order:
            acc = border.copy()
            for k in g:
                a = adj_k_list[k - 1].astype(np.int64)
                if pass_name == "forward":
                    adjm = a * _shift(core_i, k) * noncore
                    cand = adjm * _shift(labels, k)
                else:
                    if 2 * k >= N:
                        continue
                    adjm = _shift(a, N - k) * _shift(core_i, N - k) * noncore
                    cand = adjm * _shift(labels, N - k)
                np.maximum(acc, cand, out=acc)
            border = acc

    final = labels + border
    info = dict(m=m, n_groups=len(groups), R_needed=R_needed,
                rounds_executed=rounds_executed, converged=converged, history=history)
    return final, labels, info


def verify_fixpoint(adj_k_list, core: np.ndarray, core_labels: np.ndarray, N: int) -> bool:
    """고정점 검증: core 라벨 == (core-core 연결성분 내 최대 슬롯인덱스+1)."""
    rows, cols = [], []
    idx = np.arange(N)
    for k, a in enumerate(adj_k_list, start=1):
        e = a & core.astype(bool) & _shift(core.astype(bool), k)
        rows.append(idx[e]); cols.append((idx[e] + k) % N)
    if not rows:
        return True
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(N, N))
    _, comp = connected_components(g, directed=False)
    expect = np.zeros(N, dtype=np.int64)
    for cid in np.unique(comp[core.astype(bool)]):
        mem = np.where((comp == cid) & core.astype(bool))[0]
        expect[mem] = mem.max() + 1
    return bool(np.array_equal(expect, core_labels))


# ═════════════════════════════════════════════════════════════════════════════
# 진입점
# ═════════════════════════════════════════════════════════════════════════════

def run_kdlp_plaintext(pts: list, eps: float, min_pts: int, verbose: bool = True):
    """pts(원래 좌표) → (클러스터 라벨 list[int], -1=노이즈), info"""
    pts_arr = np.asarray(pts, dtype=np.float64)
    N, dim  = pts_arr.shape

    norm, eps_n, proj_n, scale, coord_r, pc1_span = normalize_like_client(pts_arr, eps)
    order, inv_perm, proj_sorted = build_pca_order(proj_n)
    k_max = compute_kmax_from_pca_window(proj_sorted, eps_n, N)
    sorted_pts = norm[order]

    print(f"[KD-LP] PC1 부호 고정: {'ON (Client_main)' if PC1_SIGN_FIX else 'OFF (test_Server_main)'}")
    print(f"[KD-LP] N={N}, dim={dim}, eps_norm={eps_n:.6f} "
          f"(scale={scale:.4f}: 좌표범위 {coord_r:.4f} / PC1span {pc1_span:.4f})")

    adj_k_list = build_adj_k_list(sorted_pts, eps_n, k_max)
    total_nb   = count_neighbors(adj_k_list, N)
    core       = (total_nb >= min_pts).astype(np.int64)

    # ★ k_max 안전성 교차검증: k ≤ k_max 누적 이웃수 == 전수 이웃수 (sklearn radius, ≤ eps)
    nn = NearestNeighbors(radius=eps_n).fit(sorted_pts)
    brute = np.array([len(x) for x in nn.radius_neighbors(sorted_pts, return_distance=False)])
    n_mis = int(np.count_nonzero(brute != total_nb))
    if n_mis:
        print(f"[KD-LP] ⚠ 이웃수 불일치 {n_mis}개 (k_max 누락 또는 ε 경계 부동소수 차이)")

    m_prev = _slot_group_size(N, k_max, SLOT_COUNT)
    print(f"[KD-LP] k_max={k_max}, 그룹 크기 m={m_prev}, "
          f"그룹 수 {math.ceil(k_max / m_prev)} × 2방향, core {int(core.sum())}/{N}\n")

    max_rounds = MAX_ROUNDS_CAP if MAX_ROUNDS_CAP is not None else max(N, 1)
    final_slot, core_labels, info = kdlp_label_propagation(
        adj_k_list, core, N, k_max, SLOT_COUNT, max_rounds, verbose=verbose)

    info["fixpoint_ok"] = verify_fixpoint(adj_k_list, core, core_labels, N)
    info.update(N=N, dim=dim, eps_norm=eps_n, k_max=k_max,
                n_core=int(core.sum()), neighbor_mismatch=n_mis,
                n_rounds_public=n_rounds_from_public(eps_n, N, min_pts))
    per_round = 2 * info["n_groups"] * math.ceil(math.log2(info["m"] + 1))
    info["fhe_max_per_round"] = per_round

    final_orig = final_slot[inv_perm]                        # 슬롯 순서 → 원래 순서
    cluster_labels = [-1 if v == 0 else int(v) for v in final_orig]
    return cluster_labels, info


def append_rounds_csv(path, dataset, eps, min_pts, info, ari):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "N", "dim", "eps", "min_pts", "k_max", "m", "n_groups",
                        "R_needed", "rounds_executed", "n_rounds_public", "fhe_max_per_round",
                        "n_core", "ARI"])
        w.writerow([os.path.basename(dataset), info["N"], info["dim"], eps, min_pts,
                    info["k_max"], info["m"], info["n_groups"], info["R_needed"],
                    info["rounds_executed"], info["n_rounds_public"],
                    info["fhe_max_per_round"], info["n_core"],
                    f"{ari:.6f}" if ari is not None else ""])


def main():
    print("==================================================")
    print("   KD-LP 평문 미러 DBSCAN — 수렴 round 측정 스크립트   ")
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

    # ── 데이터 로드 ──
    print("데이터 로딩 중...")
    start_time = time()
    try:
        pts = load_arff_to_pts(dataset_path, ignore_last_column=True)
        true_labels = load_arff_true_labels(dataset_path)
        print(f"데이터셋 로딩 완료! (총 {len(pts)}개의 점, {len(pts[0])}차원)\n")
    except FileNotFoundError:
        print(f"[치명적 오류] 파일을 찾을 수 없습니다: {dataset_path}")
        return

    # ── KD-LP 평문 미러 실행 ──
    print("================ KD-LP 평문 DBSCAN 연산 시작 ================")
    dbscan_start_time = time()
    cluster_labels, info = run_kdlp_plaintext(pts, eps_value, min_pts_value,
                                              verbose=VERBOSE_ROUNDS)
    result_pts = [pts[i] + [cluster_labels[i]] for i in range(len(pts))]
    dbscan_end_time = time()
    print("================ KD-LP 평문 DBSCAN 연산 종료 ================\n")

    # ── 결과 출력 ──
    ttime = dbscan_end_time - start_time
    print(f"총 소요 시간: {ttime:.2f} 초 ({ttime / 60:.2f} 분)\n")

    print("\n--- 클러스터링 결과 샘플 (상위 10개) ---")
    print("[ X, Y, Z, ..., Cluster_ID ] (노이즈는 -1, ID 는 성분 내 최대 슬롯인덱스+1)")
    for row in result_pts[:10]:
        print(row)

    unique_clusters = set(l for l in cluster_labels if l != -1)
    noise_count = cluster_labels.count(-1)
    print(f"\n최종 발견된 클러스터 목록 (고유 ID): {unique_clusters}")
    print(f"총 {len(unique_clusters)}개의 정상 군집과 {noise_count}개의 노이즈가 도출되었습니다.\n")

    # ── ARI ──
    print("================ ARI (Adjusted Rand Index) 측정 ================")
    ari_score = None
    if len(cluster_labels) != len(true_labels):
        print("[경고] 예측 라벨과 정답 라벨의 개수가 일치하지 않습니다. ARI 계산을 건너뜁니다.")
    else:
        ari_score = adjusted_rand_score(true_labels, cluster_labels)
        print(f"▶ ARI Score: {ari_score:.4f}")
        print("  (1.0에 가까울수록 정답 클러스터와 잘 일치, 0.0에 가까우면 랜덤 수준)")

    if COMPARE_ORIGINAL:
        from Original_DBSCAN.dbscan import dbscan, NOISE
        orig = dbscan(np.array(pts).T, eps_value, min_pts_value)
        orig = [-1 if l is NOISE else l for l in orig]
        print(f"▶ ARI (KD-LP 평문 vs 원본 dbscan): {adjusted_rand_score(orig, cluster_labels):.4f}"
              f"   ※ 원본은 dist < eps(열린 구간)·선착순 border, KD-LP 는 ≤ eps·max-label border")
    print("==================================================================\n")

    # ── ★ Round 결과 ──
    print("================ Label Propagation Round 결과 ================")
    print(f"▶ 수렴에 필요한 round 수 (R_needed)   : {info['R_needed']}")
    print(f"  실행한 round 수 (확인 round 포함)     : {info['rounds_executed']}"
          f"{'' if info['converged'] else '  ★ 상한 도달 — 미수렴'}")
    print(f"  round 별 변경 슬롯 수                 : {info['history']}")
    print(f"  서버 공개식 n_rounds = min(⌈1/ε_n⌉, ⌈3N/min_pts⌉−1) = {info['n_rounds_public']}"
          f"  → {'충분 ✓' if info['n_rounds_public'] >= info['R_needed'] else '★ 부족 (under-merge)'}"
          + (f" (여유 {info['n_rounds_public'] / info['R_needed']:.2f}배)" if info['R_needed'] else ""))
    print(f"  k_max={info['k_max']}, m={info['m']}, 그룹 {info['n_groups']}개 "
          f"→ round 당 fhe_max {info['fhe_max_per_round']}회 "
          f"(R_needed 기준 Phase1 {info['fhe_max_per_round'] * info['R_needed']}회)")
    print(f"  고정점 검증 (core 라벨 == 연결성분 max) : {'✓' if info['fixpoint_ok'] else '★ 실패'}")
    print("==================================================================\n")

    if SAVE_ROUNDS_CSV:
        append_rounds_csv(SAVE_ROUNDS_CSV, dataset_path, eps_value, min_pts_value, info, ari_score)
        print(f"round 결과 저장: {SAVE_ROUNDS_CSV}")

    return info['R_needed']


if __name__ == '__main__':
    main()
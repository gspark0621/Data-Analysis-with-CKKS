# core/ciphertext_single/Client_main.py
#
# ── 변경사항 ────────────────────────────────────────────────────────────────
# [변경 1] build_ball_tree_order: BFS → DFS in-order 레이아웃
#   BFS 문제:
#     left/right subtree 위치가 {1,3,4,7,8,...} 형태로 전체 배열에 인터리빙
#     → 클러스터 span ≈ N, k_max = N//2 불가피
#     → heap window 10개 내 평균 4.66개 클러스터 혼재 (실측)
#   DFS 해결:
#     left | root | right → left=[start, start+n_L-1] (연속)
#     → 클러스터 span ≈ cluster_size, k_max 대폭 감소
#     → enc_sum_k가 실제 작은 k_max를 올바르게 반영
#
# [변경 2] compute_kmax_from_ball_structure: DFS 레이아웃 기반 candidate 수식
#   circular stride = min(n_L+n_R, N-(n_L+n_R))  (DFS cross-boundary)
#
# [변경 3] dead code 제거
#   compute_kmax_from_ball_structure 내 return 이후 unreachable 코드 삭제
#
# [변경 4] decide_propagation_mode 중복 정의 제거
#   첫 번째 (deprecated) 정의 삭제, 단일 정의 유지
#
# [변경 5] run_client_dbscan_fhe: prepare_client_ordering 사용
#   decide_propagation_mode + 별도 build_kd_tree_order 호출 → 통합
# ────────────────────────────────────────────────────────────────────────────

from time import time
import math
import numpy as np
import desilofhe
import pynvml
from core.ciphertext_single.EncryptModule import DimensionalEncryptor
from core.ciphertext_single.Server_main import send_to_server_fhe as send_to_server
from util.keypack import KeyPack

# 전파 방식 임계값: 상단 정의로 모든 함수에서 참조 가능
_KD_DENSE_MIN_PTS_THRESHOLD = 4


def _gpu_used_mb() -> float:
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0


# ── Ball Tree DFS in-order 정렬 ───────────────────────────────────────────

def build_ball_tree_order(pts: np.ndarray) -> tuple:
    """
    Ball Tree를 DFS in-order 레이아웃으로 정렬.

    DFS in-order: left subtree | root | right subtree
    → 각 subtree 멤버가 연속된 위치를 차지:
        left  → [start, start+n_L-1]   (n_L개 연속)
        root  → [start+n_L]
        right → [start+n_L+1, end]     (n_R개 연속)

    BFS vs DFS 핵심 차이:
      BFS: left = {1,3,4,7,8,...} 전체 배열에 인터리빙
           → 클러스터 span ≈ N  → k_max = N//2 불가피 → 848회 fhe_max
      DFS: left = [start, start+n_L-1] 연속
           → 클러스터 span ≈ cluster_size  → k_max 대폭 감소 → 연산량 대폭 감소

    Returns
    -------
    order    : np.ndarray[int, N]  order[i] = DFS 위치 i의 원래 데이터 인덱스
    inv_perm : np.ndarray[int, N]  inv_perm[j] = 원래 인덱스 j의 DFS 위치
    """
    pts_arr = np.array(pts, dtype=np.float64)
    N       = len(pts_arr)
    order   = np.empty(N, dtype=int)

    def _build(indices: np.ndarray, start: int):
        if len(indices) == 0:
            return
        if len(indices) == 1:
            order[start] = indices[0]
            return

        coords   = pts_arr[indices]
        centroid = coords.mean(axis=0)
        d_sq_c   = np.einsum('ij,ij->i', coords - centroid, coords - centroid)
        pole1    = coords[int(np.argmax(d_sq_c))]
        d_sq_p1  = np.einsum('ij,ij->i', coords - pole1, coords - pole1)
        pole2    = coords[int(np.argmax(d_sq_p1))]
        axis     = pole2 - pole1
        ax_norm  = float(np.dot(axis, axis)) ** 0.5

        if ax_norm < 1e-12:
            proj_order = np.arange(len(indices))
        else:
            proj_order = np.argsort(
                (coords - pole1) @ (axis / ax_norm), kind='stable')

        mid = len(indices) // 2
        # DFS in-order: left | root | right  ← 모두 연속 범위에 배치
        _build(indices[proj_order[:mid]],      start)             # left:  [start, start+mid-1]
        order[start + mid] = indices[proj_order[mid]]             # root:  [start+mid]
        _build(indices[proj_order[mid + 1:]], start + mid + 1)    # right: [start+mid+1, end]

    _build(np.arange(N), 0)

    inv_perm        = np.empty(N, dtype=int)
    inv_perm[order] = np.arange(N)
    return order, inv_perm


def compute_kmax_from_ball_structure(
    pts: np.ndarray,
    heap_idx: np.ndarray,   # 서명 유지 (내부에서 재구성)
    eps_norm: float,
    N: int,
) -> int:
    """
    DFS in-order Ball Tree 구조 분석으로 k_max 상한 계산 — eps-이웃 조회 없음.

    DFS in-order cross-boundary circular stride:
      left  subtree: [start, start+n_L-1]
      right subtree: [start+n_L+1, start+n_L+n_R]
      max linear stride  = n_L + n_R
      max circular stride = min(n_L+n_R, N-(n_L+n_R))

    BFS와 달리 클러스터가 연속 범위에 집중 → ball_gap < eps인 split이
    cluster 내부에 집중 → k_max ≈ cluster_size (BFS: k_max ≈ N/2)

    비용: O(N log N) — eps-이웃 조회 없음 (시나리오 2 보호)
    서버의 enc_sum_k가 이후 정밀화 (단, 클라이언트 상한 초과 불가)
    """
    pts_arr = np.array(pts, dtype=np.float64)
    k_max   = 0

    def _analyze(indices: np.ndarray):
        nonlocal k_max
        if len(indices) <= 1:
            if len(indices) == 1:
                return pts_arr[indices[0]].copy(), 0.0
            return np.zeros(pts_arr.shape[1]), 0.0

        coords   = pts_arr[indices]
        centroid = coords.mean(axis=0)
        radius   = float(np.max(np.linalg.norm(coords - centroid, axis=1)))

        d_sq_c  = np.einsum('ij,ij->i', coords - centroid, coords - centroid)
        pole1   = coords[int(np.argmax(d_sq_c))]
        d_sq_p1 = np.einsum('ij,ij->i', coords - pole1, coords - pole1)
        pole2   = coords[int(np.argmax(d_sq_p1))]
        axis    = pole2 - pole1
        ax_norm = float(np.dot(axis, axis)) ** 0.5

        if ax_norm < 1e-12:
            proj_order = np.arange(len(indices))
        else:
            proj_order = np.argsort(
                (coords - pole1) @ (axis / ax_norm), kind='stable')

        mid = len(indices) // 2
        n_L = mid
        n_R = len(indices) - mid - 1

        c_L, r_L = _analyze(indices[proj_order[:mid]])
        c_R, r_R = _analyze(indices[proj_order[mid + 1:]])

        if n_L > 0 and n_R > 0:
            ball_gap = float(np.linalg.norm(c_L - c_R)) - r_L - r_R
            if ball_gap < eps_norm:
                # DFS in-order: cross-boundary circular stride
                # left=[start,start+n_L-1], right=[start+n_L+1,start+n_L+n_R]
                # max linear = n_L+n_R, circular = min(n_L+n_R, N-(n_L+n_R))
                cross_span = n_L + n_R
                candidate  = min(cross_span, N - cross_span)
                if candidate > k_max:
                    k_max = candidate

        return centroid, radius

    _analyze(np.arange(N))
    return min(max(k_max, 1), N // 2)


# ── deprecated aliases ────────────────────────────────────────────────────

def build_kd_tree_order(pts: np.ndarray) -> tuple:
    """Deprecated alias → build_ball_tree_order (DFS in-order)."""
    return build_ball_tree_order(pts)


def get_kd_dense_kmax(N: int) -> int:
    """Deprecated: 안전 상한 N//2 반환."""
    return N // 2


# ── Ball Tree DFS 정렬 + k_max 구조 분석 통합 ────────────────────────────

def prepare_client_ordering(
    norm_pts: np.ndarray,
    eps_norm: float,
    min_pts: int,
    N: int,
) -> tuple:
    """
    Ball Tree DFS in-order 정렬 + 구조 분석 k_max 상한 계산.

    k_max 결정 순서:
      1. [Client, 이 함수] DFS Ball Tree 구조 분석
         → k_max 상한 (eps-이웃 조회 없음, 시나리오 2 보호)
      2. [Server] enc_sum_k 계산 → client 복호화 → k_max 정밀화
         단, 서버 k_max ≤ 클라이언트 상한 (false positive 과대추정 방지)

    Returns (mode, k_max, order, inv_perm)
    """
    if min_pts >= _KD_DENSE_MIN_PTS_THRESHOLD:
        t0 = time()
        order, inv_perm = build_ball_tree_order(norm_pts)
        print(f"[Client] Ball Tree DFS 구축: {time()-t0:.3f}초")

        t1 = time()
        k_max = compute_kmax_from_ball_structure(norm_pts, order, eps_norm, N)
        print(f"[Client] k_max 구조 분석 (DFS): {time()-t1:.3f}초 (eps-이웃 조회 없음)")
        print(f"[Client] k_max 상한={k_max} (서버 enc_sum_k로 정밀화 예정)")
        print(f"         T({k_max})={k_max*(k_max+1)//2}  안전값={N//2}")
        return 'kd_dense', k_max, order, inv_perm
    else:
        num_sweeps = max(math.ceil(N / max(min_pts, 1) / 2), math.ceil(math.log2(N)))
        print(f"[Client] sweep (num_sweeps={num_sweeps})")
        return 'sweep', num_sweeps, np.arange(N), np.arange(N)


# ── 전파 방식 결정 ─────────────────────────────────────────────────────────

def decide_propagation_mode(min_pts: int, log2_n: int, N: int) -> tuple:
    """
    O(1): min_pts만으로 전파 방식 결정.
    prepare_client_ordering 사용 권장 (DFS Ball Tree + 구조 분석 k_max 포함).
    """
    if min_pts >= _KD_DENSE_MIN_PTS_THRESHOLD:
        k_max  = get_kd_dense_kmax(N)
        T_kmax = k_max * (k_max + 1) // 2
        print(f"[Client] min_pts={min_pts} ≥ {_KD_DENSE_MIN_PTS_THRESHOLD} "
              f"→ KD-dense  k_max={k_max}, T({k_max})={T_kmax}")
        return 'kd_dense', k_max
    else:
        num_sweeps = max(math.ceil(N / max(min_pts, 1) / 2), math.ceil(math.log2(N)))
        print(f"[Client] min_pts={min_pts} < {_KD_DENSE_MIN_PTS_THRESHOLD} "
              f"→ ALL-sweep (num_sweeps={num_sweeps}, chain형 허용)")
        return 'sweep', num_sweeps


# ── FHE 엔진 설정 ─────────────────────────────────────────────────────────

def setup_fhe_engine(verbose: bool = False):
    """FHE 엔진 및 KeyPack 생성. production/test 공통 사용."""
    engine     = desilofhe.Engine(use_bootstrap=True, mode="gpu")
    secret_key = engine.create_secret_key()

    def _create(name, fn):
        if verbose:
            before = _gpu_used_mb()
        key = fn(secret_key)
        if verbose:
            after = _gpu_used_mb()
            print(f"  [MEM] {name:<40}  delta={after-before:+.0f} MB  (used={after:.0f} MB)")
        return key

    if verbose:
        print(f"  [MEM] Engine 초기화  used={_gpu_used_mb():.0f} MB")

    pk  = _create("public_key",          engine.create_public_key)
    rk  = _create("rotation_key",        engine.create_rotation_key)
    rlk = _create("relinearization_key", engine.create_relinearization_key)
    ck  = _create("conjugation_key",     engine.create_conjugation_key)
    bk  = _create("bootstrap_key",       engine.create_bootstrap_key)
    sbk = _create("smallbootstrap_key",  engine.create_small_bootstrap_key)

    keypack = KeyPack(
        public_key=pk, rotation_key=rk, relinearization_key=rlk,
        conjugation_key=ck, bootstrap_key=bk, smallbootstrap_key=sbk,
    )

    if verbose:
        print(f"  [MEM] 모든 키 생성 완료  used={_gpu_used_mb():.0f} MB")

    return engine, secret_key, keypack


# ── Production 진입점 ─────────────────────────────────────────────────────

def run_client_dbscan_fhe(pts: list, eps: float, min_pts: int):
    """
    Production 클라이언트 진입점.

    흐름:
      1. 정규화 → [0, 1]
      2. prepare_client_ordering:
           - Ball Tree DFS in-order 정렬 (kd_dense 시)
           - 구조 분석 k_max 상한 계산 (eps-이웃 조회 없음)
      3. 암호화 → 서버 전송
         서버가 enc_sum_k로 k_max 정밀화 (클라이언트 상한 이내)
      4. 복호화 → inv_perm 적용 → 원래 순서 복원
    """
    start   = time()
    pts_arr = np.array(pts, dtype=np.float64)
    N       = len(pts_arr)
    dim     = pts_arr.shape[1]
    log2_n  = math.ceil(math.log2(N))

    engine, secret_key, keypack = setup_fhe_engine(verbose=False)

    # 1. 정규화
    g_min  = pts_arr.min()
    g_max  = pts_arr.max()
    scale  = (g_max - g_min) or 1.0
    norm   = (pts_arr - g_min) / scale
    ne     = eps / scale
    print(f"[Client] N={N}, dim={dim}, eps_norm={ne:.4f}")

    # 2. Ball Tree DFS 정렬 + k_max 구조 분석 (prepare_client_ordering 통합 사용)
    mode, k_max, heap_idx, inv_perm = prepare_client_ordering(norm, ne, min_pts, N)

    if mode == 'kd_dense':
        data_for_enc = norm[heap_idx].tolist()
        T_kmax = k_max * (k_max + 1) // 2
        print(f"[Client] Heap(DFS in-order) 정렬 완료  k_max={k_max}, T({k_max})={T_kmax}")
    else:
        data_for_enc = norm.tolist()
        print(f"[Client] 정렬 없음 (sweep 방식)")

    # 3. 암호화 → 서버
    encryptor = DimensionalEncryptor(engine, keypack)
    enc_cols  = encryptor.encrypt_data(data_for_enc, dim)

    enc_result, _ = send_to_server(
        engine=engine, keypack=keypack, secret_key=secret_key,
        encrypted_columns=enc_cols,
        num_points=N, eps=ne, min_pts=min_pts,
        k_max=k_max,                           # 클라이언트 구조 분석 상한 전달
        use_kd_propagation=(mode == 'kd_dense'),
        num_sweeps=k_max,
    )

    # 4. 복호화 + 역순열
    elapsed = time() - start
    print(f"[Client] 서버 완료 ({elapsed:.2f}초). 복호화 중...")

    heap_labels = np.real(engine.decrypt(enc_result, secret_key))[:N]
    # heap_labels[i]: DFS 위치 i = 원래 heap_idx[i]번 점의 라벨
    # original_labels[j] = heap_labels[inv_perm[j]]
    orig_labels = heap_labels[inv_perm]

    cluster_labels = []
    for x in orig_labels:
        r = round(float(x))
        if r <= 0:      cluster_labels.append(-1)
        elif r > N:     cluster_labels.append(N)
        else:           cluster_labels.append(r)

    n_c = len(set(l for l in cluster_labels if l != -1))
    print(f"[Client] 완료! 클러스터 {n_c}개 (노이즈: {cluster_labels.count(-1)}개)")
    return [list(pts[i]) + [cluster_labels[i]] for i in range(N)], cluster_labels
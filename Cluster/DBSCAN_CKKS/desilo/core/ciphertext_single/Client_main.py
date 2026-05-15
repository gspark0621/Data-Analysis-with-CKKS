# core/ciphertext_single/Client_main.py
#
# ── 변경사항 (KD-tree → Ball Tree) ──────────────────────────────────────────
#
# [변경 1] build_ball_tree_order: KD-tree 대신 Ball Tree BFS 정렬
#   KD-tree 문제:
#     축 정렬(직사각형) 분할 → eps-ball이 경계를 가로지르면
#     eps-이웃이 다른 subtree에 분산 → heap gap 증가
#     실측: hepta cluster 122 max_gap=56 > k_max=45 → propagation 실패
#
#   Ball Tree 개선:
#     구면 분할 (DBSCAN eps-이웃과 동일 geometry)
#     pole1 = centroid에서 가장 먼 점
#     pole2 = pole1에서 가장 먼 점
#     pole1→pole2 방향 투영 중앙값으로 분할
#     → eps-이웃이 같은 subtree에 집중 → heap gap 감소
#
# [변경 2] compute_kmax_from_data: 정확한 k_max 계산
#   k_max = max over ALL eps-neighbor pairs (i,j) of min(|i-j|, N-|i-j|)
#
#   근거:
#     adj_k[i] = dist(heap[i], heap[(i+k) mod N]) ≤ eps
#     pair (i,j): forward stride=j-i, backward stride=N-(j-i)
#     min(j-i, N-j+i) ≤ k_max → 1 sweep으로 반드시 연결됨
#
#   KD-tree 안전 상한(N//2=106) 대신 실측값(예상 20~40) 사용
#   → Label Propagation fhe_max 2~5배 감소
#
# [변경 3] prepare_client_ordering: Ball Tree + k_max 통합 인터페이스
#   (mode, k_max, heap_idx, inv_perm) 반환
#
# [유지] backward-compat alias: build_kd_tree_order, get_kd_dense_kmax,
#        decide_propagation_mode (test 스크립트 호환)
# ────────────────────────────────────────────────────────────────────────────

from time import time
import math
import numpy as np
import desilofhe
import pynvml
from sklearn.neighbors import BallTree as SkBallTree
from core.ciphertext_single.EncryptModule import DimensionalEncryptor
from core.ciphertext_single.Server_main import send_to_server_fhe as send_to_server
from util.keypack import KeyPack


def _gpu_used_mb() -> float:
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0


# ── Ball Tree (BFS level-order) 정렬 ─────────────────────────────────────────

def build_ball_tree_order(pts: np.ndarray) -> tuple:
    """
    Ball Tree를 Heap(BFS level-order) 레이아웃으로 정렬.

    분할 방식 (KD-tree와 비교):
      KD-tree:   split_axis = argmax(variance)
                 split_val  = coord[axis] 중앙값 (직사각형 분할)

      Ball Tree: pole1 = centroid에서 가장 먼 점
                 pole2 = pole1에서 가장 먼 점
                 split  = pole1→pole2 방향 투영값 중앙값 (구면 분할)

    DBSCAN eps-이웃(구) ↔ Ball Tree 분할(구면): 동일 geometry
    → eps-이웃이 같은 subtree에 집중될 가능성 높음
    → heap gap 감소 → k_max 감소 → Label Propagation fhe_max 감소

    BFS heap 레이아웃:
      index 0:         root
      index 2i+1, 2i+2: left/right child of node i

    Returns
    -------
    heap_indices : np.ndarray[int, N]
        heap_indices[i] = heap 위치 i에 오는 원래 데이터 인덱스
    inv_perm : np.ndarray[int, N]
        역순열. original_labels[j] = heap_labels[inv_perm[j]]
    """
    pts_arr = np.array(pts, dtype=np.float64)
    N       = len(pts_arr)
    heap    = np.full(N, -1, dtype=int)

    def _build(indices: np.ndarray, h_idx: int):
        if len(indices) == 0 or h_idx >= N:
            return
        if len(indices) == 1:
            heap[h_idx] = indices[0]
            return
        if len(indices) == 2:
            heap[h_idx] = indices[0]
            _build(indices[1:2], 2 * h_idx + 1)
            return

        coords = pts_arr[indices]

        # ── Ball Tree 분할: pole1→pole2 축 투영 ──────────────────────────
        # pole1: centroid에서 가장 먼 점
        centroid      = coords.mean(axis=0)
        d_sq_centroid = np.einsum('ij,ij->i', coords - centroid, coords - centroid)
        pole1         = coords[int(np.argmax(d_sq_centroid))]

        # pole2: pole1에서 가장 먼 점
        d_sq_pole1 = np.einsum('ij,ij->i', coords - pole1, coords - pole1)
        pole2      = coords[int(np.argmax(d_sq_pole1))]

        # pole1→pole2 단위 벡터
        axis    = pole2 - pole1
        ax_norm = float(np.dot(axis, axis)) ** 0.5

        if ax_norm < 1e-12:
            # degenerate (모든 점이 동일) → 인덱스 순서
            order = np.arange(len(indices))
        else:
            axis_unit   = axis / ax_norm
            projections = (coords - pole1) @ axis_unit
            order       = np.argsort(projections, kind='stable')

        mid         = len(indices) // 2
        heap[h_idx] = indices[order[mid]]

        _build(indices[order[:mid]],      2 * h_idx + 1)   # left subtree
        _build(indices[order[mid + 1:]], 2 * h_idx + 2)   # right subtree

    _build(np.arange(N), 0)

    # 미배치 슬롯 채우기 (N이 2의 거듭제곱이 아닐 때 드물게 발생)
    missing = np.where(heap == -1)[0]
    if len(missing) > 0:
        placed   = set(int(x) for x in heap[heap != -1])
        unplaced = list(set(range(N)) - placed)
        for slot, orig_idx in zip(missing, unplaced):
            heap[slot] = orig_idx

    inv_perm       = np.empty(N, dtype=int)
    inv_perm[heap] = np.arange(N)
    return heap, inv_perm


# ── 정확한 k_max 계산 ─────────────────────────────────────────────────────────

def compute_kmax_from_data(
    pts_sorted: np.ndarray,
    eps_norm: float,
    N: int,
) -> int:
    """
    eps-이웃 쌍의 heap 위치 최대 유효 gap → 정확한 k_max.

    수학적 근거:
      FHE adj_k[i] = (dist(heap[i], heap[(i+k) mod N]) ≤ eps)
      pair (i,j), gap=|i-j|:
        forward:  stride=gap   → adj_{gap}[min(i,j)] 커버
        backward: stride=N-gap → adj_{N-gap}[max(i,j)] 커버
        필요한 최소 stride = min(gap, N-gap)

      k_max = max over all eps-neighbor pairs of  min(|i-j|, N-|i-j|)

      이 k_max로 1 forward+backward sweep → 모든 eps-이웃 연결 보장

    시간: O(N × avg_neighbors)  (sklearn BallTree 사용)
    공간: O(N)

    Parameters
    ----------
    pts_sorted : Ball Tree BFS 순서로 정렬된 데이터 (이미 정규화됨)
    eps_norm   : 정규화된 eps
    N          : 점 개수
    """
    bt           = SkBallTree(pts_sorted)
    indices_list = bt.query_radius(pts_sorted, r=eps_norm)

    k_max = 0
    for i, neighbors in enumerate(indices_list):
        for j in map(int, neighbors):
            if j == i:
                continue
            gap           = abs(i - j)
            effective_gap = min(gap, N - gap)   # circular symmetry
            if effective_gap > k_max:
                k_max = effective_gap

    return min(k_max, N // 2)


# ── Ordering + k_max 통합 준비 ───────────────────────────────────────────────

_BALL_TREE_MIN_PTS_THRESHOLD = 4


def prepare_client_ordering(
    norm_pts: np.ndarray,
    eps_norm: float,
    min_pts: int,
    N: int,
) -> tuple:
    """
    Ball Tree BFS 정렬 + 정확한 k_max 계산.

    min_pts >= 4 → Ball Tree 정렬 + compute_kmax_from_data
      - client는 원본 데이터 보유 → plaintext 계산 가능 (Privacy 시나리오)
      - k_max: KD-tree 안전 상한(N//2) 대신 실측값 → Label Prop 시간 단축

    min_pts < 4 → 정렬 없음, sweep 방식
      - chain형 cluster 허용

    Returns
    -------
    mode     : 'kd_dense' | 'sweep'
    k_max    : dense stride 상한 (kd_dense) 또는 sweep 횟수 (sweep)
    heap_idx : np.ndarray[int, N]
    inv_perm : np.ndarray[int, N]
    """
    if min_pts >= _BALL_TREE_MIN_PTS_THRESHOLD:
        t0 = time()
        heap_idx, inv_perm = build_ball_tree_order(norm_pts)
        t_tree = time() - t0
        print(f"[Client] Ball Tree 구축: {t_tree:.3f}초")

        pts_sorted = norm_pts[heap_idx]

        t1 = time()
        k_max = compute_kmax_from_data(pts_sorted, eps_norm, N)
        t_kmax = time() - t1
        print(f"[Client] k_max 측정: {t_kmax:.3f}초")

        T_kmax    = k_max * (k_max + 1) // 2
        safe_kmax = N // 2
        reduction = (safe_kmax * 4) // max(k_max * 4, 1)
        print(f"[Client] k_max={k_max} (KD-tree 안전값={safe_kmax}), T({k_max})={T_kmax}")
        print(f"[Client] fhe_max: {k_max*4}회 (KD-tree {safe_kmax*4}회 대비 {reduction}배↓)")
        return 'kd_dense', k_max, heap_idx, inv_perm
    else:
        num_sweeps = max(
            math.ceil(N / max(min_pts, 1) / 2),
            math.ceil(math.log2(N))
        )
        print(f"[Client] min_pts={min_pts} < {_BALL_TREE_MIN_PTS_THRESHOLD}"
              f" → sweep (num_sweeps={num_sweeps}, chain형 허용)")
        return 'sweep', num_sweeps, np.arange(N), np.arange(N)


# ── 하위 호환성 alias (기존 test 스크립트 호환) ───────────────────────────────

def build_kd_tree_order(pts: np.ndarray) -> tuple:
    """Deprecated: build_ball_tree_order 사용 권장."""
    print("[WARNING] build_kd_tree_order deprecated → build_ball_tree_order 사용")
    return build_ball_tree_order(pts)


def get_kd_dense_kmax(N: int) -> int:
    """Deprecated: compute_kmax_from_data 사용 권장. 안전 상한 반환."""
    return N // 2


def decide_propagation_mode(min_pts: int, log2_n: int, N: int) -> tuple:
    """Deprecated: prepare_client_ordering 사용 권장.
    eps 정보 없이 호출 시 k_max=N//2 (안전 상한) 반환."""
    if min_pts >= _BALL_TREE_MIN_PTS_THRESHOLD:
        k_max = N // 2
        print(f"[Client] (legacy) kd_dense, k_max={k_max} (안전 상한, eps 미전달)")
        return 'kd_dense', k_max
    else:
        num_sweeps = max(math.ceil(N / max(min_pts, 1) / 2), math.ceil(math.log2(N)))
        return 'sweep', num_sweeps


# ── FHE 엔진 설정 (단일 출처) ─────────────────────────────────────────────────

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


# ── Production 진입점 ─────────────────────────────────────────────────────────

def run_client_dbscan_fhe(pts: list, eps: float, min_pts: int):
    """
    Production 클라이언트 진입점.

    흐름:
      1. 정규화 → [0, 1]
      2. Ball Tree 정렬 + k_max 측정  (prepare_client_ordering)
      3. 암호화 → 서버 전송
      4. 복호화 → inv_perm 역순열 → 원래 순서 복원
    """
    start   = time()
    pts_arr = np.array(pts, dtype=np.float64)
    N       = len(pts_arr)
    dim     = pts_arr.shape[1]

    engine, secret_key, keypack = setup_fhe_engine(verbose=False)

    # 1. 정규화
    g_min = pts_arr.min()
    g_max = pts_arr.max()
    scale = (g_max - g_min) or 1.0
    norm  = (pts_arr - g_min) / scale
    ne    = eps / scale
    print(f"[Client] N={N}, dim={dim}, eps_norm={ne:.4f}")

    # 2. Ball Tree 정렬 + k_max 측정
    mode, k_max, heap_idx, inv_perm = prepare_client_ordering(norm, ne, min_pts, N)

    # 3. 정렬된 순서로 암호화
    data_for_enc = norm[heap_idx].tolist() if mode == 'kd_dense' else norm.tolist()

    encryptor = DimensionalEncryptor(engine, keypack)
    enc_cols  = encryptor.encrypt_data(data_for_enc, dim)

    enc_result, _ = send_to_server(
        engine=engine, keypack=keypack, secret_key=secret_key,
        encrypted_columns=enc_cols,
        num_points=N, eps=ne, min_pts=min_pts,
        k_max=k_max,
        use_kd_propagation=(mode == 'kd_dense'),
        num_sweeps=k_max,
    )

    # 4. 복호화 + 역순열
    elapsed = time() - start
    print(f"[Client] 서버 완료 ({elapsed:.2f}초). 복호화 중...")

    heap_labels = np.real(engine.decrypt(enc_result, secret_key))[:N]
    orig_labels = heap_labels[inv_perm]   # heap 순서 → 원래 순서

    cluster_labels = []
    for x in orig_labels:
        r = round(float(x))
        if r <= 0:          cluster_labels.append(-1)
        elif r > N:         cluster_labels.append(N)
        else:               cluster_labels.append(r)

    n_c = len(set(l for l in cluster_labels if l != -1))
    print(f"[Client] 완료! 클러스터 {n_c}개 (노이즈: {cluster_labels.count(-1)}개)")
    return [list(pts[i]) + [cluster_labels[i]] for i in range(N)], cluster_labels
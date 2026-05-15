# core/ciphertext_single/Client_main.py
#
# ── 변경사항 ────────────────────────────────────────────────────────────────
# [변경 1] build_kd_tree_order: Heap(BFS level-order) KD-tree
#   - root(index 0) = 전체 중앙값
#   - left child(2i+1), right child(2i+2) = 재귀 분할
#   - depth d cross-boundary stride = 2^(d-1)  (설계 의도와 일치)
#   - DFS in-order와 달리 모든 depth의 stride가 명확히 2^(d-1)로 분리됨
#
# [변경 2] decide_propagation_mode: O(1), BallTree 계산 없음
#   - DBSCAN 정의상 min_pts = 모든 core point의 최소 이웃 수 보장
#   - "avg 편향" 문제 원천 제거: avg 계산 자체가 불필요
#   - min_pts ≥ 4 → kd_depth (doubling 보장)
#   - min_pts < 4 → sweep (chain형 허용)
#   - num_sweeps = N // min_pts // 2 (보수적 상한)
#
# [변경 3] run_client_dbscan_fhe: Heap 정렬 + inv_perm 역순열 적용
# ────────────────────────────────────────────────────────────────────────────

from time import time
import math
import numpy as np
import desilofhe
import pynvml
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


# ── Heap(BFS level-order) KD-tree 정렬 ───────────────────────────────────

def build_kd_tree_order(pts: np.ndarray) -> tuple:
    """
    KD-tree를 Heap(BFS level-order) 레이아웃으로 정렬.

    레이아웃:
      index 0:       root (전체 중앙값)
      index 1, 2:    depth-1 left/right subtree root
      index 3~6:     depth-2 subtree roots
      index 2^d-1 ~ 2^(d+1)-2: depth-d nodes

    핵심 특성:
      depth d cross-boundary stride = 2^(d-1)
      → bottom-up strides = [1, 2, 4, 8, ..., N//2]  (2의 거듭제곱)
      → depth(i) = floor(log₂(i+1)): N에만 의존, data-independent → plaintext 무방

    DFS in-order와 차이:
      DFS: cross-boundary stride 모두 stride≈1 근방에 몰림 (큰 stride 무의미)
      Heap: depth d마다 stride가 2^(d-1)로 명확히 분리됨

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

        # 분산 최대 축 기준 정렬 (max-variance axis split)
        coords = pts_arr[indices]
        axis   = int(np.argmax(np.var(coords, axis=0)))
        order  = np.argsort(coords[:, axis], kind='stable')
        mid    = len(indices) // 2

        # 중앙값 → 현재 heap 위치 배치 (internal node)
        heap[h_idx] = indices[order[mid]]

        # 좌/우 subtree 재귀
        _build(indices[order[:mid]],    2 * h_idx + 1)   # left child
        _build(indices[order[mid+1:]], 2 * h_idx + 2)   # right child

    _build(np.arange(N), 0)

    # 누락 검증 (N이 2의 거듭제곱이 아닌 경우 일부 미배치 가능)
    missing = np.where(heap == -1)[0]
    if len(missing) > 0:
        # 미배치 점을 빈 슬롯에 채움 (드문 경우)
        unplaced = list(set(range(N)) - set(heap[heap != -1].tolist()))
        for slot, orig_idx in zip(missing, unplaced):
            heap[slot] = orig_idx

    inv_perm           = np.empty(N, dtype=int)
    inv_perm[heap]     = np.arange(N)
    return heap, inv_perm


def get_kd_dense_kmax(N: int) -> int:
    """
    dense stride 상한 k_max 계산.

    FHE sequential T(k_max) 정리:
      k=1→2→...→k_max 순서로 처리 시 T(k_max)=k_max(k_max+1)/2 위치까지 전파
      k_max = min(N//2, 3×ceil(√N)) → T(k_max) >> N → 1 sweep 완전 수렴 보장

    vs depth_strides [1,2,4,...,64] (8개):
      power-of-2만 체크 → stride=3,5,7,... 영구 누락 → 비구형 cluster 실패

    dense k=1..k_max (45개, N=212):
      모든 stride 체크 → 비구형 cluster 포함 완전 수렴 ✓
      T(45)=1035 >> N=212 → 1 forward sweep으로 임의 cluster 수렴 ✓

    N=212: k_max=45, T(45)=1035 >> 212
    """
    return max(N // 2, 3 * math.ceil(math.sqrt(N)))


# ── 전파 방식 결정 (O(1), BallTree 불필요) ──────────────────────────────

_KD_DENSE_MIN_PTS_THRESHOLD = 4


def decide_propagation_mode(min_pts: int, log2_n: int, N: int) -> tuple:
    """
    O(1): BallTree 쿼리 없이 min_pts만으로 전파 방식 결정.

    min_pts ≥ 4 → kd_dense (KD-tree + dense k=1..k_max)
      - k_max = min(N//2, 3×ceil(√N))
      - T(k_max) >> N → 1 sweep 완전 수렴 보장
      - dense stride → stride=3,5,7,... 연결도 커버 (power-of-2 문제 해결)

    min_pts < 4 → sweep (chain형 허용)
      - num_sweeps = max(N//min_pts//2, ceil(log2N))

    Returns
    -------
    mode    : 'kd_dense' | 'sweep'
    k_max   : dense stride 상한 (kd_dense 시)
              = num_sweeps (sweep 방식 시)
    """
    if min_pts >= _KD_DENSE_MIN_PTS_THRESHOLD:
        k_max = get_kd_dense_kmax(N)
        T_kmax = k_max * (k_max + 1) // 2
        print(f"[Client] min_pts={min_pts} ≥ {_KD_DENSE_MIN_PTS_THRESHOLD} "
              f"→ KD-dense 선택  k_max={k_max}, T({k_max})={T_kmax}")
        return 'kd_dense', k_max
    else:
        num_sweeps = max(math.ceil(N / max(min_pts, 1) / 2), math.ceil(math.log2(N)))
        print(f"[Client] min_pts={min_pts} < {_KD_DENSE_MIN_PTS_THRESHOLD} "
              f"→ ALL-sweep 선택 (num_sweeps={num_sweeps}, chain형 허용)")
        return 'sweep', num_sweeps


# ── FHE 엔진 설정 (단일 출처) ────────────────────────────────────────────

def setup_fhe_engine(verbose: bool = False):
    """
    FHE 엔진 및 KeyPack 생성.
    production/test 공통 사용 (단일 출처 원칙).

    Returns: engine, secret_key, keypack
    """
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


# ── Production 진입점 ──────────────────────────────────────────────────────

def run_client_dbscan_fhe(pts: list, eps: float, min_pts: int):
    """
    Production 클라이언트 진입점.

    흐름:
      1. 정규화 → [0, 1]
      2. decide_propagation_mode (O(1), min_pts 기반)
      3. kd_depth 선택 시: Heap KD-tree 정렬
         sweep 선택 시: 원래 순서 유지
      4. 암호화 → 서버 전송
      5. 복호화 → inv_perm 적용 → 원래 순서 복원
    """
    start    = time()
    pts_arr  = np.array(pts, dtype=np.float64)
    N        = len(pts_arr)
    dim      = pts_arr.shape[1]
    log2_n   = math.ceil(math.log2(N))

    engine, secret_key, keypack = setup_fhe_engine(verbose=False)

    # 1. 정규화
    g_min  = pts_arr.min()
    g_max  = pts_arr.max()
    scale  = (g_max - g_min) or 1.0
    norm   = (pts_arr - g_min) / scale
    ne     = eps / scale
    print(f"[Client] N={N}, dim={dim}, eps_norm={ne:.4f}")

    # 2. 전파 방식 결정 (O(1))
    mode, k_max = decide_propagation_mode(min_pts, log2_n, N)

    # 3. 정렬
    if mode == 'kd_dense':
        heap_idx, inv_perm = build_kd_tree_order(norm)
        data_for_enc       = norm[heap_idx].tolist()
        T_kmax = k_max * (k_max + 1) // 2
        print(f"[Client] Heap 정렬 완료  k_max={k_max}, T({k_max})={T_kmax}")
    else:
        heap_idx   = np.arange(N)
        inv_perm   = np.arange(N)
        data_for_enc = norm.tolist()
        print(f"[Client] 정렬 없음 (sweep 방식)")

    # 4. 암호화 → 서버
    encryptor  = DimensionalEncryptor(engine, keypack)
    enc_cols   = encryptor.encrypt_data(data_for_enc, dim)

    enc_result, _ = send_to_server(
        engine=engine, keypack=keypack, secret_key=secret_key,
        encrypted_columns=enc_cols,
        num_points=N, eps=ne, min_pts=min_pts,
        k_max=k_max,
        use_kd_propagation=(mode == 'kd_dense'),
        num_sweeps=k_max,
    )

    # 5. 복호화 + 역순열
    elapsed = time() - start
    print(f"[Client] 서버 완료 ({elapsed:.2f}초). 복호화 중...")

    heap_labels  = np.real(engine.decrypt(enc_result, secret_key))[:N]
    # heap_labels[i]: heap 위치 i의 라벨 (= 원래 heap_idx[i] 점의 라벨)
    # original_labels[j] = heap_labels[inv_perm[j]]
    orig_labels  = heap_labels[inv_perm]

    cluster_labels = []
    for x in orig_labels:
        r = round(float(x))
        if r <= 0:           cluster_labels.append(-1)
        elif r > N:          cluster_labels.append(N)
        else:                cluster_labels.append(r)

    n_c = len(set(l for l in cluster_labels if l != -1))
    print(f"[Client] 완료! 클러스터 {n_c}개 (노이즈: {cluster_labels.count(-1)}개)")
    return [list(pts[i]) + [cluster_labels[i]] for i in range(N)], cluster_labels
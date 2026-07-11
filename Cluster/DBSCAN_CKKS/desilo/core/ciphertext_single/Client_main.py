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
def _kd_dense_threshold(dim: int) -> int:
    """
    Sander et al. (1998) 권장 min_pts = 2 × dim.
    dim차원 공간에서 밀집 구조를 신뢰할 수 있는 최소 이웃 수.
    """
    return 2 * dim

def _gpu_used_mb() -> float:
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0


# ── Ball Tree DFS in-order 정렬 ───────────────────────────────────────────

# ★ 초기 cluster label 방식: 'depth_asc'(사용자 8번) | 'baseline'(대조)
_INIT_LABEL_SCHEME = 'depth_asc'


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
    depth   = np.zeros(N, dtype=int)          # ★ heap 위치별 트리 깊이 (라벨 생성용)

    def _build(indices: np.ndarray, start: int, d: int):
        if len(indices) == 0:
            return
        if len(indices) == 1:
            order[start] = indices[0]
            depth[start] = d
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
        _build(indices[proj_order[:mid]],      start,        d + 1)   # left
        order[start + mid] = indices[proj_order[mid]]                 # root
        depth[start + mid] = d                                        # ★ in-order root 깊이
        _build(indices[proj_order[mid + 1:]], start + mid + 1, d + 1) # right

    _build(np.arange(N), 0, 0)

    inv_perm        = np.empty(N, dtype=int)
    inv_perm[order] = np.arange(N)
    return order, inv_perm, depth          # ★ depth 추가 반환


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


# ── 밀도 지표 + 초기 cluster label 생성 (client 측, ball-tree 1회 구축 재사용) ──

def compute_density_and_mode(norm_pts, order, eps_norm, k_max, N, dim,
                             min_pts):
    """전파 그래프의 '지름'으로 mode/round 결정. client 평문 분석(서버엔 스칼라만).

    ★ 핵심 발견 (hepta/lsun/tetra/two_moons/chain 2D·3D 측정):
      수렴 round 는 밀도(이웃수)나 dim 이 아니라 **전파 그래프 지름**(클러스터를
      가로지르는 최장 최단경로 hop)이 결정한다. 지름이 dim·밀도·모양 효과를 모두 흡수.
      회귀: round ≈ 지름/3.8.  안전식: round = ceil(지름/4) + 2  (5개 데이터셋 전부 커버)
        측정: hepta 지름6→2, lsun 11→2, tetra 8→2, two_moons 33→6, chain 71→20
        예측: ceil(d/4)+2 = 4,5,4,11,20 (모두 실제 이상)

    mode 구분: 지름이 너무 크면(전파 비현실적) all_sweep. 경계는 보수적으로 잡되,
      kd_dense round 상한(예: 12)을 넘기면 all_sweep 권장.

    Returns: mode('kd_dense'|'all_sweep'), n_rounds(int|None), diameter(int)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    eps2 = eps_norm * eps_norm
    P = norm_pts[order]                       # heap 순서
    # core mask (전체 stride 이웃수 = pairwise)
    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(2)
    nbr = (D2 <= eps2).sum(1)                 # self 포함
    core = (nbr >= min_pts)

    # 전파 그래프(<=k_max, core-core) 간선
    rows, cols = [], []
    for k in range(1, k_max + 1):
        for i in range(N):
            j = (i + k) % N
            if D2[i, j] <= eps2 and core[i] and core[j]:
                rows += [i, j]; cols += [j, i]
    if not rows:
        return 'all_sweep', None, 0
    G = csr_matrix(([1] * len(rows), (rows, cols)), shape=(N, N))

    # 각 연결성분의 지름 (최장 최단경로 hop). core 노드 기준.
    core_idx = np.where(core)[0]
    sp = shortest_path(G, indices=core_idx, directed=False)[:, core_idx]
    sp[np.isinf(sp)] = 0
    diameter = int(sp.max())

    # round = ceil(지름/4) + 2.  상한 넘으면 all_sweep.
    ROUND_CAP = 12
    n_rounds = int(np.ceil(diameter / 4.0) + 2)
    if n_rounds > ROUND_CAP:
        return 'all_sweep', None, diameter      # chain류: 전파 비현실적
    return 'kd_dense', max(n_rounds, 2), diameter


def build_initial_labels(depth, N, scheme='depth_asc'):
    """초기 cluster label 생성 (heap 순서). client가 tree depth로 생성 → 암호화해 전송.

    scheme:
      'baseline'   : 1..N (heap 위치+1). 인접 slot 라벨차 1/N.
      'depth_asc'  : depth 오름차순(루트→리프) 1..N 부여. 예 [4,2,5,1,6,3,7].
                     인접 slot 라벨차 ≥ 2 → max 오차 강건 가설. 1..N 순열이라
                     label_scale=1.1N 그대로 유효(init_label_max=N).

    Returns: labels(heap순, 1..N), init_label_max(float)
    """
    if scheme == 'baseline':
        return np.array([float(i + 1) for i in range(N)]), float(N)
    if scheme == 'depth_asc':
        od = np.lexsort((np.arange(N), depth))     # depth 우선, 같으면 heap 위치순
        labels = np.empty(N, dtype=float)
        labels[od] = np.arange(1, N + 1)
        return labels, float(N)
    raise ValueError(f"unknown scheme: {scheme}")


def gap_decode_labels(heap_labels, N, tau=1.0):
    """압축된 FHE 라벨을 gap 기반으로 디코딩 (τ=1 고정).

    배경: FHE 전파 후 라벨은 클러스터별로 압축되어 좁은 띠를 이룸(예 138.16~138.83).
      round()/floor() 같은 고정 정수 경계는 이 띠가 경계(x.5)를 관통하면 한 클러스터를
      둘로 가름 (실측 LSUN: 138/139 분리 → ARI 0.97). 노이즈가 양/음 양방향이라
      어떤 고정 경계도 본질적으로 취약.

    해결: 절대 위치가 아니라 '이웃 값과의 거리(gap)'로 판정.
      라벨을 오름차순 정렬한 뒤, 인접 값의 차이가 tau(=1)를 넘으면 클러스터 경계.
      전제: FHE 오차로 인한 클러스터 내부폭 < 1 < 클러스터간 라벨 간격.
        - 내부폭 < 1: depth_asc 라벨이 정수 단위라 같은 클러스터는 1 미만으로 압축됨
          (실측 LSUN 내부 최대 gap 0.47, class간 최소 간격 2.0+).
        - 한 클러스터 내 점이 1.0 이상 떨어지는 일은 압축이 연속 띠를 만들어 발생 안 함.

    Parameters
    ----------
    heap_labels : np.ndarray  복호화된 라벨 (값만 사용, 순서 무관)
    tau         : float       분리 임계. 인접 정렬값 차이가 tau 초과면 새 클러스터.
                              FHE 오차 < 1 전제에서 tau=1.0이 자연값.

    Returns
    -------
    cluster_ids : np.ndarray[int]  0,1,2,... 클러스터 ID, noise는 -1
    """
    lab = np.asarray(heap_labels, dtype=float)
    out = np.full(len(lab), -1, dtype=int)
    valid = np.where(lab > 0.5)[0]                  # 양의 라벨만 (0/음수=noise)
    if len(valid) == 0:
        return out
    sv = valid[np.argsort(lab[valid])]              # 값 오름차순 인덱스
    cur = 0
    out[sv[0]] = cur
    for a, b in zip(sv[:-1], sv[1:]):
        if lab[b] - lab[a] > tau:                   # 인접 차이가 tau 초과 → 새 클러스터
            cur += 1
        out[b] = cur
    return out

# ── deprecated aliases ────────────────────────────────────────────────────

def build_kd_tree_order(pts: np.ndarray) -> tuple:
    """Deprecated alias → build_ball_tree_order (DFS in-order).
    주의: build_ball_tree_order는 (order, inv_perm, depth) 3개 반환."""
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
    dim: int,       # ← 인자 추가
) -> tuple:
    # ★ [2026-05c] kd_dense 통일.
    #   이전: min_pts >= 2×dim 이면 kd_dense, 아니면 sweep.
    #   문제: all-sweep은 mask damping이 누적 곱셈으로 라벨을 0으로 소멸시켜
    #         iris 등에서 -1 완전 붕괴. hepta도 dim=3,min_pts=4 → 2×dim=6 미만이라
    #         의도치 않게 sweep으로 빠짐 (관측된 버그).
    #   결정: 작업 A(mask=1.0) + 작업 B(log₂N round)로 kd_dense가 모든 경우 커버.
    #         sweep 폐기. mode는 항상 'kd_dense'.
    #         (min_pts < 2×dim 인 희소 데이터도 kd_dense + 적응적 round로 처리;
    #          작업 C 검증에서 min_pts=4 데이터들이 kd_dense로 수렴 확인.)
    threshold = _kd_dense_threshold(dim)  # 2 × dim (참고용 로그만)

    # ── ball-tree 1회 구축: order, inv_perm, depth 동시 획득 ──
    t0 = time()
    order, inv_perm, depth = build_ball_tree_order(norm_pts)   # ★ depth 추가
    print(f"[Client] Ball Tree DFS 구축: {time()-t0:.3f}초")

    t1 = time()
    k_max = compute_kmax_from_ball_structure(norm_pts, order, eps_norm, N)
    print(f"[Client] k_max 구조 분석 (DFS): {time()-t1:.3f}초 (eps-이웃 조회 없음)")
    print(f"[Client] k_max 상한={k_max}  T({k_max})={k_max*(k_max+1)//2}  안전값={N//2}")

    # ── 밀도 기반 mode/round 결정 (client 평문 분석; 서버엔 스칼라만 전달) ──
    mode, n_rounds, diameter = compute_density_and_mode(
        norm_pts, order, eps_norm, k_max, N, dim, min_pts)
    if mode == 'kd_dense':
        print(f"[Client] 전파 방식: KD-dense  (전파그래프 지름={diameter}) "
              f"→ n_rounds=⌈{diameter}/4⌉+2={n_rounds}")
    else:
        print(f"[Client] 전파 방식: all-sweep (지름={diameter} 과대 → num_sweeps=k_max)")

    # ── 초기 cluster label 생성 (heap 순서, depth 기반) ──
    init_labels_heap, init_label_max = build_initial_labels(depth, N, scheme=_INIT_LABEL_SCHEME)
    print(f"[Client] 초기 라벨 scheme='{_INIT_LABEL_SCHEME}' "
          f"범위[{init_labels_heap.min():.0f},{init_labels_heap.max():.0f}] max={init_label_max:.0f}")

    return mode, k_max, order, inv_perm, n_rounds, init_labels_heap, init_label_max


# ── 전파 방식 결정 ─────────────────────────────────────────────────────────

def decide_propagation_mode(min_pts: int, log2_n: int, N: int, dim: int) -> tuple:
    """
    [DEPRECATED + 2026-05c kd_dense 통일]
      현재 어디서도 호출되지 않음 (prepare_client_ordering이 통합 담당).
      sweep 폐기에 맞춰 항상 'kd_dense' 반환하도록 정리.
      향후 완전 제거 예정.
    """
    threshold = _kd_dense_threshold(dim)  # 2 × dim (참고용)
    k_max  = get_kd_dense_kmax(N)
    T_kmax = k_max * (k_max + 1) // 2
    print(f"[Client][deprecated] kd_dense 통일 → k_max={k_max}, T({k_max})={T_kmax} "
          f"(min_pts={min_pts}, 2×dim={threshold})")
    return 'kd_dense', k_max


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

    # 2. Ball Tree DFS 정렬 + k_max 구조 분석 + 밀도/지름 기반 mode·round + 초기라벨
    #    ★ prepare_client_ordering은 7개 반환 (test 하니스와 동일 인터페이스)
    (mode, k_max, heap_idx, inv_perm,
     n_rounds_client, init_labels_heap, init_label_max) = prepare_client_ordering(
        norm, ne, min_pts, N, dim)

    # ★ kd_dense / all_sweep 모두 heap 정렬 데이터 사용 (전파 방식만 다름)
    data_for_enc = norm[heap_idx].tolist()
    if mode == 'kd_dense':
        T_kmax = k_max * (k_max + 1) // 2
        print(f"[Client] Heap(DFS in-order) 정렬 완료  k_max={k_max}, "
              f"T({k_max})={T_kmax}, n_rounds={n_rounds_client}")
    else:
        print(f"[Client] all_sweep 방식 (num_sweeps={k_max})")

    # 3. 암호화 → 서버
    encryptor = DimensionalEncryptor(engine, keypack)
    enc_cols  = encryptor.encrypt_data(data_for_enc, dim)

    # ★ 초기 cluster label 암호화 (client depth 생성, heap 순서)
    slot_count = engine.slot_count
    enc_init_labels = engine.encrypt(
        init_labels_heap.tolist() + [0.0] * (slot_count - N),
        keypack.public_key,
    )

    enc_result, _ = send_to_server(
        engine=engine, keypack=keypack, secret_key=secret_key,
        encrypted_columns=enc_cols,
        num_points=N, eps=ne, min_pts=min_pts,
        k_max=k_max,                                # 클라이언트 구조 분석 상한 전달
        use_kd_propagation=(mode == 'kd_dense'),
        num_sweeps=k_max if mode == 'all_sweep' else None,  # ★ 'all_sweep' (오타 수정)
        n_rounds=n_rounds_client,                   # ★ 지름 기반 (log₂N 하드코딩 폐기)
        enc_init_labels=enc_init_labels,            # ★ depth_asc 초기라벨 전달
        init_label_max=init_label_max,              # ★ label_scale 조정
    )

    # 4. 복호화 + 역순열
    elapsed = time() - start
    print(f"[Client] 서버 완료 ({elapsed:.2f}초). 복호화 중...")

    heap_labels = np.real(engine.decrypt(enc_result, secret_key))[:N]
    # heap_labels[i]: DFS 위치 i = 원래 heap_idx[i]번 점의 라벨
    # original_labels[j] = heap_labels[inv_perm[j]]
    orig_labels = heap_labels[inv_perm]

    # ★ gap 기반 디코딩 (데이터 적응형 τ). round()는 압축된 라벨 띠가 정수 경계를
    #   관통할 때 한 클러스터를 둘로 가름 (실측 LSUN ARI 0.97 → gap으로 1.0).
    cluster_labels = gap_decode_labels(orig_labels, N).tolist()

    n_c = len(set(l for l in cluster_labels if l != -1))
    print(f"[Client] 완료! 클러스터 {n_c}개 (노이즈: {cluster_labels.count(-1)}개)")
    return [list(pts[i]) + [cluster_labels[i]] for i in range(N)], cluster_labels
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
#
# [변경 6] ★ [2026-07] 정렬: Ball Tree DFS → PCA(제1주성분) 사영 정렬
#   문제: Ball Tree pole-to-pole 분할은 확산/대칭 구조에서 클러스터를 슬롯
#         양극단으로 흩어 k_max ≈ N//2 최악화.
#         실측 k_max: lsun 199, tetra 199, twodiamonds 399 (= N//2 상한 근접)
#   해결: PCA(PC1) 정렬 + 1-Lipschitz ε-window k_max
#         실측 k_max: hepta 70, lsun 96, tetra 75, twodiamonds 63
#         → LP fhe_max 호출 1.5~6.3배 감소 (호출수 = (2·n_rounds+2)·k_max·2)
#   검증: 평문 미러(Normalize→Core→LP 동일 로직)에서 4개 데이터셋 ARI=1.0000
#         (sklearn DBSCAN 일치). k_max 상한 안전성: oracle 대비 1.01~1.21배.
#   신규 함수: build_pca_order, compute_kmax_from_pca_window
#   유지: build_ball_tree_order / compute_kmax_from_ball_structure (비교·폴백용)
# ────────────────────────────────────────────────────────────────────────────

# ★ [2026-07] LP Core-Core 반복 횟수. 이전 math.ceil(log2(N))에서 고정 상수로 교체.
#
#   [log₂N 폐기 이유 — 실측 반증]
#     log₂N은 수렴 보장이 아니라 휴리스틱이었고, 실제로 틀림:
#       chainlink: log₂1000=10  <  실제 필요 R=12   → under-merge (오답)
#       target   : log₂770 =10  <  실제 필요 R=11   → under-merge (오답)
#     원인: max-전파는 라운드당 1 graph-hop만 보장 → R = Θ(graph diameter).
#           log₂N 라운드로 연결성분을 푸는 기법(pointer jumping)은 암호화 인덱스
#           gather가 필요해 CKKS에서 O(N²) → 사용 불가.
#
#   [상한을 공식으로 못 잡는 이유 — 반례 3종 확인]
#     f(N,k_max)      : 나선 데이터에서 window 과다계수(37 vs 실제 stride 11종)
#                       → R=448, c=R·k/N=27.6. 무계.
#     f(N,k_box_max)  : blob+chain 혼합에서 밀집부가 희소 병목을 은폐 (R=199, 상한 6)
#     f(N,k_box_min)  : 1로 퇴화 → 2N (자명한 상한)
#     증명 가능한 유일한 상한은 n_rounds = N−1 (diameter ≤ N−1) → 비용 4N·k_max 비현실적.
#     ⇒ n_rounds는 eps/min_pts와 같은 '파라미터'로 취급. (FHE k-means가 반복수를
#       하이퍼파라미터로 고정하는 관행과 동일.)
#
#   [32의 근거 — 대상 7개 데이터셋 평문 실측 (그룹 트리-max, m=min(k,⌊slot/N⌋−1))]
#     hepta R=3, tetra R=7, lsun R=11, chainlink R=31, target R=26, atom R=6, moons R=29
#     → 최대 31 → 2의 거듭제곱 올림 = 32 (여유 1). 7개 전부 ARI=1.0 (sklearn 일치).
#   [한계] 곡률이 큰 매니폴드(나선 등)에는 불충분. 논문 한계 절에 명시할 것.
_N_ROUNDS = 8

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


# ══════════════════════════════════════════════════════════════════════════
# ★ [2026-07] PCA(제1주성분) 정렬 + 1-Lipschitz window k_max  ← 현행 기본 경로
# ══════════════════════════════════════════════════════════════════════════
#
# [교체 이유] Ball Tree DFS의 pole-to-pole 분할은 확산/대칭 구조에서 클러스터를
#   슬롯 양극단으로 흩어 k_max ≈ N//2로 최악화 (실측: lsun/tetra 199, twodiamonds 399).
#   PCA 정렬로 교체 시 k_max 70/96/75/63 → LP fhe_max 호출 1.5~6.3배 감소.
#   평문 미러 검증에서 4개 데이터셋 ARI=1.0000 (sklearn DBSCAN 일치) 유지.
#
# [Ball Tree 대비 유지되는 성질]
#   회전 불변: PC1은 데이터 분포의 최대분산 방향 → 좌표계 임의 선택에 비의존
#   (coord-lex은 x축 우선이라 회전에 취약; 그래서 PCA 채택)
#
# [주의] 사영값은 '슬롯 순서 결정'에만 쓰이고 버려짐. 서버 거리 계산은 여전히
#   원래 dim차원 암호문으로 수행 → 차원축소로 인한 정보 손실 없음.

def build_pca_order(pts: np.ndarray) -> tuple:
    """
    제1주성분(PC1) 축 사영 후 정렬 — Ball Tree DFS 대체.

    비용: 중심화 O(N·d) + SVD O(N·d² + d³) + argsort O(N log N)
          d는 데이터 차원(상수) → 지배항 O(N log N) ✓ 예산 충족

    Returns
    -------
    order    : np.ndarray[int, N]  order[i] = 슬롯 i에 놓일 원래 데이터 인덱스
    inv_perm : np.ndarray[int, N]  inv_perm[j] = 원래 인덱스 j의 슬롯 위치
    proj_sorted : np.ndarray[float, N]
        슬롯 순서(오름차순)의 PC1 사영값. compute_kmax_from_pca_window에 재사용
        (재계산 방지). 이 값은 k_max 산출 후 폐기 — 서버로 전송되지 않음.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    N = len(pts_arr)

    X = pts_arr - pts_arr.mean(axis=0)                 # 중심화
    # full_matrices=False: N×d (N≫d) → thin SVD, O(N·d²)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = Vt[0]                                       # PC1 단위벡터 (‖axis‖=1)

    proj  = X @ axis                                   # d차원 → 스칼라 (사영)
    order = np.argsort(proj, kind='stable')

    inv_perm        = np.empty(N, dtype=int)
    inv_perm[order] = np.arange(N)
    return order, inv_perm, proj[order]


def compute_kmax_from_pca_window(
    proj_sorted: np.ndarray,
    eps_norm: float,
    N: int,
) -> int:
    """
    1-Lipschitz 정렬키의 ε-window 상한으로 k_max 산출 — eps-이웃 조회 없음.

    [정당성]
      PC1 사영 z(p) = (p − μ)·v, ‖v‖=1 이므로
        |z(p) − z(q)| = |(p−q)·v| ≤ ‖p−q‖‖v‖ = ‖p−q‖    (Cauchy-Schwarz)
      → 1-Lipschitz. 따라서 두 점이 ε-이웃이면 반드시 |z_i − z_j| ≤ ε.
      (역은 성립 안 함 → 상한이지 등식이 아님. 안전 방향.)

      z-정렬 슬롯에서 슬롯 i의 ε-window = {j : z_j − z_i ≤ ε} 는 연속 구간.
      i와 ε-이웃인 모든 j는 이 window 안 → linear stride ≤ W := max_i window_span(i).

    [circular stride 변환 — ★ 함정 주의]
      LP는 k=1..k_max forward+backward → edge(i,j)는 min(s, N−s) ≤ k_max면 커버.
      s ≤ W 만 알 때 min(s, N−s)의 worst case는:
        W ≤ N/2  →  W          (s=W에서 최대)
        W >  N/2 →  N/2        (s=N/2에서 최대; N−W가 아님!)
      ∴ k_max = min(W, N//2).   ← min(W, N−W)로 쓰면 W>N/2 시 과소추정(under-merge).

    [비용] two-pointer O(N). eps-이웃 전수조회 없음 → 시나리오 2 보호 유지.

    [tight성 실측] oracle(실제 max core-core stride) 대비 1.01~1.21배
      hepta 70/65, lsun 96/95, tetra 75/62, twodiamonds 63/55

    [퇴화 케이스] 데이터가 PC1으로 거의 안 퍼지면(완전 등방 구형) window가
      넓어져 k_max → N//2. 정확성은 유지(상한이므로), 연산 이득만 소멸.
      서버 enc_sum_k 정밀화가 이후 추가로 조임.
    """
    if N <= 1:
        return 1

    j = 0
    W = 0
    for i in range(N):
        if j < i:
            j = i
        # z는 오름차순 → window는 단조 확장 (two-pointer, 총 O(N))
        while j + 1 < N and proj_sorted[j + 1] - proj_sorted[i] <= eps_norm:
            j += 1
        span = j - i
        if span > W:
            W = span

    return min(max(W, 1), N // 2)


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

    # ★ [2026-07] Ball Tree DFS → PCA(PC1) 정렬 교체.
    #   Ball Tree는 확산/대칭 구조에서 k_max≈N//2 최악화 (lsun/tetra 199, twodia 399).
    #   PCA 정렬 + 1-Lipschitz window로 k_max 1.5~6.3배 감소, ARI=1.0 유지 (평문 검증).
    t0 = time()
    order, inv_perm, proj_sorted = build_pca_order(norm_pts)
    print(f"[Client] PCA(PC1) 정렬 구축: {time()-t0:.3f}초  "
          f"(중심화+thin SVD+argsort, O(N log N))")

    t1 = time()
    k_max = compute_kmax_from_pca_window(proj_sorted, eps_norm, N)
    print(f"[Client] k_max ε-window 상한: {time()-t1:.3f}초 "
          f"(two-pointer O(N), eps-이웃 조회 없음)")
    print(f"[Client] k_max 상한={k_max}   (1-Lipschitz 보장: ε-이웃 ⇒ |Δz|≤ε)")
    print(f"         T({k_max})={k_max*(k_max+1)//2}  안전값={N//2}")
    if k_max >= N // 2:
        print(f"         ⚠ k_max가 안전값({N//2})에 도달 — PC1 분산 부족(등방 분포) 의심. "
              f"정확성은 유지되나 LP 연산 이득 소멸.")
    if min_pts >= threshold:
        print(f"[Client] 전파 방식: KD-dense "
              f"(min_pts={min_pts} ≥ 2×dim={threshold}, Sander et al. 1998)")
    else:
        print(f"[Client] 전파 방식: KD-dense (통일) "
              f"— min_pts={min_pts} < 2×dim={threshold} 이지만 sweep 폐기, "
              f"작업 A+B로 kd_dense가 희소 데이터도 커버")
    return 'kd_dense', k_max, order, inv_perm


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

# ══════════════════════════════════════════════════════════════════════════
# ★ [2026-07] 간격 기반 클러스터 판정 — 정수 반올림(round) 대체
# ══════════════════════════════════════════════════════════════════════════
#
# [문제] 기존 round(float(x)) 는 "FHE 라벨이 정수 근처에 있다"를 가정한다.
#   그러나 fhe_max 는 호출마다 라벨을 미세하게 깎는다(sign 근사 오차 × 라벨차).
#   tetra 실측: n_rounds=32 → fhe_max 768회 → 라벨 400 → 384.55 (×0.961).
#   감쇠는 정수를 보존하지 않으므로 감쇠 후 값이 우연히 x.5 근처에 앉으면
#   같은 클러스터가 두 정수로 쪼개진다.
#
# [실측 — tetra eps=0.43(정규화 0.0996) minPts=3]
#   주 클러스터 96점: 187.5592 → round → 188
#   border     2점: 187.4796 → round → 187     ← 값 차이 0.0796 뿐인데 187.5 를 사이에 둠
#   4개 클러스터 전부 동일 → FHE 9클러스터(평문 4) → ARI 0.9731
#   간격 기반으로 바꾸면 → 5클러스터(노이즈1+4) → ARI 1.0000
#
# [해법] 정렬 후 간격으로 분리. "같은 클러스터의 라벨은 서로 가깝고 다른
#   클러스터와는 멀다"만 사용하므로 비례 감쇠(×c)·균일 이동(+c) 모두에 불변.
#   FHE 라벨은 본래 실수이며 정수는 초기값일 뿐이므로, 정수 강제가 인위적이었다.
#
# [실측 간격]
#   dataset  내부 최대 간격   사이 최소 간격   여유
#   tetra        0.0790          47.92        606배
#   lsun         (라벨 200/333/400)  67
#   hepta        (라벨 32/62/…/212)  30
#   → thr=1.0 은 (0.08, 30) 한가운데. thr 0.5~10 전 구간에서 ARI 1.0000 확인.
#
# [안전 조건] 내부 퍼짐 < thr < 최소 클러스터간 간격.
#   클러스터간 간격 = 두 클러스터의 최대 슬롯 인덱스 차. PCA 정렬은 국소성을
#   보존하므로 각 클러스터의 최대 슬롯은 그 클러스터 구간의 끝에 위치한다.
#   서로 다른 클러스터의 최대 슬롯이 인접(간격 1)하려면 두 클러스터가 슬롯상
#   촘촘히 교대하면서 하필 경계에서 끝나야 한다.
#
#   [7개 FCPS 실측 — 라벨 = 클러스터의 최대 core 슬롯 + 1]
#     atom      13  ← 최소   [220,293,306,401,438,477,530]
#     hepta     17          [37,74,125,142,164,188,212]
#     tetra     48          [202,284,351,399]
#     lsun      67          [200,333,400]
#     moons    110          [290,400]
#     target   151          [616,767]
#     chainlink 999         [1000]  (클러스터 1개)
#   클러스터 '내부' 최대 퍼짐: 0.079 (tetra 실측)
#   → thr=1.0 은 0.079 와 13 사이. 분할측 12.7배 / 병합측 13배 여유.
#
#   ※ 이는 경험적 관찰이지 상한 보장이 아니다.
#     반례 구성 가능: A={51,101,198}, B={50,100,199} 처럼 두 클러스터가 슬롯상
#     뒤섞이면 최대 슬롯이 인접할 수 있다 (lsun 이 실제 겹침 구조 —
#     클러스터1 슬롯[202,399], 클러스터2 슬롯[200,339]. 다만 최대 슬롯이 멀어
#     간격 67 확보). 아래 진단 출력으로 매 실행 확인할 것.
_GAP_THRESHOLD = 1.0      # 라벨 간격이 이 값을 넘으면 다른 클러스터
_NOISE_BAND = 0.5         # |라벨| < 이 값이면 노이즈


def assign_clusters_by_gap(orig_labels, N, thr=None, noise_band=None):
    """복호화된 실수 라벨 → 클러스터 ID. 정렬 후 간격으로 분리.

    반환: list[int]. 노이즈는 -1, 클러스터는 대표 라벨(그룹 중앙값의 반올림).
    """
    thr = _GAP_THRESHOLD if thr is None else thr
    noise_band = _NOISE_BAND if noise_band is None else noise_band
    v = np.asarray(orig_labels, dtype=float)
    out = np.full(len(v), -1, dtype=int)

    live = np.where(np.abs(v) >= noise_band)[0]
    if len(live) == 0:
        print(f"[Client] 간격판정: 전부 노이즈 (|라벨| < {noise_band})")
        return out.tolist()

    order = live[np.argsort(v[live])]
    grp, groups = [order[0]], []
    for a, b in zip(order[:-1], order[1:]):
        if v[b] - v[a] > thr:
            groups.append(grp); grp = [b]
        else:
            grp.append(b)
    groups.append(grp)

    for gidx in groups:
        rep = int(round(float(np.median(v[gidx]))))
        out[gidx] = max(1, min(N, rep))

    sv = np.sort(v[live]); gaps = np.diff(sv)
    inner = gaps[gaps <= thr]; outer = gaps[gaps > thr]
    imax = float(inner.max()) if len(inner) else 0.0
    omin = float(outer.min()) if len(outer) else float("inf")
    print(f"[Client] 간격판정: thr={thr} → 클러스터 {len(groups)}개, "
          f"노이즈 {int((out == -1).sum())}개")
    print(f"[Client]   내부 최대 간격 {imax:.4f} < thr {thr} < 사이 최소 간격 {omin:.4f}"
          f"   여유 {omin / max(imax, 1e-9):.0f}배"
          if len(outer) else
          f"[Client]   내부 최대 간격 {imax:.4f} (클러스터 1개)")
    if len(outer) and not (imax < thr < omin):
        print("[Client]   ★★★ 경고: thr 이 안전 구간 밖! 분리/병합 오류 가능")
    return out.tolist()


def run_client_dbscan_fhe(pts: list, eps: float, min_pts: int):
    """
    Production 클라이언트 진입점.

    흐름:
      1. 정규화 → [0, 1]
      2. prepare_client_ordering:
           - PCA(PC1) 사영 정렬  ★ [2026-07] Ball Tree DFS에서 교체
           - ε-window k_max 상한 계산 (1-Lipschitz, eps-이웃 조회 없음)
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

    # 2. PCA(PC1) 정렬 + ε-window k_max 상한 (prepare_client_ordering 통합 사용)
    mode, k_max, heap_idx, inv_perm = prepare_client_ordering(norm, ne, min_pts, N, dim)

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
        num_sweeps=k_max if mode == 'sweep' else None, # sweep일 때만 사용
        n_rounds=_N_ROUNDS,                    # ★ [2026-07] log₂N → 고정 32 (아래 상수 주석 참조)
    )

    # 4. 복호화 + 역순열
    elapsed = time() - start
    print(f"[Client] 서버 완료 ({elapsed:.2f}초). 복호화 중...")

    heap_labels = np.real(engine.decrypt(enc_result, secret_key))[:N]
    # heap_labels[i]: DFS 위치 i = 원래 heap_idx[i]번 점의 라벨
    # original_labels[j] = heap_labels[inv_perm[j]]
    orig_labels = heap_labels[inv_perm]

    cluster_labels = assign_clusters_by_gap(orig_labels, N)
    n_c = len(set(l for l in cluster_labels if l != -1))
    print(f"[Client] 완료! 클러스터 {n_c}개 (노이즈: {cluster_labels.count(-1)}개)")
    return [list(pts[i]) + [cluster_labels[i]] for i in range(N)], cluster_labels
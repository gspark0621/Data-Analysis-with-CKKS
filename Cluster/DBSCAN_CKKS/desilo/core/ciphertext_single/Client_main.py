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

    t0 = time()
    order, inv_perm = build_ball_tree_order(norm_pts)
    print(f"[Client] Ball Tree DFS 구축: {time()-t0:.3f}초")

    t1 = time()
    k_max = compute_kmax_from_ball_structure(norm_pts, order, eps_norm, N)
    print(f"[Client] k_max 구조 분석 (DFS): {time()-t1:.3f}초 (eps-이웃 조회 없음)")
    print(f"[Client] k_max 상한={k_max}")
    print(f"         T({k_max})={k_max*(k_max+1)//2}  안전값={N//2}")
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

def gap_cluster_labels(orig_labels, N, min_gap=0.5, gap_factor=5.0):
    """
    ★ [Tier 0 / 2026-06] round() 대체용 클라이언트 후처리.

    복호화된 연속 라벨값을 '큰 gap'에서 잘라 클러스터링한다.
    배경(진단): 같은 클러스터 점들의 라벨값 std는 매우 작지만(응집 완벽),
      전파 누적오차로 값이 정수에서 벗어나 한 클러스터가 14.29/14.71처럼
      정수 경계(x.5)를 가로지르면 round()가 한 클러스터를 둘로 쪼갠다(과분할).
      gap 기반 클러스터링은 '값이 거의 같은 무리'를 한 클러스터로 묶어 이를 복구.

    한계(솔직히): 이건 *과분할*만 고친다. 서로 다른 클러스터의 라벨값이 1 미만으로
      겹쳐버린 *병합*(hepta 유형)은 정보가 이미 소실되어 복구 불가하며, 임계값을
      잘못 잡으면 오히려 더 합쳐질 수 있다. 따라서 Tier 1(전파 정밀도)과 병행해야 한다.

    Parameters
    ----------
    orig_labels : 원래 순서로 복원된 연속 라벨값 (np.ndarray, 길이 N)
    min_gap     : 클러스터 분리로 인정할 최소 gap (기본 0.5)
    gap_factor  : 클러스터 '내부' gap(중앙값)의 몇 배를 분리 기준으로 볼지

    Returns
    -------
    labels : list[int]  (노이즈 = -1, 클러스터 = 1,2,3,...)
    """
    vals = np.asarray(orig_labels, dtype=float)
    out  = np.full(len(vals), -1, dtype=int)
    core_idx = np.where(vals > 0.5)[0]            # 양수 라벨만 클러스터 후보
    if len(core_idx) == 0:
        return out.tolist()

    sv_order = core_idx[np.argsort(vals[core_idx])]
    sv       = vals[sv_order]
    gaps     = np.diff(sv)
    pos_gaps = gaps[gaps > 1e-9]
    thr      = max(min_gap, (np.median(pos_gaps) * gap_factor) if len(pos_gaps) else min_gap)

    cluster_id = 1
    out[sv_order[0]] = cluster_id
    for i in range(1, len(sv)):
        if gaps[i - 1] > thr:
            cluster_id += 1
        out[sv_order[i]] = cluster_id
    return out.tolist()


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

def run_client_dbscan_fhe(pts: list, eps: float, min_pts: int,
                          mask_mode: str = "per_stride",   # ★ Tier 1b ("per_stride"|"per_pass")
                          post_process: str = "round",      # ★ Tier 0 ("round"|"gap")
                          lp_snapshot: bool = False):       # ★ LP pass별 CSV 저장
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
        n_rounds=math.ceil(math.log2(N)),      # ★ [2026-05c 작업 B] log₂N round
        mask_mode=mask_mode,                   # ★ Tier 1b
        lp_snapshot=lp_snapshot,               # ★ LP pass별 스냅샷
    )

    # 4. 복호화 + 역순열
    elapsed = time() - start
    print(f"[Client] 서버 완료 ({elapsed:.2f}초). 복호화 중...")

    heap_labels = np.real(engine.decrypt(enc_result, secret_key))[:N]
    # heap_labels[i]: DFS 위치 i = 원래 heap_idx[i]번 점의 라벨
    # original_labels[j] = heap_labels[inv_perm[j]]
    orig_labels = heap_labels[inv_perm]

    # ── 4-1. 후처리: round (기존) vs gap (Tier 0) ──────────────────────
    #   양쪽을 모두 계산해 클러스터 수를 비교 출력 → 데이터셋별 선택 근거 제공.
    round_labels = []
    for x in orig_labels:
        r = round(float(x))
        if r <= 0:      round_labels.append(-1)
        elif r > N:     round_labels.append(N)
        else:           round_labels.append(r)
    gap_labels = gap_cluster_labels(orig_labels, N)

    n_round = len(set(l for l in round_labels if l != -1))
    n_gap   = len(set(l for l in gap_labels  if l != -1))
    print(f"[Client] 후처리 비교: round→클러스터 {n_round}개 | "
          f"gap→클러스터 {n_gap}개  (선택: {post_process})")
    print(f"         라벨값 범위 min={float(np.min(orig_labels)):.3f} "
          f"max={float(np.max(orig_labels)):.3f}")

    cluster_labels = gap_labels if post_process == "gap" else round_labels

    n_c = len(set(l for l in cluster_labels if l != -1))
    print(f"[Client] 완료! 클러스터 {n_c}개 (노이즈: {cluster_labels.count(-1)}개, "
          f"post_process={post_process})")
    return [list(pts[i]) + [cluster_labels[i]] for i in range(N)], cluster_labels
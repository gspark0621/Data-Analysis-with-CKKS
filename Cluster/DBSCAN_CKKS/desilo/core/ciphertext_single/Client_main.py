# core/ciphertext_single/Client_main.py
#
# ── 현재 구조 ────────────────────────────────────────────────────────────────
# 정렬      : PCA(제1주성분) 사영 정렬  (Ball Tree DFS 는 2026-09 삭제)
# k_max     : 1-Lipschitz ε-window 상한, two-pointer O(N), eps-이웃 조회 없음
# 전파 방식 : kd_dense 단일 경로 (all-sweep 은 2026-05c 폐기, 2026-10 코드 삭제)
# n_rounds  : ★[2026-10] 클라이언트가 grid(셀 한 변 ε_n/2)로 산출해 암호문과 함께 전송
#             (GridRounds.compute_grid_rounds). 서버의 n_rounds_from_public 은 미전달 시 대비용.
# 후처리    : assign_clusters_by_gap — 정수 반올림 대신 간격 기반 판정
#
# ── 2026-10 변경 ────────────────────────────────────────────────────────────
#  · n_rounds 를 클라이언트 grid 방식으로 산출: n = max(1, max_g ⌈(Λ_g + s√d)/ε_n⌉)
#      셀 한 변 s = ε_n/2,  Λ_g = 셀 묶음 g 의 측지 지름(셀 그래프 최단 거리 최댓값, 정확값)
#      가지치기 Dijkstra(BoundingDiameters)로 계산 — 결과는 모든 셀 Dijkstra 와 동일
#    점 사이 거리 비교 없음, α·margin 등 서버 파라미터 미사용. 상한 ⌈1/ε_n⌉ 로 자르지 않음.
#    서버에는 n_rounds 정수 하나가 추가로 드러난다.
#  · PC1 부호 규칙: '절댓값 최대 성분 양수' → '음수' (S1 등 이전 32라운드 실행과 같은 방향)
#  · sweep 잔재 삭제: _kd_dense_threshold(Sander 2×dim), mode 반환값, mode 분기,
#    send_to_server 의 use_kd_propagation / num_sweeps 인자.
#    prepare_client_ordering(norm_pts, eps_norm, N, proj=None) → (k_max, order, inv_perm)
#    (min_pts·dim 인자는 sweep 판정에만 쓰였으므로 함께 제거)
#
# ── 2026-09 정리 ────────────────────────────────────────────────────────────
#  · _N_ROUNDS 상수(29→32) 폐기 → n_rounds_from_public() 로 교체, 이후 서버로 이관
#  · 정규화 스케일: 좌표범위 → max(좌표범위, PC1 span)
#  · choose_gap_threshold 의 도달 불가 "비율점프 보조 경로" 삭제 (문서도 정정)
#  · build_ball_tree_order / compute_kmax_from_ball_structure /
#    build_kd_tree_order / get_kd_dense_kmax / decide_propagation_mode 삭제
#    (전부 호출부 없음. 비교 실험이 필요하면 git 이력에서 복원)
# ────────────────────────────────────────────────────────────────────────────

from time import time
import math
import numpy as np
import desilofhe
import pynvml
from core.ciphertext_single.EncryptModule import DimensionalEncryptor
from core.ciphertext_single.Server_main import send_to_server_fhe as send_to_server
from core.ciphertext_single.GridRounds import compute_grid_rounds
from util.keypack import KeyPack

# ══════════════════════════════════════════════════════════════════════════
# [2026-10] LP Phase-1 라운드 수 → 클라이언트 grid 산출 (GridRounds.compute_grid_rounds)
# ══════════════════════════════════════════════════════════════════════════
#   이전(2026-09): n_rounds = min(⌈1/ε_n⌉, ⌈3N/min_pts⌉−1) 를 서버가 공개값만으로 산출.
#     → ⌈1/ε_n⌉ 은 '데이터 전체 scale' 기준이라 클러스터 사이 빈 공간만큼 과잉
#       (zelnik6 56 vs 필요 10, S1 24 vs 1, longsquare 33 vs 11).
#       공개값만으로는 클러스터 크기를 알 수 없고(불가능성), RGG 근거는 우리 밀도에서 공허.
#   현재: 클라이언트가 셀(ε_n/2) 점유 정보로 셀 묶음(클러스터를 통째로 품는 덩어리)을 만들고,
#     묶음마다 측지 지름 Λ_g 를 정확히 구해 n = max(1, max_g ⌈(Λ_g + s√d)/ε_n⌉) 를
#     암호문과 함께 보낸다 (one-way 유지). 가정 A(hop 마다 ε_n 전진) 아래에서 FHE 가 평문과
#     똑같이 재현한 모든 클러스터의 전파 완료가 보장된다 (2026-10 증명 시리즈).
#     · 점 사이 거리 비교 없음 → DBSCAN 범위 질의를 하지 않는다.
#     · 누출: n_rounds 정수 하나 (가장 큰 셀 묶음의 측지 지름 / ε_n 수준).
#     · 서버 n_rounds_from_public 은 n_rounds 미전달 시의 대비 경로로 남긴다.


def _gpu_used_mb() -> float:
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0


# ── Ball Tree DFS in-order 정렬 ───────────────────────────────────────────

# [삭제됨 2026-09] build_ball_tree_order / compute_kmax_from_ball_structure
#   Ball Tree DFS 정렬은 확산·대칭 구조에서 k_max≈N//2 로 최악화되어(lsun/tetra 199,
#   twodiamonds 399) PCA(PC1) 정렬로 완전 교체되었다. 호출부가 남아 있지 않아 삭제.
#   비교 실험이 필요하면 git 이력에서 복원할 것.


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

def build_pca_order(pts: np.ndarray, proj: np.ndarray = None) -> tuple:
    """
    제1주성분(PC1) 축 사영 후 정렬 — 슬롯 배치 순서 결정.

    비용: 중심화 O(N·d) + thin SVD O(N·d² + d³) + argsort O(N log N)
          d 는 데이터 차원(상수) → 지배항 O(N log N)

    ★ [2026-09] proj 를 넘기면 중심화·SVD 를 건너뛴다.
      send_to_server_fhe 는 정규화 스케일(PC1 span)을 구하려고 이미 사영을 계산하므로,
      그 결과를 그대로 넘겨 O(N·d²) 중복을 제거한다. 실측(N=5000,d=2)에서 중심화+SVD 가
      전처리의 25%, (N=50000,d=10) 에서는 54% 였다.

    Parameters
    ----------
    pts  : (N, d) 정규화된 좌표
    proj : (N,) 사전 계산된 PC1 사영값. None 이면 여기서 계산한다.
           균등 스케일링·평행이동에 PC1 축이 불변이므로, 원시 좌표에서 계산한 사영을
           같은 scale 로 나눠 넘겨도 정규화 좌표계의 사영과 일치한다.

    Returns
    -------
    order       : (N,) order[i] = 슬롯 i 에 놓일 원래 데이터 인덱스
    inv_perm    : (N,) inv_perm[j] = 원래 인덱스 j 의 슬롯 위치
    proj_sorted : (N,) 슬롯 순서(오름차순)의 PC1 사영값.
                  compute_kmax_from_pca_window 에 재사용 후 폐기 — 서버로 보내지 않는다.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    N = len(pts_arr)

    if proj is None:
        X = pts_arr - pts_arr.mean(axis=0)                 # 중심화 O(N·d)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)   # thin SVD O(N·d²)
        axis = Vt[0]
        # ★ SVD 의 특이벡터 부호는 구현 의존이다. 부호가 뒤집히면 슬롯 순서가 전부
        #   역전되어 로그·재현성이 흔들리고, 두 클러스터에 걸친 border 점의 max-label
        #   귀속도 반대로 바뀐다. 절댓값 최대 성분을 '음수'로 고정해 결정론적으로 만든다.
        #   [2026-10] 양수 → 음수: S1 등 이전(부호 고정 전) 32라운드 실행과 같은 방향.
        if axis[np.argmax(np.abs(axis))] > 0:
            axis = -axis
        proj = X @ axis
    proj = np.asarray(proj, dtype=np.float64)

    order       = np.argsort(proj, kind="stable")          # O(N log N)
    inv_perm    = np.empty(N, dtype=np.int64)
    inv_perm[order] = np.arange(N, dtype=np.int64)
    proj_sorted = proj[order]
    return order, inv_perm, proj_sorted


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
        while j + 1 < N and proj_sorted[j + 1] - proj_sorted[i] <= eps_norm:
            j += 1
        W = max(W, j - i)

    # ★ 함정: min(W, N//2) 이다. min(W, N−W) 로 쓰면 W > N/2 일 때 과소추정 → under-merge.
    return min(max(W, 1), N // 2)


def prepare_client_ordering(
    norm_pts: np.ndarray,
    eps_norm: float,
    N: int,
    proj: np.ndarray = None,   # ★ [2026-09] 사전 계산된 PC1 사영 (중복 SVD 제거)
) -> tuple:
    """PCA(PC1) 슬롯 정렬 + ε-window k_max.  반환: (k_max, order, inv_perm).

    ★ [2026-10] sweep 삭제로 mode 반환값과 min_pts·dim 인자를 제거했다.
      (이력: 2026-05c 까지는 min_pts ≥ 2×dim 이면 kd_dense, 아니면 sweep 이었고,
       sweep 은 mask damping 누적으로 iris 등에서 라벨이 소멸해 폐기되었다.)
    """

    # ★ [2026-07] Ball Tree DFS → PCA(PC1) 정렬 교체.
    #   Ball Tree는 확산/대칭 구조에서 k_max≈N//2 최악화 (lsun/tetra 199, twodia 399).
    #   PCA 정렬 + 1-Lipschitz window로 k_max 1.5~6.3배 감소, ARI=1.0 유지 (평문 검증).
    t0 = time()
    order, inv_perm, proj_sorted = build_pca_order(norm_pts, proj=proj)
    print(f"[Client] PCA(PC1) 정렬 구축: {time()-t0:.3f}초  "
          f"({'argsort only, 사영 재사용' if proj is not None else '중심화+thin SVD+argsort'}"
          f", O(N log N))")

    t1 = time()
    k_max = compute_kmax_from_pca_window(proj_sorted, eps_norm, N)
    print(f"[Client] k_max ε-window 상한: {time()-t1:.3f}초 "
          f"(two-pointer O(N), eps-이웃 조회 없음)")
    print(f"[Client] k_max 상한={k_max}   (1-Lipschitz 보장: ε-이웃 ⇒ |Δz|≤ε)")
    print(f"         T({k_max})={k_max*(k_max+1)//2}  안전값={N//2}")
    if k_max >= N // 2:
        print(f"         ⚠ k_max가 안전값({N//2})에 도달 — PC1 분산 부족(등방 분포) 의심. "
              f"정확성은 유지되나 LP 연산 이득 소멸.")
    return k_max, order, inv_perm


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
    sbk = _create("smallbootstrap_key",  engine.create_small_bootstrap_key)

    # ★ [2026-07] full bootstrap_key 생성 생략 (13.4GB → 0).
    #   DesiloFHE 는 engine.bootstrap(ct, relin, conj, rotation, small_bootstrap)
    #   형태를 지원하므로, 값 보존 bootstrap(EvalMod) 도 small 키로 수행 가능하다.
    #   'small' 은 EvalSign 전용 키가 아니라 **키 변형**이다.
    #   (sign_bootstrap 과 bootstrap 은 여전히 다른 연산 — 라벨처럼 임의 실수를
    #    나르는 암호문에는 반드시 bootstrap 을 써야 한다. 바뀐 건 키뿐이다.)
    #   실사용처: Label_Propagation._refresh, cleaning._refresh — 둘 다
    #   _USE_SMALL_BOOT_FOR_REFRESH=True 로 small 키 경로를 쓴다.
    #   나머지 3곳(Core.py:72, Normalize.py:68, chebyshev_eval.py:157)은
    #   죽은 할당/죽은 분기라 None 이어도 무해하다.
    #   ⇒ GPU 사용량 18.6GB → 약 5.2GB. Step 1(Normalize) 여유 6GB → 19GB.
    #   문제 시 _CREATE_FULL_BOOTSTRAP_KEY=True + 각 파일의
    #   _USE_SMALL_BOOT_FOR_REFRESH=False 로 원복.
    _CREATE_FULL_BOOTSTRAP_KEY = False
    if _CREATE_FULL_BOOTSTRAP_KEY:
        bk = _create("bootstrap_key",    engine.create_bootstrap_key)
    else:
        bk = None
        if verbose:
            print(f"  [MEM] bootstrap_key                             "
                  f"생략 (small 키로 대체)  (used={_gpu_used_mb():.0f} MB)")

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
_GAP_THRESHOLD = 1.0      # 고정 thr (적응형 실패 시 폴백)
_NOISE_BAND = 0.5         # |라벨| < 이 값이면 노이즈

# ★ [2026-07] 적응형 thr — 고정 1.0 대신 복호화된 라벨에서 직접 결정.
#
# [동기] thr 의 안전조건은 "내부퍼짐 < thr < 클러스터간 최소간격" 인데,
#   두 경계 모두 **데이터 의존**이다. 특히 클러스터간 최소간격은 N·k_max·dim
#   같은 공개값으로는 유도할 수 없다(위 §반례: 슬롯상 뒤섞이면 간격 1 가능).
#   고정 1.0 은 FCPS 7개에서 우연히 (0.079, 13) 사이에 들어맞았을 뿐 보장이 아니다.
#
# [정보유출 없음] 이 계산은 **클라이언트가 이미 복호화해 손에 쥔 라벨**만 사용한다.
#   서버로 아무것도 보내지 않고, 서버는 thr 을 알지 못한다. 즉 기존 대비 추가 유출 0.
#   (반대로 서버측에서 thr 을 정하려면 클러스터 구조를 알아야 하므로 유출이 생긴다.)
#
# [원리] 정렬된 gap 들은 '내부 gap(작음)'과 '클러스터간 gap(큼)' 두 무리로 갈린다.
#   그 사이의 최대 *비율 점프* 를 찾아 기하평균을 thr 로 취한다. 비례감쇠·균일이동에
#   불변이며, 감쇠가 심해지거나 내부퍼짐이 커져도 경계를 스스로 옮긴다.
#
# [검증 — 실측 수치 재현]
#   tetra(내부0.079,간격48)→thr 1.61 (4클러스터 ✓) / atom(간격13)→0.56 (7 ✓)
#   hepta(간격17)→0.73 (7 ✓) / chainlink(단일)→분리없음 판정 (1 ✓)
#   감쇠 ×0.77 → 0.44 (7 ✓) / 내부퍼짐 10배 악화 → 1.46 (7 ✓)
#   ※ 고정 1.0 도 이 케이스들은 통과하나, 여유가 줄면 먼저 깨진다.
_ADAPTIVE_THR      = True    # False 면 _GAP_THRESHOLD 고정 사용
# [삭제됨 2026-09] _ADAPTIVE_MIN_RATIO — 비율점프 판정에만 쓰였는데 그 경로가
#   도달 불가 코드로 삭제되면서 함께 죽었다. choose_gap_threshold 는 이제 감쇠계수 c
#   기반 단일 경로이고 비율 판정을 하지 않는다.

# ★ [2026-07 최종] thr = 1.0·c  (c 추정 불가 시 1.0 폴백)
#
#   [원리] 라벨 = c·(정수). 여기서 c 는 FHE 감쇠계수이고, **감쇠 후 좌표계에서의
#     라벨 1칸**이 바로 c 다. 따라서 thr=1.0·c 는 "라벨 한 칸"을 뜻한다.
#     감쇠가 없으면 c=1 이라 예전의 하드코딩 1.0 과 정확히 같아진다.
#
#   [안전 구간]  내부퍼짐 s  <  thr  <  최소 진짜 간격
#     · 상한: 라벨 = 성분 내 최대(슬롯인덱스+1)이고 PCA 정렬로 클러스터가 인덱스
#       구간을 차지하므로, 인접 두 클러스터의 라벨 차 = 뒤 클러스터의 크기 ≥ min_pts.
#       ⇒ 최소 진짜 간격 ≥ min_pts·c.  thr=1.0·c 면 상한 여유 = **정확히 min_pts 배**로
#         감쇠와 무관한 불변량이 된다. (고정 1.0 은 3c 라서 c 가 떨어지면 여유도 준다.)
#     · 하한: s < c 여야 한다. 실측 s/c = 0.56(16R), 0.45(32R).
#
#   [실측 검증 — lsun]
#     n_rounds=16: c=0.9316, s=0.5206 → thr=0.9316 ✓ (하한 1.79배, 상한 3.00배)
#     n_rounds=32: c=0.8782, s=0.3977 → thr=0.8782 ✓ (하한 2.21배, 상한 3.00배)
#     ※ 한때 쓴 0.5·c 는 16라운드에서 s 아래로 내려가 클러스터를 갈랐다(ARI 99.47).
#       문제는 c 스케일링이 아니라 **계수 0.5** 였다. 계수는 1.0 이 맞다.
#
#   [분모가 N 이 아니라 max(live_idx)+1 인 이유]
#     live = |라벨| ≥ noise_band 인 점(=노이즈 아님). 뒤쪽에 노이즈가 몰리면
#     최고 core 인덱스도 함께 낮아지므로, 분모를 max(live_idx)+1 로 두면 그 효과가
#     상쇄되어 c 를 정확히 복원한다. 분모를 N 으로 고정하면 과소평가된다.
#       예) N=400, 마지막 10점이 노이즈, 실제 c=0.88
#           분모 N     → c_est=0.858 (2.5% 과소)
#           분모 390   → c_est=0.880 (오차 0)
#     남는 오차원은 뒤쪽 **border**(live 지만 core 아님)뿐이고, 그 경우에도
#     과소평가 폭은 수 % 수준이라 하한 여유(1.8~2.2배) 안에서 흡수된다.
_THR_C_FRACTION = 1.0
_GAP_THRESHOLD_FIXED = 1.0      # c 추정 불가 시 폴백


def choose_gap_threshold(v_live, live_idx=None):
    """복호화된 라벨에서 thr 결정. 반환: (thr, 설명문자열).

    ★ [2026-09] 경로는 하나뿐이다: 감쇠계수 c 기반. c 추정 실패 시 고정 폴백.
      (이전 버전에 남아 있던 "비율점프 보조 경로" 는 return 뒤에 있어 도달 불가였고,
       docstring 만 그것이 동작하는 것처럼 서술하고 있었다 → 코드/문서 모두 삭제.)

    [이전 버그] 비율점프를 1차로 쓰고 `gaps = gaps[gaps > 0]` 로 0 을 걸렀더니,
      **평문 경로에서 오작동**했다. 평문은 라벨이 정확한 정수라 내부 gap 이 정확히 0
      → 필터에 전부 제거 → 남는 건 클러스터간 gap 뿐 → '작은 무리 vs 큰 무리' 대비가
      사라져 비율이 3.6 밖에 안 나오고 '분리 없음'으로 오판(thr=532, 1클러스터).
      FHE 에서는 노이즈로 내부 gap 이 0.079 같은 양수라 우연히 작동했다.

    [원리] 라벨은 항상 c·(정수) 다 (초기값=슬롯인덱스+1, max 전파는 정수 보존,
      FHE 감쇠는 비례 ×c). 따라서
        · 클러스터간 간격 ≥ c·1 = c
        · 내부퍼짐 = 비균일 오차 ≪ c
      ⇒ thr = 0.5·c 는 항상 안전구간 (내부퍼짐, 최소간격) 안에 있다.
      c ≈ max(라벨)/(최대 live 슬롯+1).  c 추정이 2배 틀려도 안전구간이 넓어
      (예: atom 0.06 ~ 10) thr 은 여전히 구간 안에 남는다.

    [검증] 평문 정확(5성분)→5 ✓ / 평문 단일→1 ✓ / FHE atom c=0.77→7 ✓ /
      FHE 간격1 인접 c=0.961→3 ✓  (고정 thr=1.0 은 마지막 케이스에서 병합 실패)

    [유출 없음] 클라이언트가 이미 복호화해 가진 라벨과 live 슬롯 인덱스만 사용.
      서버로 아무것도 보내지 않고 서버는 thr 을 모른다.
    """
    v = np.asarray(v_live, dtype=float)
    sv = np.sort(v)
    gaps = np.diff(sv)

    # ── 1차: 감쇠계수 c 기반 (라벨 = c·정수 구조) ──
    c_est = None
    if live_idx is not None and len(live_idx) > 0:
        denom = int(np.max(live_idx)) + 1
        vmax = float(np.max(v))
        if denom > 0 and vmax > 0:
            c_est = vmax / denom
    # ── 판정: thr = _THR_C_FRACTION · c  (c 추정 불가 시 고정 폴백) ──
    thr = (_THR_C_FRACTION * c_est) if (c_est is not None and c_est > 0) \
          else _GAP_THRESHOLD_FIXED
    inner = gaps[gaps < thr]
    inner_max = float(inner.max()) if len(inner) else 0.0
    n_cut = int((gaps >= thr).sum())
    why = f"thr={thr:.4f} | 내부최대 {inner_max:.4f}, 분리 {n_cut}곳"
    # ── c 는 진단용: 감쇠가 얼마나 진행됐는지 + 여유가 얼마나 남았는지 ──
    if c_est is not None and c_est > 0:
        why += f" | 감쇠 c≈{c_est:.4f}"
        if 3.0 * c_est < thr:
            why += (f"  ★★경고: 3c={3*c_est:.3f} < thr={thr} — 감쇠가 심해 최소 클러스터"
                    f" 간격(≥min_pts·c)이 thr 아래로 내려갔다. 클러스터 병합 위험.")
    if inner_max > 0.5 * thr:
        why += (f"  ★★경고: 내부퍼짐 {inner_max:.4f} 이 thr 의 50% 초과 — "
                f"라벨 감쇠 누적. n_rounds 축소 또는 _SGN_CLEANING_ITERS 상향 필요")
    return thr, why


def assign_clusters_by_gap(orig_labels, N, thr=None, noise_band=None):
    """복호화된 실수 라벨 → 클러스터 ID. 정렬 후 간격으로 분리.

    반환: list[int]. 노이즈는 -1, 클러스터는 대표 라벨(그룹 중앙값의 반올림).
    """
    noise_band = _NOISE_BAND if noise_band is None else noise_band
    v = np.asarray(orig_labels, dtype=float)
    out = np.full(len(v), -1, dtype=int)

    live = np.where(np.abs(v) >= noise_band)[0]
    if len(live) == 0:
        print(f"[Client] 간격판정: 전부 노이즈 (|라벨| < {noise_band})")
        return out.tolist()

    # ★ thr 결정: 명시 지정 > 적응형 > 고정
    if thr is None:
        if _ADAPTIVE_THR:
            thr, why = choose_gap_threshold(v[live], live_idx=live)
            print(f"[Client] 적응형 thr: {why}")
        else:
            thr = _GAP_THRESHOLD

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
      3. grid(셀 ε_n/2) 로 n_rounds 산출  ★ [2026-10]
      4. 암호화 → 서버 전송 (k_max, n_rounds 함께)
      5. 복호화 → inv_perm 적용 → 원래 순서 복원
    """
    start   = time()
    pts_arr = np.array(pts, dtype=np.float64)
    N       = len(pts_arr)
    dim     = pts_arr.shape[1]
    log2_n  = math.ceil(math.log2(N))

    engine, secret_key, keypack = setup_fhe_engine(verbose=False)

    # 1. 정규화 — 스케일 = max(좌표범위, PC1 span)
    #    PC1 span 을 포함시키는 이유: n_rounds 의 기하항이 ⌈PC1extent/ε⌉ 이어야 하는데,
    #    좌표범위만 쓰면 PC1 span > 좌표범위 인 데이터(lsun/longsquare/target/s1)에서
    #    과소평가된다. max() 를 쓰면 항상 덮으면서 데이터는 [0,1]^d 안에 유지되어
    #    Normalize 의 max_dist_sq = dim 가정도 그대로 성립한다.
    #    PC1 사영은 슬롯 정렬용으로 어차피 계산하므로 추가 연산은 O(N) 리덕션 2회뿐.
    g_min   = pts_arr.min()
    g_max   = pts_arr.max()
    coord_r = (g_max - g_min) or 1.0

    # ★ [2026-09] 중심화·SVD·사영을 여기서 딱 한 번만 하고 결과를 prepare_client_ordering
    #   으로 넘긴다. 이전에는 여기서 1회 + build_pca_order 에서 1회로 O(N·d²) 를 두 번
    #   했다. PC1 축은 균등 스케일링·평행이동에 불변이므로 원시 좌표에서 계산한 사영을
    #   같은 scale 로 나누면 정규화 좌표계의 사영과 정확히 일치한다:
    #       norm 의 중심화 사영 = (X − μ)·v / scale = _pc1 / scale
    _Xc  = pts_arr - pts_arr.mean(axis=0)
    _ax  = np.linalg.svd(_Xc, full_matrices=False)[2][0]
    if _ax[np.argmax(np.abs(_ax))] > 0:      # SVD 부호 고정: 최대 성분 음수 [2026-10]
        _ax = -_ax
    _pc1     = _Xc @ _ax
    pc1_span = float(_pc1.max() - _pc1.min())

    scale   = max(coord_r, pc1_span) or 1.0
    norm    = (pts_arr - g_min) / scale
    ne      = eps / scale
    proj_n  = _pc1 / scale                    # 정규화 좌표계의 PC1 사영
    print(f"[Client] N={N}, dim={dim}, eps_norm={ne:.5f}  "
          f"(scale={scale:.4f}: 좌표범위 {coord_r:.4f} / PC1span {pc1_span:.4f})")
    # ★ [2026-10] n_rounds: 클라이언트 grid(셀 ε_n/2) 산출 → 암호문과 함께 전송
    #   pc1 은 각 묶음의 첫 Dijkstra 출발 셀(전파 소스 근처)을 고르는 데만 쓰임 (결과 불변)
    n_rounds, grid_info = compute_grid_rounds(norm, ne, min_pts, pc1=proj_n)

    # 2. PCA(PC1) 정렬 + ε-window k_max 상한 (prepare_client_ordering 통합 사용)
    k_max, heap_idx, inv_perm = prepare_client_ordering(norm, ne, N, proj=proj_n)

    data_for_enc = norm[heap_idx].tolist()
    T_kmax = k_max * (k_max + 1) // 2
    print(f"[Client] PC1 정렬 순서로 재배치 완료  k_max={k_max}, T({k_max})={T_kmax}")

    # 3. 암호화 → 서버
    encryptor = DimensionalEncryptor(engine, keypack)
    enc_cols  = encryptor.encrypt_data(data_for_enc, dim)

    enc_result, _ = send_to_server(
        engine=engine, keypack=keypack,
        encrypted_columns=enc_cols,
        num_points=N, eps=ne, min_pts=min_pts,
        k_max=k_max,                           # 클라이언트 구조 분석 상한 전달
        n_rounds=n_rounds,                     # ★ [2026-10] 클라이언트 grid 산출값
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
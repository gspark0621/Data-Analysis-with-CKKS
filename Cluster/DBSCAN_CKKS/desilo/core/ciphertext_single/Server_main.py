# core/ciphertext_single/Server_main.py
#
# ── 변경사항 ────────────────────────────────────────────────────────────────
# [변경 1] 서버가 k_max를 직접 결정 (방법 2: server-side adj sum)
#   Normalize에서 이미 adj_k 계산 → enc_sum_k = sum(adj_k) 추가 계산
#   O(N//2 × log N) rotation only (bootstrap 없음, ~수초)
#   client가 복호화 → k_max = max k where sum_k > 0
#   → client-side BallTree eps-이웃 전수조회 불필요 (시나리오 2 보호)
#
# [변경 2] 2-Phase 프로토콜
#   Phase 1: normalize_and_core() → adj_k_list  + core_ct 반환
#   Phase 2: label_propagation_phase() → final_ct 반환
#   Client가 Phase 1 결과에서 k_max 결정 후 Phase 2 호출
#
# [유지] send_to_server_fhe: Phase 1+2 통합 호출 (기존 인터페이스 유지)
#
# [변경 3, 2026-09] n_rounds 산출을 서버로 이관 (n_rounds_from_public)
#   클라이언트가 전송하던 값을 서버가 {ε_n, N, min_pts} 로 직접 계산한다.
#   전송 채널 하나 제거 + 클라이언트의 값 조작 여지 제거. 누출은 이전과 같이 0 비트.
# [변경 4, 2026-09] secret_key → debug_secret_key (키워드 전용, 기본 None)
#   전 사용처가 진단 출력뿐임을 확인하고 가드를 씌웠다. 운영 경로에서 서버는
#   비밀키를 받지 않으며 어떤 복호도 수행하지 않는다.
# [변경 5, 2026-10] n_rounds 를 다시 클라이언트가 산출해 전송 (grid, 셀 ε_n/2)
#   n = max(1, max_g ⌈(Λ_g + s√d)/ε_n⌉), Λ_g = 셀 묶음 측지 지름 — GridRounds.compute_grid_rounds 참조.
#   ⌈1/ε_n⌉ 은 데이터 scale 기준이라 빈 공간만큼 과잉이고, 공개값만으로는 클러스터
#   크기를 알 수 없어 클라이언트 산출로 전환했다. 서버에는 n_rounds 정수 하나가 드러난다.
#   n_rounds_from_public 은 n_rounds 미전달 시의 대비 경로로 남긴다.
# [변경 6, 2026-10] ALL-strides sweep 경로 삭제 (LP 는 kd_dense 단일 경로)
#   삭제: fhe_sweep_propagation import, 인자 use_kd_propagation / num_sweeps,
#         num_sweeps 기본값(⌈log₂N⌉), Normalize 루프 상한 분기(N//2),
#         adj_k_list 절단 분기, Step 3 의 sweep 분기.
#   배경: sweep 은 2026-05c 부터 deprecated(mask damping 누적으로 iris 등에서
#         라벨 소멸 실측)였고 클라이언트는 항상 'kd_dense' 를 보내 왔다.
#   ※ 이 두 인자를 키워드로 넘기던 호출부는 TypeError 가 나므로 함께 고쳐야 한다
#     (Client_main.run_client, test_Server_main_call_dataset 는 같은 날 수정).
# ────────────────────────────────────────────────────────────────────────────

import os
import math
import gc
from time import time
import desilofhe
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
import numpy as np
import pynvml
from core.ciphertext_single.Normalize import check_neighbor_closed_interval
from core.ciphertext_single.Core import identify_core_points_fhe_converted as identify_core_points_fhe
from core.ciphertext_single.Label_Propagation import (
    fhe_kd_dense_propagation,   # ★ dense stride k=1..k_max — 유일한 LP 경로 [변경 6]
    fhe_circular_shift,
)


def _gpu_used_mb() -> float:
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0

def _print_mem(label: str):
    print(f"  [MEM][Server] {label:<45}  used={_gpu_used_mb():.0f} MB")

def _mem_delta(label: str, before: float) -> float:
    after = _gpu_used_mb()
    print(f"  [MEM][Server] {label:<45}  delta={after-before:+.0f} MB  (used={after:.0f} MB)")
    return after


def save_vector_csv(filename, values, header):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for i, v in enumerate(values):
                f.write(f"{i},{float(np.real(v)):.6f}\n")
        print(f"✅ 저장 완료: {filename}")
    except Exception as e:
        print(f"❌ 저장 실패 ({filename}): {e}")


# ══════════════════════════════════════════════════════════════════════════
# LP Phase-1 라운드 수 — 공개 파라미터 기반 기하 산출  [2026-09 확정]
#   ★ 산출 주체는 서버다. {ε_n, N, min_pts} 는 이미 서버가 받는 값이므로
#     클라이언트가 이 값을 전송할 필요가 없고, 추가 누출도 0 비트다.
# ══════════════════════════════════════════════════════════════════════════
#
#   n_rounds = min( ⌈1/ε_n⌉ , ⌈3N/min_pts⌉ − 1 )
#       ε_n = ε / max(좌표범위, PC1 span)        ← 정규화 스케일
#
# ── 제1항: 기하 하한 ⌈1/ε_n⌉ ───────────────────────────────────────────
#   1-Lipschitz: ‖p_i−p_j‖ ≤ ε ⟹ |z_i−z_j| ≤ ε  (z = 임의 선형사영)
#   ⟹ 한 hop 은 어떤 사영축에서도 ε 이상 전진하지 못한다.
#   ⟹ extent 를 가로지르려면 ⌈extent/ε⌉ hop 이 필요하다.
#   정규화 스케일을 max(좌표범위, PC1 span) 으로 잡으면 이 값이 정확히 ⌈1/ε_n⌉ 이고,
#   동시에 데이터가 [0,1]^d 안에 유지되어 Normalize 의 max_dist_sq=dim 가정이 보존된다.
#   임의 상수(안전계수) 없음.
#
# ── 제2항: 증명된 상한 ⌈3N/min_pts⌉ − 1 ────────────────────────────────
#   성분이 전부 core 일 때 BFS 층 논증:
#       |L_{j-1}|+|L_j|+|L_{j+1}| ≥ min_pts  ⟹  N ≥ ⌊(D+1)/3⌋·min_pts
#       ⟹  D_hop ≤ ⌈3N/min_pts⌉ − 1
#   tight: 간격을 정확히 ε 로 두고 주기-3 패턴(합=min_pts)으로 뭉친 "구슬목걸이"가
#   이를 달성한다 (평문 미러: N=1000, min_pts=4 → D=747 vs 상한 749).
#   min_pts=3 이면 N−1 로 퇴화. 실데이터 11종에서는 한 번도 발동하지 않았고,
#   구슬목걸이형 극값에 대한 방어로만 남는다.
#
# ── 실측 (연구자 실사용 파라미터, 평문 미러) ───────────────────────────
#   dataset       N     d  mp      eps      제안   실제필요   여유
#   hepta       212     3   4    1.0000       8       3     2.67x
#   tetra       400     3   3    0.4300      11       7     1.57x
#   lsun        400     2   4    0.5000      12       8     1.50x
#   twodiamonds 800     2   6    0.1400      37      13     2.85x
#   longsquare  900     2   6    1.1500      33      11     3.00x
#   zelnik5     512     2   4    0.0287      27      18     1.50x
#   target      770     2   4    0.2500      32      20     1.60x
#   atom        800     3   4   20.0000       6       5     1.20x
#   twomoons    200     2   5    0.2800      13      12     1.08x
#   s1         5000     2  50 44000.0000      24       1    24.00x
#   zelnik6     238     2   3    0.0130      56       9     6.22x
#   → 11/11 안전, 여유 중앙값 1.60배. 총 fhe_max 는 기존 고정 32 대비 −30%.
#
# ── 누출 ───────────────────────────────────────────────────────────────
#   n_rounds 는 {ε_n, N, min_pts} 의 결정론적 함수이고 셋 다 이미 서버에 평문
#   전달된다 → 서버가 스스로 계산 가능 → 추가 누출 0 비트.
#   프로토콜 트랜스크립트의 길이·메시지 수는 데이터에 독립적으로 유지된다
#   (재시도/조기종료 없음).
#
# ── 한계 (논문 한계 절에 명시할 것) ────────────────────────────────────
#   (a) 사영 접힘: 측지선이 사영 extent 를 넘는 구조(다중 나선)에서는 부족하다.
#       3_spiral(N=312, ε=1.4, min_pts=3): 제안 21 < 실제 필요 55.
#       기존 고정 32 도 동일하게 실패하므로 새로 생긴 약점은 아니다.
#       공개 파라미터로는 검출 불가: 3_spiral 과 s1 은 ε_n·k_max·PC축 extent·
#       설명분산비가 사실상 동일한데 필요 라운드가 55 와 1 이다.
#   (b) 성분 분할 미인지: 식은 "하나의 성분이 PC1 extent 전체를 가로지른다"고
#       가정하므로, PC1 축을 따라 클러스터가 여러 개 늘어서면 대략 그 개수만큼
#       과잉 계산된다 (s1: 15개 클러스터 → 24배).
#   (c) 부족해도 런타임에 신호가 없다(조용한 under-merge). 재시도를 넣지 않는 것은
#       의도적이다 — 재시도는 "기하 추정이 부족했다"는 1비트를 서버에 노출하는데,
#       그 1비트는 어떤 공개 파라미터도 제공하지 못하는 정보다(위 (a) 참조).
#       새 데이터셋 편입 시에는 오프라인 평문 검증으로 사전에 확인할 것.


def n_rounds_from_public(eps_norm: float, N: int, min_pts: int) -> int:
    """공개 파라미터만으로 LP Phase-1 라운드 수를 산출한다.

    Parameters
    ----------
    eps_norm : float   정규화된 ε  ( = ε / max(좌표범위, PC1 span) )
    N        : int     점 개수
    min_pts  : int     DBSCAN MinPts

    Returns
    -------
    int : n_rounds ≥ 1
    """
    if eps_norm <= 0.0:
        raise ValueError(f"eps_norm must be positive, got {eps_norm}")
    geo = math.ceil(1.0 / eps_norm)                  # 기하 하한 (1-Lipschitz)
    cap = math.ceil(3.0 * N / max(min_pts, 1)) - 1   # 증명된 상한 (BFS 층 논증)
    return max(1, min(geo, cap))

# 전파 방식 임계값: 상단 정의로 모든 함수에서 참조 가능


def send_to_server_fhe(
    engine, keypack,
    encrypted_columns,
    num_points, eps, min_pts,
    k_max: int,                       # 클라이언트 PCA window 상한 (필수)
    n_rounds: int = None,             # [2026-10] 클라이언트 grid 산출값. None 이면 대비 경로
    *,
    debug_secret_key=None,            # ★ 디버그 전용. 프로토콜은 이 값 없이 완결된다.
):
    """
    서버 메인 파이프라인.

    Parameters
    ----------
    k_max : int
        클라이언트 PCA window 상한으로 결정된 값.
        서버는 이 값을 그대로 사용하며 재계산하지 않음.
        T(k_max) = k_max*(k_max+1)//2 ≥ N → kd_dense 수렴 보장.
    n_rounds : int
        kd_dense 의 Core-Core 전파 반복 횟수.
        [2026-10] 클라이언트가 GridRounds.compute_grid_rounds 로 산출해 전달하는 것이 기본
        (n = max(1, max_g ⌈(Λ_g + s√d)/ε_n⌉)). 서버는 수신값을 그대로 쓴다.
        None 이면 대비 경로로 n_rounds_from_public(eps, N, min_pts) 를 쓴다.
    debug_secret_key
        ★ 개발용 진단 전용. 주지 않으면 서버는 어떤 중간 암호문도 복호하지 않으며,
        프로토콜은 이 값 없이 완결된다. 보고된 모든 결과는 None 으로 실행했다.
    """
    dim  = len(encrypted_columns)
    N    = num_points
    # ★ [2026-09] n_rounds 는 서버가 공개 파라미터로 직접 산출한다.
    #   {eps(=ε_n), N, min_pts} 는 이미 이 함수가 받는 값이므로 추가 전송·누출이 없다.
    #   (이전 기본값 ⌈log₂N⌉ 은 수렴 보장이 아니라 휴리스틱이었고 실측 반증되었다:
    #    chainlink log₂1000=10 < 실제 필요 12, target log₂770=10 < 11 → under-merge.)
    if n_rounds is not None:
        n_rounds = int(n_rounds)
        print(f"[Server] n_rounds={n_rounds} (클라이언트 grid 산출값 수신)")
    else:
        n_rounds = n_rounds_from_public(eps, N, int(min_pts))
        print(f"[Server] n_rounds={n_rounds} 산출  "
              f"(기하 ⌈1/ε_n⌉={math.ceil(1.0/eps)}, "
              f"상한 ⌈3N/min_pts⌉-1={math.ceil(3.0*N/max(int(min_pts),1))-1})")

    T_kmax = k_max * (k_max + 1) // 2

    adj_k_list         = []
    total_neighbors_ct = None
    debug_fhe          = {}
    timings            = {}

    _print_mem("send_to_server_fhe() 진입")
    print(f"  N={N}  dim={dim}  eps^2={eps**2:.4f}  min_pts={min_pts}")
    print(f"  k_max={k_max}  T(k_max)={T_kmax}  전파방식=KD-dense  n_rounds={n_rounds}")

    # ══════════════════════════════════════════════════════════════
    # Step 1. Normalize: adj_k 계산 (k=1..k_max, 대칭 최적화)
    # adj_{N-k}[i] = adj_k[(i-k) mod N] = rotate(adj_k, N-k)[i]
    # → k=1..k_max만 MCP 계산, 역방향은 회전으로 유도
    #
    # ★ [2026-07] 루프 상한 N//2 → k_max.
    #   [근거] 1-Lipschitz window 상한: stride > k_max 인 두 점은 |z_i−z_j| > eps
    #     ⟹ ‖p_i−p_j‖ > eps ⟹ 이웃 아님 ⟹ adj_k = 0. 계산해도 전부 0.
    #   [검증] 7개 FCPS 전부에서 total_neighbors(k=1..k_max) == total_neighbors(k=1..N//2)
    #     최대차 0.0000 (Core 판정 불변). k>k_max 의 adj_k 는 합산·회전에 0 만 기여.
    #   [효과] Normalize MCP 호출 N/2 → k_max.  tetra 62% / moons 80% / chainlink 80% 절감.
    #     퇴화 케이스(atom 등 등방분포, k_max=N//2)에서는 절감 0 이나 정확성 유지.
    #   [대칭 유도] 아래 adj_{N-k} = rotate(adj_k, N-k) 는 그대로 유효.
    #     순환상 stride s 인 간선은 min(s, N-s) ≤ k_max 이면 커버되며,
    #     k=1..k_max forward + rotate 역방향이 min(s,N-s) ≤ k_max 를 모두 포함한다.
    # ★ [2026-07 진단] Normalize 가 표준 bootstrap(13.4GB 키)을 쓰는지 계측
    from core.ciphertext_single.cleaning import std_boot_count
    std_boot_count(reset=True)
    print(f"\n[Step 1] Normalize 시작 (k=1..{k_max}, eps^2={eps**2:.4f})")
    normalize_start = time()

    # ★ [2026-07 메모리] Step 1 은 다음 두 가지로 GPU 메모리를 소모한다.
    #   (a) adj_k_list: k_max 개 암호문을 **끝까지 보유**한다(LP 의 adjm_cache 가 필요).
    #       N=800/k_max=83 이면 신선 암호문 ~28MB × 83 ≈ 2.3GB.
    #   (b) 루프 중간값: rotated_col / diff_ct / sq_ct / dist_sq_k / adj_Nk 가
    #       매 반복 dim 회 생성되는데 명시적으로 해제되지 않아, GC 시점에 따라
    #       여러 세대가 동시에 살아있을 수 있다.
    #   키가 이미 18.6GB(그중 bootstrap_key 13.4GB)를 점유한 상태라 여유가 ~6GB 뿐이라
    #   (b) 가 쌓이면 OOM 이 난다.
    #   ⇒ 아래는 (b) 를 즉시 해제하고 주기적으로 gc 를 돌린다. 알고리즘/정확도 변경 없음.
    #   ※ 되돌리려면 이 블록을 원래 루프로 교체하면 된다(로직 동일).
    _MEM_AGGRESSIVE_FREE = True     # False 면 기존 동작(해제 없음)
    _GC_EVERY = 8                   # 몇 stride 마다 gc.collect()

    for k in range(1, k_max + 1):
        dist_sq_k = None
        for d in range(dim):
            base_col    = encrypted_columns[d]
            rotated_col = fhe_circular_shift(engine, base_col, k, N, keypack)
            diff_ct     = engine.subtract(base_col, rotated_col)
            if _MEM_AGGRESSIVE_FREE:
                del rotated_col
            sq_ct       = engine.square(diff_ct, keypack.relinearization_key)
            if _MEM_AGGRESSIVE_FREE:
                del diff_ct
            if dist_sq_k is None:
                dist_sq_k = sq_ct
            else:
                _prev     = dist_sq_k
                dist_sq_k = engine.add(_prev, sq_ct)
                if _MEM_AGGRESSIVE_FREE:
                    del _prev, sq_ct

        before_adj = _gpu_used_mb()
        adj_k = check_neighbor_closed_interval(
            engine, dist_sq_k, eps**2, keypack, dim,
            num_points=N,          # ★ N 기반 α 자동선택(⌈log₂N⌉+8). mcp_path 고정 제거.
        )
        if _MEM_AGGRESSIVE_FREE:
            del dist_sq_k               # sign-eval 끝나면 즉시 불필요
        adj_k_list.append(adj_k)   # k=1..k_max 저장

        # total_neighbors 누적 (대칭 최적화)
        if 2 * k < N:
            adj_Nk = fhe_circular_shift(engine, adj_k, N - k, N, keypack)
            _pair  = engine.add(adj_k, adj_Nk)
            if _MEM_AGGRESSIVE_FREE:
                del adj_Nk
            if total_neighbors_ct is None:
                total_neighbors_ct = _pair
            else:
                _acc = total_neighbors_ct
                total_neighbors_ct = engine.add(_acc, _pair)
                if _MEM_AGGRESSIVE_FREE:
                    del _acc, _pair
        else:   # k == N//2: double counting 방지
            if total_neighbors_ct is None:
                total_neighbors_ct = adj_k
            else:
                _acc = total_neighbors_ct
                total_neighbors_ct = engine.add(_acc, adj_k)
                if _MEM_AGGRESSIVE_FREE:
                    del _acc

        if _MEM_AGGRESSIVE_FREE and (k % _GC_EVERY == 0):
            gc.collect()

        if k % 10 == 0 or k == N // 2:
            _mem_delta(f"adj_k[{k}] 생성 (누적 {k}회 MCP)", before_adj)

    # 자기 자신을 이웃으로 포함 (DBSCAN 정의: 자기 자신 포함 min_pts 이상)
    ones_pt            = engine.encode([1.0] * N + [0.0] * (engine.slot_count - N))
    total_neighbors_ct = engine.add(total_neighbors_ct, ones_pt)

    timings["normalize_sec"] = time() - normalize_start
    print(f"[TIME] Normalize: {timings['normalize_sec']:.2f}초")
    _n_std = std_boot_count(reset=True)
    print(f"  [진단] Normalize 구간 표준 bootstrap 호출 수 = {_n_std}회"
          f"  {'→ bootstrap_key 불필요! 생성 지연 가능(13.4GB 확보)' if _n_std == 0 else '→ bootstrap_key 필요'}")
    _print_mem(f"Normalize 완료 (adj_k_list {len(adj_k_list)}개 = k=1..{k_max})")

    # ★ [2026-09] 아래 블록은 전적으로 개발용 진단이다. debug_secret_key 를 주지 않는
    #   운영 경로에서는 서버가 어떤 중간 암호문도 복호하지 않는다.
    if debug_secret_key is not None:
        dec_total = np.real(engine.decrypt(total_neighbors_ct, debug_secret_key)[:N])
        debug_fhe["total_neighbors"] = np.array(dec_total)
        print(f"\n[DEBUG] total_neighbors (앞 10개): {np.round(dec_total[:10], 2)}")
        save_vector_csv(
            f"debug_normalize_eps{eps:.4f}_min{int(min_pts)}.csv",
            dec_total, "Point_ID,Total_Neighbors"
        )

    # [변경 6] 절단 블록 삭제: 루프가 k=1..k_max 만 생성하므로 len(adj_k_list) == k_max.
    #   (절단이 실제로 일어나던 경우는 N//2 까지 만들던 sweep 경로뿐이었다.)

    debug_fhe["k_max_used"] = k_max

    # ══════════════════════════════════════════════════════════════
    # Step 2. Core Point 판별
    # ══════════════════════════════════════════════════════════════
    print(f"\n[Step 2] Core Point 판별 시작")
    before_core = _gpu_used_mb()
    core_start  = time()

    core_ct = identify_core_points_fhe(
        engine, total_neighbors_ct, min_pts, N, keypack=keypack
    )

    timings["core_sec"] = time() - core_start
    _mem_delta("identify_core_points_fhe() 완료", before_core)
    print(f"[TIME] Core: {timings['core_sec']:.2f}초")

    if debug_secret_key is not None:
        dec_core = np.real(engine.decrypt(core_ct, debug_secret_key)[:N])
        debug_fhe["core_mask"] = np.array(dec_core)
        n_core = int(np.sum(dec_core[:N] > 0.5))
        print(f"[DEBUG] Core 마스크 (앞 10개): {np.round(dec_core[:10], 4)}")
        print(f"  → Core 포인트: {n_core}/{N}")
        if n_core == 0:
            print("  ❌ [FATAL] 코어 포인트 미검출!")
        save_vector_csv(
            f"debug_core_eps{eps:.4f}_min{int(min_pts)}.csv",
            dec_core, "Point_ID,Core_Mask"
        )

    # ══════════════════════════════════════════════════════════════
    # Step 3. Label Propagation
    # ══════════════════════════════════════════════════════════════
    print(f"\n[Step 3] Label Propagation 시작")
    before_lp = _gpu_used_mb()
    lp_start  = time()

    print(f"  방식: KD-dense  k=1..{k_max}, n_rounds={n_rounds}")
    print(f"  T({k_max})={T_kmax} {'≥' if T_kmax >= N else '<'} N={N}  "
          f"(fhe_max≈{(2*n_rounds+2)*k_max*2}회)")

    final_ct = fhe_kd_dense_propagation(
        engine, keypack,
        adj_k_half_list=adj_k_list,
        core_ct=core_ct,
        num_points=N,
        k_max=k_max,
        secret_key=debug_secret_key,
        n_rounds=n_rounds,
    )

    timings["label_propagation_sec"] = time() - lp_start
    _mem_delta("Label Propagation 완료", before_lp)
    print(f"[TIME] Label_Propagation: {timings['label_propagation_sec']:.2f}초")

    if debug_secret_key is not None:
        dec_final = np.real(engine.decrypt(final_ct, debug_secret_key)[:N])
        debug_fhe["final_labels"] = np.array(dec_final)
        print(f"\n[DEBUG] 최종 라벨 (Heap 순서, 앞 10개): {np.round(dec_final[:10], 2)}")
        print(f"  범위: min={dec_final.min():.2f}, max={dec_final.max():.2f}  (정상: [0,{N}])")
        print(f"  ※ 클라이언트에서 inv_perm 적용 후 원래 순서 복원")
        save_vector_csv(
            f"debug_labelprop_final_eps{eps:.4f}_min{int(min_pts)}.csv",
            dec_final, "Point_ID,Final_Label_Heap_Order"
        )

    debug_fhe["timings"] = timings
    _print_mem("send_to_server_fhe() 완료")
    return final_ct, debug_fhe
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
    fhe_kd_dense_propagation,   # ★ dense stride k=1..k_max (power-of-2 누락 수정)
    fhe_sweep_propagation,      # fallback 유지
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


def send_to_server_fhe(
    engine, keypack, secret_key,
    encrypted_columns,
    num_points, eps, min_pts,
    k_max: int,                       # 클라이언트 Ball Tree 구조 분석 결과 (필수)
    use_kd_propagation: bool = True,
    num_sweeps: int = None,
    n_rounds: int = None,             # ★ [2026-05c 작업 B] LP Core-Core 반복 횟수
):
    """
    서버 메인 파이프라인.

    Parameters
    ----------
    k_max : int
        클라이언트 PCA window 상한으로 결정된 값.
        서버는 이 값을 그대로 사용하며 재계산하지 않음.
        T(k_max) = k_max*(k_max+1)//2 ≥ N → kd_dense 수렴 보장.
    use_kd_propagation : bool
        ★ [2026-05c] 기본 True (kd_dense 통일 권장).
        all-sweep은 mask damping이 누적 곱셈으로 라벨을 0으로 소멸시켜
        iris 등에서 -1 완전 붕괴 발생 (실측). 작업 A로 mask=1.0 확보 +
        작업 B로 적응적 round 사용 시 kd_dense가 모든 경우를 커버하므로
        sweep은 deprecated 예정. (호환 위해 인자는 유지.)
    num_sweeps : int
        [deprecated] sweep 방식 잔존 시에만 사용.
    n_rounds : int
        ★ [2026-05c 작업 B] kd_dense의 Core-Core 전파 반복 횟수.
        None이면 ⌈log₂N⌉ (검증: 2*log₂N pass가 ARI>=0.9 커버).
        작업 A(mask=1.0)가 안전망이라 과대 추정해도 라벨 안 죽음.
    """
    dim  = len(encrypted_columns)
    N    = num_points
    if num_sweeps is None:
        num_sweeps = math.ceil(math.log2(N))
    if n_rounds is None:
        n_rounds = math.ceil(math.log2(N))   # ★ 작업 B: log₂N round 기본값

    T_kmax = k_max * (k_max + 1) // 2

    adj_k_list         = []
    total_neighbors_ct = None
    debug_fhe          = {}
    timings            = {}

    _print_mem("send_to_server_fhe() 진입")
    print(f"  N={N}  dim={dim}  eps^2={eps**2:.4f}  min_pts={min_pts}")
    print(f"  k_max={k_max}  T(k_max)={T_kmax}  "
          f"전파방식={'KD-dense' if use_kd_propagation else f'ALL-sweep ({num_sweeps}회)'}")

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
    _k_upper = k_max if use_kd_propagation else (N // 2)
    # ★ [2026-07 진단] Normalize 가 표준 bootstrap(13.4GB 키)을 쓰는지 계측
    from core.ciphertext_single.cleaning import std_boot_count
    std_boot_count(reset=True)
    print(f"\n[Step 1] Normalize 시작 (k=1..{_k_upper}, eps^2={eps**2:.4f})")
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

    for k in range(1, _k_upper + 1):
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
        adj_k_list.append(adj_k)   # k=1..N//2 저장

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
    _print_mem(f"Normalize 완료 (adj_k_list {len(adj_k_list)}개 = k=1..{_k_upper})")

    dec_total = np.real(engine.decrypt(total_neighbors_ct, secret_key)[:N])
    debug_fhe["total_neighbors"] = np.array(dec_total)
    print(f"\n[DEBUG] total_neighbors (앞 10개): {np.round(dec_total[:10], 2)}")
    save_vector_csv(
        f"debug_normalize_eps{eps:.4f}_min{int(min_pts)}.csv",
        dec_total, "Point_ID,Total_Neighbors"
    )

    # adj_k_list 트런케이트: ★ [2026-07] 루프가 이미 k_max 까지만 생성하므로 보통 no-op.
    #   (use_kd_propagation=False 인 all_sweep 경로는 N//2 까지 생성 → 이때만 실제 절단)
    if use_kd_propagation and len(adj_k_list) > k_max:
        adj_k_list = adj_k_list[:k_max]
        print(f"  adj_k_list 트런케이트: → {k_max}개 (k_max 이내로 제한)")
    elif not use_kd_propagation:
        print(f"  adj_k_list 유지: {len(adj_k_list)}개 (sweep 모드는 전체 stride 사용)")

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

    dec_core = np.real(engine.decrypt(core_ct, secret_key)[:N])
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

    if use_kd_propagation:
        print(f"  방식: KD-dense  k=1..{k_max}, n_rounds={n_rounds} (★ 작업 B)")
        print(f"  T({k_max})={T_kmax} {'≥' if T_kmax >= N else '<'} N={N}  "
              f"(fhe_max≈{(2*n_rounds+2)*k_max*2}회)")

        final_ct = fhe_kd_dense_propagation(
            engine, keypack,
            adj_k_half_list=adj_k_list,
            core_ct=core_ct,
            num_points=N,
            k_max=k_max,
            secret_key=secret_key,
            n_rounds=n_rounds,          # ★ 작업 B
        )
    else:
        print(f"  방식: ALL strides sweep  num_sweeps={num_sweeps}")
        print(f"  fhe_max≈{2 * num_sweeps * (N//2) * 2}회")

        final_ct = fhe_sweep_propagation(
            engine, keypack,
            adj_k_half_list=adj_k_list,
            core_ct=core_ct,
            num_points=N,
            secret_key=secret_key,
            num_sweeps=num_sweeps,
        )

    timings["label_propagation_sec"] = time() - lp_start
    _mem_delta("Label Propagation 완료", before_lp)
    print(f"[TIME] Label_Propagation: {timings['label_propagation_sec']:.2f}초")

    dec_final = np.real(engine.decrypt(final_ct, secret_key)[:N])
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
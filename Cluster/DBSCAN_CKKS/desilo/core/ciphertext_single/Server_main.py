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
    mask_mode: str = "per_stride",    # ★ [Tier 1b] LP 마스킹 빈도 ("per_stride"|"per_pass")
    debug_adj_sample=(1, 2, 5, 10),   # ★ [2026-06] 이 stride들의 adj_k를 복호화해 점검
    lp_snapshot: bool = False,        # ★ [2026-06] LP pass별 라벨 CSV 저장 여부
):
    """
    서버 메인 파이프라인.

    Parameters
    ----------
    k_max : int
        클라이언트 Ball Tree DFS 구조 분석으로 결정된 값.
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
    # Step 1. Normalize: adj_k 계산 (k=1..N//2, 대칭 최적화)
    # adj_{N-k}[i] = adj_k[(i-k) mod N] = rotate(adj_k, N-k)[i]
    # → k=1..N//2만 MCP 계산, 역방향은 회전으로 유도
    # ══════════════════════════════════════════════════════════════
    print(f"\n[Step 1] Normalize 시작 (k=1..{N//2}, eps^2={eps**2:.4f})")
    normalize_start = time()

    for k in range(1, N // 2 + 1):
        dist_sq_k = None
        for d in range(dim):
            base_col    = encrypted_columns[d]
            rotated_col = fhe_circular_shift(engine, base_col, k, N, keypack)
            diff_ct     = engine.subtract(base_col, rotated_col)
            sq_ct       = engine.square(diff_ct, keypack.relinearization_key)
            dist_sq_k   = sq_ct if dist_sq_k is None else engine.add(dist_sq_k, sq_ct)

        before_adj = _gpu_used_mb()
        adj_k = check_neighbor_closed_interval(
            engine, dist_sq_k, eps**2, keypack, dim,
            mcp_path="mcp_alpha15_lp_cheb.json",
        )
        adj_k_list.append(adj_k)   # k=1..N//2 저장

        # total_neighbors 누적 (대칭 최적화)
        if 2 * k < N:
            adj_Nk = fhe_circular_shift(engine, adj_k, N - k, N, keypack)
            total_neighbors_ct = (
                engine.add(adj_k, adj_Nk)
                if total_neighbors_ct is None
                else engine.add(total_neighbors_ct, engine.add(adj_k, adj_Nk))
            )
        else:   # k == N//2: double counting 방지
            total_neighbors_ct = (
                adj_k
                if total_neighbors_ct is None
                else engine.add(total_neighbors_ct, adj_k)
            )

        if k % 10 == 0 or k == N // 2:
            _mem_delta(f"adj_k[{k}] 생성 (누적 {k}회 MCP)", before_adj)

    # 자기 자신을 이웃으로 포함 (DBSCAN 정의: 자기 자신 포함 min_pts 이상)
    ones_pt            = engine.encode([1.0] * N + [0.0] * (engine.slot_count - N))
    total_neighbors_ct = engine.add(total_neighbors_ct, ones_pt)

    timings["normalize_sec"] = time() - normalize_start
    print(f"[TIME] Normalize: {timings['normalize_sec']:.2f}초")
    _print_mem(f"Normalize 완료 (adj_k_list {len(adj_k_list)}개 = k=1..{N//2})")

    dec_total = np.real(engine.decrypt(total_neighbors_ct, secret_key)[:N])
    debug_fhe["total_neighbors"] = np.array(dec_total)
    print(f"\n[DEBUG] total_neighbors (앞 10개): {np.round(dec_total[:10], 2)}")
    save_vector_csv(
        f"debug_normalize_eps{eps:.4f}_min{int(min_pts)}.csv",
        dec_total, "Point_ID,Total_Neighbors"
    )

    # ── ★ [2026-06] neighbor 디버그: 샘플 stride의 adj_k를 복호화해 점검 ──
    #   "이웃이 각각 어떻게 나오는지" = stride k에서 (i, i+k)가 이웃인지의 0/1 벡터.
    #   확인 포인트: ① 그 stride의 이웃-쌍 개수(=1인 슬롯 수),
    #               ② 값이 깨끗한 0/1인지(intermediate 값 = sign 근사 경계 실패).
    if debug_adj_sample:
        print(f"[DEBUG][neighbor] 샘플 stride adj_k 점검 (1=이웃, 0=비이웃)")
        adj_rows = []
        for k in debug_adj_sample:
            if k < 1 or k > len(adj_k_list):
                continue
            v = np.real(engine.decrypt(adj_k_list[k - 1], secret_key)[:N])
            n_pair  = int(np.sum(v > 0.5))
            n_ambig = int(np.sum((v > 0.1) & (v < 0.9)))   # 모호값(경계 실패 신호)
            print(f"   k={k:>3}: 이웃쌍={n_pair:>4}  "
                  f"min={v.min():+.4f} max={v.max():+.4f}  모호슬롯(0.1~0.9)={n_ambig}")
            adj_rows.append((k, v))
        # 전체 stride별 이웃쌍 분포까지 보고 싶으면 lp_snapshot=True 시 저장
        if lp_snapshot and adj_rows:
            with open(f"debug_adj_sample_eps{eps:.4f}_min{int(min_pts)}.csv",
                      "w", encoding="utf-8") as f:
                f.write("k,Point_ID,adj_value\n")
                for k, v in adj_rows:
                    for i, val in enumerate(v):
                        f.write(f"{k},{i},{float(val):.6f}\n")
            print(f"   ↳ adj 샘플 저장: debug_adj_sample_eps{eps:.4f}_min{int(min_pts)}.csv")

    # adj_k_list 트런케이트: k_max 초과분은 LP에서 불필요 (메모리 절약)
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
    n_ambig_core = int(np.sum((dec_core > 0.1) & (dec_core < 0.9)))  # ★ 경계 실패 신호
    print(f"[DEBUG] Core 마스크 (앞 10개): {np.round(dec_core[:10], 4)}")
    print(f"  → Core 포인트: {n_core}/{N}")
    # ★ [2026-06] "core여부 값이 어떻게 나오는지": 1.0/0.0에 얼마나 붙었는지 + 모호값
    print(f"  → 값 분포: ~1(>0.9)={int(np.sum(dec_core>0.9))}  "
          f"~0(<0.1)={int(np.sum(dec_core<0.1))}  모호(0.1~0.9)={n_ambig_core}")
    print(f"  → 코어 슬롯 1과의 최대 오차={np.max(np.abs(dec_core[dec_core>0.5]-1.0)) if n_core else 0:.2e} "
          f"(0에 가까울수록 mask=1.0 — LP 감쇠 위험↓)")
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
            mask_mode=mask_mode,        # ★ Tier 1b ("per_stride"|"per_pass")
            debug_snapshot_prefix=(
                f"debug_lp_eps{eps:.4f}_min{int(min_pts)}_{mask_mode}"
                if lp_snapshot else None),
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
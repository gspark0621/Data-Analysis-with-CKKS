# core/ciphertext_single/Server_main.py
#
# ── 변경사항 ────────────────────────────────────────────────────────────────
# [변경 1] send_to_server_fhe 파라미터 추가
#   - use_kd_propagation: True → fhe_kd_dense_propagation (k_max 기반 dense stride)
#                         False → fhe_sweep_propagation (fallback)
#
# [변경 2] fhe_kd_dense_propagation import 및 사용
#   - k_max = min(N//2, 3×ceil(√N)) : T(k_max)≥N → 1 sweep 수렴 보장
#   - dense k=1..k_max: power-of-2 stride 누락 문제 해결
#
# [변경 3] Normalize mcp_path="mcp_normalize_alpha12.json"
#   - alpha=12: margin 0.012→0.00077, false positive zone 32%→2.4% 해결
#
# [유지] Normalize, Core, adj 대칭 최적화, 디버그 코드 모두 동일
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
    encrypted_columns, num_points, eps, min_pts,
    k_max: int = None,
    use_kd_propagation: bool = True,
    num_sweeps: int = None,
):
    """
    서버 메인 파이프라인.

    Parameters
    ----------
    k_max : int
        dense stride 상한. Client_main.decide_propagation_mode에서 전달.
        None이면 내부 계산: min(N//2, 3×ceil(√N)).
        T(k_max) ≥ N → 연속 stride ≤ k_max인 모든 cluster 1 sweep 수렴 보장.
    use_kd_propagation : bool
        True  → fhe_kd_dense_propagation  (min_pts≥4, Heap 정렬 전제, 고속+정확)
        False → fhe_sweep_propagation     (min_pts<4, chain형 허용)
    num_sweeps : int
        sweep 방식 사용 시 sweep 횟수.

    주의:
        use_kd_propagation=True 시 encrypted_columns는
        Client_main.build_kd_tree_order로 정렬된 데이터여야 함.
    """
    dim  = len(encrypted_columns)
    N    = num_points
    if k_max is None:
        k_max = min(N // 2, 3 * math.ceil(math.sqrt(N)))
    if num_sweeps is None:
        num_sweeps = math.ceil(math.log2(N))

    adj_k_list         = []
    total_neighbors_ct = None
    debug_fhe          = {}
    timings            = {}

    _print_mem("send_to_server_fhe() 진입")
    print(f"  N={N}  dim={dim}  eps^2={eps**2:.4f}  min_pts={min_pts}")
    T_kmax = k_max * (k_max + 1) // 2
    print(f"  k_max={k_max}  T(k_max)={T_kmax}  전파방식={'KD-dense' if use_kd_propagation else f'ALL-sweep ({num_sweeps}회)'}")

    # ══════════════════════════════════════════════════════════════
    # Step 1. Normalize: adj_k 계산 (k=1..N//2, 대칭 최적화)
    # adj_{N-k}[i] = adj_k[(i-k) mod N] = rotate(adj_k, N-k)[i]
    # → k=1..N//2만 MCP 계산, 역방향은 회전으로 유도
    # ══════════════════════════════════════════════════════════════
    print(f"\n[DEBUG] 1. Normalize 시작... eps^2={eps**2:.4f}")
    print(f"  adj 대칭 최적화: k=1..{N//2} MCP ({N//2}회) + k>{N//2} 회전 유도")
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
        adj_k      = check_neighbor_closed_interval(
            engine, dist_sq_k, eps**2, keypack, dim,
            mcp_path="mcp_normalize_alpha12.json",  # ★ alpha=12: false positive 32%→2.4%
        )
        adj_k_list.append(adj_k)  # k=1..N//2 저장

        if 2 * k < N:
            adj_Nk = fhe_circular_shift(engine, adj_k, N - k, N, keypack)
            total_neighbors_ct = (
                engine.add(adj_k, adj_Nk)
                if total_neighbors_ct is None
                else engine.add(total_neighbors_ct, engine.add(adj_k, adj_Nk))
            )
        else:  # k == N//2: double counting 방지
            total_neighbors_ct = (
                adj_k
                if total_neighbors_ct is None
                else engine.add(total_neighbors_ct, adj_k)
            )

        if k % 10 == 0 or k == N // 2:
            _mem_delta(f"adj_k[{k}] 생성 (누적 {k}회 MCP)", before_adj)

        if k == N // 2:
            valid_adj = np.real(engine.decrypt(adj_k, secret_key)[:N])
            print(f"  -> k={k}(N//2) 이웃 배열 (하위 10개): {np.round(valid_adj[-10:], 4)}")

    ones_pt            = engine.encode([1.0] * N + [0.0] * (engine.slot_count - N))
    total_neighbors_ct = engine.add(total_neighbors_ct, ones_pt)

    timings["normalize_sec"] = time() - normalize_start
    print(f"[TIME] Normalize: {timings['normalize_sec']:.2f}초")
    _print_mem(f"Normalize 완료 (adj_k_half_list {len(adj_k_list)}개 = k=1..{N//2})")

    dec_total = np.real(engine.decrypt(total_neighbors_ct, secret_key)[:N])
    debug_fhe["total_neighbors"] = np.array(dec_total)
    print(f"\n[DEBUG] 2. 총 이웃 수 (앞 10개): {np.round(dec_total[:10], 2)}")
    save_vector_csv(f"debug_normalize_eps{eps:.4f}_min{int(min_pts)}.csv",
                    dec_total, "Point_ID,Total_Neighbors")

    # ══════════════════════════════════════════════════════════════
    # Step 2. Core Point 판별
    # ══════════════════════════════════════════════════════════════
    before_core = _gpu_used_mb()
    core_start  = time()
    core_ct     = identify_core_points_fhe(
        engine, total_neighbors_ct, min_pts, N, keypack=keypack
    )
    timings["core_sec"] = time() - core_start
    _mem_delta("identify_core_points_fhe() 완료", before_core)
    print(f"[TIME] Core: {timings['core_sec']:.2f}초")

    dec_core = np.real(engine.decrypt(core_ct, secret_key)[:N])
    debug_fhe["core_mask"] = np.array(dec_core)
    n_core = int(np.sum(dec_core[:N] > 0.5))
    print(f"\n[DEBUG] 3. Core 마스크 (앞 10개): {np.round(dec_core[:10], 4)}")
    print(f"  -> Core 포인트: {n_core}/{N}")
    if n_core == 0:
        print("  ❌ [FATAL] 코어 포인트 미검출!")
    save_vector_csv(f"debug_core_eps{eps:.4f}_min{int(min_pts)}.csv",
                    dec_core, "Point_ID,Core_Mask")

    # ══════════════════════════════════════════════════════════════
    # Step 3. Label Propagation
    # ══════════════════════════════════════════════════════════════
    print(f"\n[DEBUG] 4. Label Propagation 시작...")

    before_lp = _gpu_used_mb()
    lp_start  = time()

    if use_kd_propagation:
        # KD-tree dense stride (min_pts≥4 보장)
        T_kmax = k_max * (k_max + 1) // 2
        print(f"  방식: KD-dense k=1..{k_max}")
        print(f"  T({k_max})={T_kmax} {'≥' if T_kmax >= N else '<'} N={N}")
        print(f"  fhe_max≈{2*2*k_max*2}회 (ALL-sweep {2*num_sweeps*(N//2)*2}회 대비 {2*num_sweeps*(N//2)*2//(2*2*k_max*2)}배↓)")

        final_ct = fhe_kd_dense_propagation(
            engine, keypack,
            adj_k_half_list=adj_k_list,
            core_ct=core_ct,
            num_points=N,
            k_max=k_max,
            secret_key=secret_key,
        )
    else:
        # ALL strides sweep (min_pts<4, chain형)
        print(f"  방식: ALL strides sweep (num_sweeps={num_sweeps})")
        print(f"  fhe_max: {2 * num_sweeps * (N//2) * 2}회")

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
    print(f"\n[DEBUG] 5. 최종 라벨 (Heap 순서, 앞 10개): {np.round(dec_final[:10], 2)}")
    print(f"  범위: min={dec_final.min():.2f}, max={dec_final.max():.2f}  (정상: [0,{N}])")
    print(f"  ※ 클라이언트에서 inv_perm 적용 후 원래 순서 복원")
    save_vector_csv(f"debug_labelprop_final_eps{eps:.4f}_min{int(min_pts)}.csv",
                    dec_final, "Point_ID,Final_Label_Heap_Order")

    debug_fhe["timings"] = timings
    _print_mem("send_to_server_fhe() 완료")
    return final_ct, debug_fhe
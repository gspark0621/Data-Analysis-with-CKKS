# test_Server_main_call_dataset.py
#
# ── 변경사항 ────────────────────────────────────────────────────────────────
# [변경 1] KD-tree → Ball Tree 교체 (유지)
#
# [변경 2] prepare_client_ordering으로 통합 (유지)
#
# [변경 3] k_max 결정 방식 단순화                          ← 수정
#   기존: 서버 enc_sum_k 계산 → 클라이언트 복호화 → k_max 정밀화
#   변경: 클라이언트 PCA window 상한 결과를 최종값으로 사용.
#         서버는 수신한 k_max를 그대로 사용하며 재계산하지 않음.
#
# [변경 4] (2026-10 삭제) min_pts 로 kd_dense/sweep 을 고르던 분기(4 → 2×dim).
#   sweep 경로가 코드에서 사라져 분기 자체가 없다. LP 는 항상 kd_dense.
# ─────────────────────────────────────────────────────────────────────────
#
# [변경 5, 2026-05b → 2026-09 폐기] 옵션 B(전부 α=15 통일)
#   - 당시: 단일 파일 mcp_alpha15_lp_cheb.json 을 Normalize/Core/LP 가 공유.
#   - 현재: 세 모듈 모두 N 기반 가변 α 를 쓴다.
#       Core      : clamp(⌈log₂2N⌉+2, 8, 16)   → mcp_alpha{α}_lp_cheb.json
#       LP        : clamp(⌈log₂1.1N⌉+2, 8, 16) → mcp_alpha{α}_lp_clean1.json
#       Normalize : clamp(⌈log₂N⌉+8, 15, 20)   → mcp_alpha{α}_norm_cheb.json
#     따라서 α=15 공유 파일은 참조되지 않으며, 이를 만들던 STEP 0 블록을 삭제했다.
#
# [변경 6, 2026-05c] 정확도 버그 수정 통합
#   - 작업 A: Core/Normalize 마지막 일반 bootstrap → bit_cleaning (mask noise 제거)
#     + LP의 core_mask _refresh 직후 bit_cleaning 추가 (0.714 감쇠 버그 수정)
#   - [2026-09] n_rounds = min(⌈1/ε_n⌉, ⌈3N/min_pts⌉−1), Step 2 에서 산출
#   - [2026-10] n_rounds 를 클라이언트 grid(셀 ε_n/2, 측지 지름)로 산출해 명시 전달 (GridRounds)
#               send_to_server_fhe 의 디버그 키 인자명 secret_key → debug_secret_key 로 정정
#   - 작업 C: verify_convergence.py (FHE 불필요 평문 수렴 검증, 별도 도구)
#   - sgn 정밀화: fhe_sgn에 sign_cleaning 추가 (fhe_max 라벨 단조하강 수정)
#   - sweep 폐기 → kd_dense 통일 (prepare_client_ordering 항상 'kd_dense' 반환)
#   - [2026-10] sweep 코드 삭제: prepare_client_ordering(norm_pts, eps_norm, N) 은
#               (k_max, order, inv_perm) 를 반환하고, send_to_server_fhe 의
#               use_kd_propagation / num_sweeps 인자는 없어졌다.
# ────────────────────────────────────────────────────────────────────────────

import os
import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from time import time
import pynvml
from sklearn.metrics import adjusted_rand_score

from core.ciphertext_single.Client_main import (
    setup_fhe_engine,
    prepare_client_ordering,     # ★ Ball Tree + k_max 구조 분석 통합
    assign_clusters_by_gap,      # ★ [2026-07] 간격 기반 클러스터 판정
)
from core.ciphertext_single.Server_main import send_to_server_fhe
from core.ciphertext_single.GridRounds import compute_grid_rounds
from core.ex.plaintext.Server_main import send_to_server_np
from core.ciphertext_single.EncryptModule import DimensionalEncryptor

from core.ciphertext_single.minimax import (
    load_mcp,
    save_mcp,
)

# ─────────────────────────────────────────────────────────────────────────
# [삭제됨 2026-09] _MCP_ALPHA15_CHEB_PATH / _ensure_mcp_files()
# ─────────────────────────────────────────────────────────────────────────
#   "α=15 Chebyshev MCP 생성 중 (Normalize 전용)" 로그를 찍으며
#   mcp_alpha15_lp_cheb.json 을 만들던 STEP 0 블록. 2026-05b "옵션 B(전부 α=15
#   통일)" 시절의 잔재이고, 현재는 어디서도 쓰이지 않는다:
#
#     · Normalize 는 런타임에 Server_main.py:169 → check_neighbor_closed_interval(
#       num_points=N) → Normalize._ensure_norm_mcp(N) 경로로
#       **mcp_alpha{_norm_alpha(N)}_norm_cheb.json** 을 연다. 파일 이름 자체가 다르다
#       (_norm_cheb vs _lp_cheb). α 는 clamp(⌈log₂N⌉+8, 15, 20) 로 N 에 따라 변한다.
#         hepta(212)→16  tetra/lsun(400)→17  twodiamonds(800)/longsquare(900)→18
#         s1(5000)→20(clamp)
#     · mcp_alpha15_lp_cheb.json 은 _CORE_ALPHA_AUTO 또는 _LP_ALPHA_AUTO 가 False 일
#       때만 쓰이는 공유 파일인데 둘 다 True 이므로 참조되지 않는다.
#
#   → Remez 계산 시간만 쓰고 버려지는 죽은 산출물이었고, 로그가 "Normalize 가 고정
#     α=15 를 쓴다"는 오해를 유발했다. 삭제한다.
#   (Core/LP 용 α 별 MCP 는 아래 _ensure_mcp_files_for_N(N) 이 N 확정 후 생성하고,
#    Normalize 용은 런타임에 _ensure_norm_mcp 가 자동 생성한다.)

DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/s1.arff"


def _ensure_mcp_files_for_N(N: int):
    """[2026-07] N이 정해진 뒤 Core/LP가 쓸 alpha별 MCP를 미리 생성.

    미리 만들지 않아도 파이프라인 실행 중 자동 생성되지만, 그러면 Remez 계산
    시간과 로그가 FHE 연산 로그에 섞여 읽기 어렵다.

    alpha 선택 근거 (둘 다 N에만 의존 -> 공개값, 추가 유출 없음):
      LP  : 라벨이 정수 -> 최소 gap = 1/(1.1N)  -> alpha = ceil(log2(1.1N)) + 1
      Core: neighbor_count가 정수 -> gap = 0.5/N -> alpha = ceil(log2(2N)) + 2
    Normalize 는 여기서 만들지 않는다 — 런타임에 check_neighbor_closed_interval 이
      Normalize._ensure_norm_mcp(N) 로 mcp_alpha{α}_norm_cheb.json 을 자동 생성한다.
      α = clamp(⌈log₂N⌉ + 8, 15, 20).
      [2026-09 정정] 이 docstring 은 예전에 "Normalize 만 alpha=15 를 유지한다"고
      서술했으나 이는 2026-05b 시절 얘기이고, Normalize 도 이미 N 기반 가변 α 다.
      Normalize α 를 닫힌 식으로 유도하지 못하는 이유(=dist_sq 가 연속값이라 최소 gap
      의 공개 하한이 없다)는 여전히 유효하며, 그래서 실측 캘리브레이션 상수 C=8 을
      쓴다. 다만 α 는 데이터마다 달라진다.
    """
    from core.ciphertext_single.Core import _core_alpha, _CORE_ALPHA_AUTO
    from core.ciphertext_single.Label_Propagation import (
        _lp_alpha, _LP_ALPHA_AUTO, _LP_USE_CLEANING_DEGREES, _SGN_CLEANING_ITERS,
    )
    from core.ciphertext_single.minimax import (
        compute_mcp_for_label_prop_chebyshev,
        compute_mcp_for_label_prop_cleaning, save_mcp,
    )

    print(f"\n[MCP] N={N} 기반 alpha별 MCP 준비")

    if _CORE_ALPHA_AUTO:
        a = _core_alpha(N)
        path = f"mcp_alpha{a}_lp_cheb.json"
        if os.path.exists(path):
            print(f"      Core: alpha={a} -> {path} 존재, 스킵")
        else:
            print(f"      Core: alpha={a} (gap=0.5/{N}={0.5/N:.6f}) 생성 -> {path}")
            save_mcp(compute_mcp_for_label_prop_chebyshev(alpha=a, verbose=True), path)
    else:
        print("      Core: alpha 자동선택 OFF -> alpha=15 공유 파일 사용")

    if _LP_ALPHA_AUTO:
        a = _lp_alpha(N)
        path = (f"mcp_alpha{a}_lp_clean{_SGN_CLEANING_ITERS}.json"
                if _LP_USE_CLEANING_DEGREES else f"mcp_alpha{a}_lp_cheb.json")
        if os.path.exists(path):
            print(f"      LP:   alpha={a} -> {path} 존재, 스킵")
        else:
            print(f"      LP:   alpha={a} (gap=1/(1.1*{N})={1.0/(1.1*N):.6f}) 생성 -> {path}")
            if _LP_USE_CLEANING_DEGREES:
                comps = compute_mcp_for_label_prop_cleaning(
                    alpha=a, cleaning_iters=_SGN_CLEANING_ITERS, verbose=True)
            else:
                comps = compute_mcp_for_label_prop_chebyshev(alpha=a, verbose=True)
            save_mcp(comps, path)
    else:
        print("      LP:   alpha 자동선택 OFF -> alpha=15 고정 파일 사용")


def _gpu_used_mb() -> float:
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0

def _gpu_total_mb() -> float:
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 2)
    except Exception:
        return 0.0

def _print_mem(label: str):
    used = _gpu_used_mb(); total = _gpu_total_mb()
    print(f"  [MEM] {label:<45}  used={used:.0f} MB  free={total-used:.0f} MB")

def _mem_delta(label: str, before: float) -> float:
    after = _gpu_used_mb()
    print(f"  [MEM] {label:<45}  delta={after-before:+.0f} MB  (used={after:.0f} MB)")
    return after


# ── 데이터 로드 ──────────────────────────────────────────────────────────

def load_arff_to_pts_with_labels(filepath: str):
    pts, true_labels = [], []
    data_section = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'): continue
            if line.lower().startswith('@data'):
                data_section = True; continue
            if data_section:
                line = line.replace('\t', ' ').replace(',', ' ')
                values = line.split()
                if len(values) < 2: continue
                pts.append([float(v) for v in values[:-1]])
                true_labels.append(int(float(values[-1])))
    if not pts:
        raise ValueError("데이터를 찾을 수 없습니다.")
    return np.array(pts, dtype=np.float64), np.array(true_labels, dtype=int)


# ── 결과 저장 유틸 ────────────────────────────────────────────────────────

def save_timings_txt(filename, timings_dict):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for k, v in timings_dict.items():
                f.write(f"{k}: {float(v):.6f}\n")
        print(f"✅ 타이밍 저장: {filename}")
    except Exception as e:
        print(f"❌ 타이밍 저장 실패: {e}")


def save_heap_order_csv(filename, heap_idx, inv_perm, N):
    """Heap 정렬 인덱스 디버그 저장."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Heap_Position,Original_Index,Inv_Perm\n")
            for i in range(N):
                f.write(f"{i},{heap_idx[i]},{inv_perm[i]}\n")
        print(f"✅ Heap 순서 저장: {filename}")
    except Exception as e:
        print(f"❌ Heap 순서 저장 실패: {e}")


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FHE DBSCAN E2E 테스트 (Heap KD-tree depth 전파)")
    print("=" * 60)

    # ── Step 1: 파라미터 입력 ────────────────────────────────────
    #   ([2026-09] 기존 STEP 0 의 α=15 MCP 사전생성은 죽은 코드라 삭제.
    #    Core/LP 용은 N 확정 후 _ensure_mcp_files_for_N(N), Normalize 용은
    #    런타임에 Normalize._ensure_norm_mcp(N) 이 자동 생성한다.)
    print(f"\n▶ 데이터셋: {DATASET_PATH}")
    eps_val     = float(input("eps 값 > "))
    min_pts_val = int(input("min_pts 값 > "))
    print(f"\n▶ eps={eps_val}, min_pts={min_pts_val}\n")

    pts, true_labels = load_arff_to_pts_with_labels(DATASET_PATH)
    N         = len(pts)
    dimension = pts.shape[1]


    # ── Step 2: 정규화 ───────────────────────────────────────────
    # ★ [2026-09] 스케일 = max(좌표범위, PC1 span).  Client_main.send_to_server_fhe 와 동일.
    #   PC1 span 을 포함해야 n_rounds 의 기하항 ⌈1/ε_n⌉ 이 PC1 extent 를 항상 덮는다
    #   (lsun/longsquare/target/s1 은 PC1 span > 좌표범위).
    #   max() 이므로 데이터는 [0,1]^d 안에 유지되어 Normalize 의 max_dist_sq=dim 도 보존.
    global_min   = np.min(pts)
    global_max   = np.max(pts)
    _coord_range = (global_max - global_min) if (global_max - global_min) != 0.0 else 1.0
    _Xc          = pts - pts.mean(axis=0)
    _pc1         = _Xc @ np.linalg.svd(_Xc, full_matrices=False)[2][0]
    _pc1_span    = float(_pc1.max() - _pc1.min())
    scale_factor = max(_coord_range, _pc1_span) or 1.0
    normalized_pts = (pts - global_min) / scale_factor
    normalized_eps = eps_val / scale_factor

    # ★ [2026-10] n_rounds: 클라이언트 grid(셀 ε_n/2) 산출 → 서버로 명시 전달
    #   n = max(1, max_g ⌈(Λ_g + s√d)/ε_n⌉).  점 사이 거리 비교 없음, 상한 ⌈1/ε_n⌉ 없음.
    #   pc1 은 첫 Dijkstra 출발 셀 선택에만 쓰임 (부호·결과 무관). build_pca_order 와 같은 부호 규칙 적용.
    _ax_g = np.linalg.svd(_Xc, full_matrices=False)[2][0]
    if _ax_g[np.argmax(np.abs(_ax_g))] > 0:
        _ax_g = -_ax_g
    n_rounds_grid, grid_info = compute_grid_rounds(
        normalized_pts, normalized_eps, int(min_pts_val), pc1=(_Xc @ _ax_g) / scale_factor)

    # [2026-07] N이 정해졌으니 Core/LP용 alpha별 MCP를 여기서 미리 생성

    _ensure_mcp_files_for_N(N)


    print(f"데이터: {N}개, {dimension}차원  정규화 eps={normalized_eps:.6f}")
    if N > 100:
        print(f"⚠️  {N}개 → 긴 시간 소요 예상\n")

    # ── Step 3+4: Ball Tree 정렬 + k_max 구조 분석 (eps-이웃 조회 없음) ──
    print("\n[Ball Tree] BFS level-order 정렬 + k_max 구조 분석...")
    k_max, heap_idx, inv_perm = prepare_client_ordering(
        normalized_pts, normalized_eps, N
    )

    ball_sorted_pts = normalized_pts[heap_idx]
    save_heap_order_csv(f"debug_heap_order_N{N}.csv", heap_idx, inv_perm, N)
    T_kmax = k_max * (k_max + 1) // 2
    print(f"  → k_max={k_max}  T({k_max})={T_kmax}  "
        f"{'✓ ≥' if T_kmax >= N else '⚠ <'} N={N}")


    # ── Step 5: 평문 DBSCAN (원래 순서 기준, 비교 기준) ──────────
    print("\n================ Plaintext (sklearn DBSCAN) ===================")
    pt_start = time()
    _db = DBSCAN(eps=normalized_eps, min_samples=int(min_pts_val)).fit(normalized_pts)
    cluster_labels_np = _db.labels_.tolist()          # -1=노이즈, 0,1,…=클러스터

    # debug_np 재구성 (Step 8 디버그 CSV 호환)
    _nn    = NearestNeighbors(radius=normalized_eps).fit(normalized_pts)
    _cnt   = np.array([len(x) for x in _nn.radius_neighbors(normalized_pts,
                                                            return_distance=False)], dtype=float)
    debug_np = {
        'total_neighbors': _cnt,                                  # 자기 포함
        'core_mask'      : (_cnt >= min_pts_val).astype(float),
        'final_labels'   : np.array([l + 1 if l >= 0 else 0 for l in _db.labels_], dtype=float),
    }
    print(f"[평문/sklearn] 클러스터 {len(set(cluster_labels_np) - {-1})}개, "
        f"노이즈 {cluster_labels_np.count(-1)}개, core {int(debug_np['core_mask'].sum())}개")
    print(f"▶ Plaintext: {time()-pt_start:.2f}초\n")

    # ── Step 6: FHE DBSCAN (Heap 정렬 데이터) ────────────────────
    print("================ FHE (kd_dense) ========================")
    _print_mem("Engine 초기화 전")

    engine, secret_key, keypack = setup_fhe_engine(verbose=True)
    slot_count = engine.slot_count
    print(f"  slot_count={slot_count:,}  N={N}  dim={dimension}")

    fhe_start = time()

    # Heap 순서로 정렬된 데이터 암호화
    encryptor = DimensionalEncryptor(engine, keypack)
    before    = _gpu_used_mb()
    encrypted_columns = encryptor.encrypt_data(ball_sorted_pts.tolist(), dimension)
    _mem_delta(f"encrypt_data Ball-sorted ({dimension}개)", before)

    # 서버 연산 (Phase 1: Normalize+Core+enc_sum_k, Phase 2: Label Prop)
    # 서버가 enc_sum_k 계산 후 k_max 자동 정밀화
    before = _gpu_used_mb()
    # ★ [2026-10] sweep 삭제 → use_kd_propagation / num_sweeps 인자 없음 (항상 kd_dense)
    # ★ [2026-10] n_rounds 는 Step 2 에서 grid 로 산출한 값을 명시 전달
    fhe_final_ct, debug_fhe = send_to_server_fhe(
        engine=engine, keypack=keypack, debug_secret_key=secret_key,
        encrypted_columns=encrypted_columns,
        num_points=N, eps=normalized_eps, min_pts=float(min_pts_val),
        k_max=k_max,                                        # PCA(PC1) ε-window 상한
        n_rounds=n_rounds_grid,                             # ★ [2026-10] grid 산출값
    )
    
    print(f"\n[k_max] {k_max}  (PCA(PC1) ε-window 상한, 클라이언트 산출)")

    # 복호화 (Heap 순서)
    before = _gpu_used_mb()
    decrypted_heap = np.real(engine.decrypt(fhe_final_ct, secret_key))
    _mem_delta("decrypt", before)

    # ── inv_perm: Heap 순서 → 원래 순서 복원 ─────────────────────
    # heap_labels[i]: Heap 위치 i = 원래 heap_idx[i]번 점의 라벨
    # original_labels[j] = heap_labels[inv_perm[j]]
    heap_labels_raw    = decrypted_heap[:N]
    original_labels_raw = heap_labels_raw[inv_perm]

    # ★ [2026-07] 간격 기반 판정 (정수 반올림 대체)
    print("[FHE]", end=" ")
    cluster_labels_fhe = assign_clusters_by_gap(original_labels_raw, N)

    fhe_elapsed = time() - fhe_start
    print(f"▶ FHE: {fhe_elapsed:.2f}초\n")

    # ── Step 7: 소요 시간 출력 ───────────────────────────────────
    if "timings" in debug_fhe:
        print("================ 소요 시간 =====================")
        t = debug_fhe["timings"]
        print(f"Normalize         : {t.get('normalize_sec', 0):.2f}초")
        print(f"Core              : {t.get('core_sec', 0):.2f}초")
        print(f"Label_Propagation : {t.get('label_propagation_sec', 0):.2f}초")
        total = sum(t.get(k, 0) for k in
                    ['normalize_sec', 'core_sec', 'label_propagation_sec'])
        print(f"합계              : {total:.2f}초\n")
        save_timings_txt(f"debug_timings_eps{eps_val}_min{min_pts_val}.txt", t)

    # ── Step 8: 상세 비교 CSV ────────────────────────────────────
    debug_filename = f"debug_fhe_vs_np_eps{eps_val}_min{min_pts_val}.csv"
    try:
        with open(debug_filename, 'w', encoding='utf-8') as f:
            f.write("Point_ID,Heap_Position,"
                    "NP_Neighbors,FHE_Neighbors,Diff_Neighbors,"
                    "NP_Core,FHE_Core,Diff_Core,"
                    "NP_Label,FHE_Label_Heap,FHE_Label_Orig\n")
            for i in range(N):
                heap_pos  = inv_perm[i]   # 원래 i번 점의 Heap 위치
                n_np  = debug_np['total_neighbors'][i]
                c_np  = debug_np['core_mask'][i]
                l_np  = debug_np['final_labels'][i]
                # FHE 중간값은 Heap 순서 → inv_perm[i]로 대응
                n_fhe      = debug_fhe['total_neighbors'][heap_pos]
                c_fhe      = debug_fhe['core_mask'][heap_pos]
                l_fhe_heap = debug_fhe['final_labels'][heap_pos]
                l_fhe_orig = cluster_labels_fhe[i]
                f.write(f"{i},{heap_pos},"
                        f"{n_np:.4f},{n_fhe:.4f},{abs(n_np-n_fhe):.4f},"
                        f"{c_np:.4f},{c_fhe:.4f},{abs(c_np-c_fhe):.4f},"
                        f"{l_np:.4f},{l_fhe_heap:.4f},{l_fhe_orig}\n")
        print(f"✅ 디버깅 파일: {debug_filename}")
    except Exception as e:
        print(f"❌ 디버깅 파일 실패: {e}")

    # ── Step 9: 검증 (ARI) ───────────────────────────────────────
    print("\n================ 검증 결과 ==============================")
    pt_valid  = [c for c in set(cluster_labels_np)  if c != -1]
    fhe_valid = [c for c in set(cluster_labels_fhe) if c != -1]
    print(f"[True]      클러스터 {len(set(true_labels))}개")
    print(f"[Plaintext] 클러스터 {len(pt_valid)}개")
    print(f"[FHE]       클러스터 {len(fhe_valid)}개")

    # 실제 정답 기반 ARI (절대적 정확도 — 핵심 지표)
    ari_true_fhe = adjusted_rand_score(true_labels.tolist(), cluster_labels_fhe)
    ari_true_np  = adjusted_rand_score(true_labels.tolist(), cluster_labels_np)
    # 평문 vs FHE 상대 비교 (FHE 구현 충실도)
    ari_np_fhe   = adjusted_rand_score(cluster_labels_np, cluster_labels_fhe)

    print(f"\n📊 [ARI] 실제 정답 vs FHE:  {ari_true_fhe*100:.2f}점 / 100점  ← 핵심 지표")
    print(f"📊 [ARI] 실제 정답 vs 평문: {ari_true_np*100:.2f}점 / 100점")
    print(f"📊 [ARI] 평문 vs FHE:       {ari_np_fhe*100:.2f}점 / 100점")

    if   ari_true_fhe > 0.99: print("  => 🎉 대성공!")
    elif ari_true_fhe > 0.80: print("  => 👍 대부분 일치")
    else:                     print("  => ❌ 군집 구조 붕괴")

    # ── Step 10: 최종 결과 저장 ──────────────────────────────────
    output_filename = f"longsquare_fhe_heap_result_eps{eps_val}_min{min_pts_val}.csv"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            axis_headers = [f"x{i+1}" for i in range(dimension)]
            f.write(",".join(axis_headers) +
                    ",True_Class,PT_Cluster,FHE_Cluster,Heap_Position\n")
            for i in range(N):
                coords = ",".join([f"{val:.4f}" for val in pts[i]])
                f.write(f"{coords},{true_labels[i]},"
                        f"{cluster_labels_np[i]},{cluster_labels_fhe[i]},"
                        f"{inv_perm[i]}\n")
        print(f"✅ 결과 저장: {output_filename}")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")


if __name__ == '__main__':
    main()
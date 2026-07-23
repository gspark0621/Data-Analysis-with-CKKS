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
# [변경 4] min_pts 분기 기준: 고정값 4 → 2×dim (Sander et al. 1998) ← 신규
#   기존: min_pts >= 4 → kd_dense
#   변경: min_pts >= 2×dim → kd_dense
#   근거: d차원 공간에서 밀집 구조 신뢰를 위한 최소 이웃 수 = 2d
# ─────────────────────────────────────────────────────────────────────────
#
# [변경 5, 2026-05b] 옵션 B: 전부 α=15 Chebyshev로 통일
#   - mcp_alpha11.json / mcp_normalize_alpha12.json / mcp_alpha15_lp.json (power basis)
#     3개 stale 파일 자동 생성 제거. 코드 어디서도 참조 안 됨.
#   - 단일 파일 mcp_alpha15_lp_cheb.json 만 자동 생성 (Normalize/Core/LP 공유).
#   - 이유: Core α=12 worst case 안전성 미확인 → α=15에서 77τ 마진 확보.
#           Chebyshev basis는 high-deg coefficient 폭발 회피.
#
# [변경 6, 2026-05c] 정확도 버그 수정 통합
#   - 작업 A: Core/Normalize 마지막 일반 bootstrap → bit_cleaning (mask noise 제거)
#     + LP의 core_mask _refresh 직후 bit_cleaning 추가 (0.714 감쇠 버그 수정)
#   - 작업 B: LP에 n_rounds=⌈log₂N⌉ 적응적 round (test에서 명시 전달)
#   - 작업 C: verify_convergence.py (FHE 불필요 평문 수렴 검증, 별도 도구)
#   - sgn 정밀화: fhe_sgn에 sign_cleaning 추가 (fhe_max 라벨 단조하강 수정)
#   - sweep 폐기 → kd_dense 통일 (prepare_client_ordering 항상 'kd_dense' 반환)
# ────────────────────────────────────────────────────────────────────────────

import os
import math
import numpy as np
from time import time
import pynvml
from sklearn.metrics import adjusted_rand_score

from core.ciphertext_single.Client_main import (
    setup_fhe_engine,
    prepare_client_ordering,     # ★ Ball Tree + k_max 구조 분석 통합
    build_ball_tree_order,       # 개별 접근 (디버그용)
    compute_kmax_from_ball_structure,  # 개별 접근 (디버그용)
    assign_clusters_by_gap,      # ★ [2026-07] 간격 기반 클러스터 판정
    _N_ROUNDS,                   # ★ [2026-07] 고정 n_rounds
)
from core.ciphertext_single.Server_main import send_to_server_fhe
from core.ex.plaintext.Server_main import send_to_server_np
from core.ciphertext_single.EncryptModule import DimensionalEncryptor

from core.ciphertext_single.minimax import (
    compute_mcp_for_label_prop_chebyshev,   # ★ Normalize/Core/LP 공유 α=15 Chebyshev
    load_mcp,
    save_mcp,
)

# ─────────────────────────────────────────────────────────────────────────
# MCP 파일 경로 (옵션 B: α=15 Chebyshev 단일 파일 공유)
# ─────────────────────────────────────────────────────────────────────────
_MCP_ALPHA15_CHEB_PATH = "mcp_alpha15_lp_cheb.json"   # Normalize/Core/LP 공유

DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/3_spiral.arff"


# ── MCP 파일 준비 ─────────────────────────────────────────────────────────

def _ensure_mcp_files():
    """
    α=15 Chebyshev MCP (Normalize/Core/LP 공유) 자동 생성/확인.

    옵션 B (2026-05b) 정리 결정사항:
      - 이전: 3개 power basis MCP 자동 생성
              mcp_alpha11.json            (Core용 α=11, [7,15,15,15])
              mcp_normalize_alpha12.json  (Normalize용 α=12, [15,15,15,15])
              mcp_alpha15_lp.json         (LP용 α=15, [7,15,15,15,27])
      - 현재: Normalize/Core/LP 모두 mcp_alpha15_lp_cheb.json 단일 파일 공유.
              위 3개 power basis 파일은 더 이상 어디서도 참조되지 않으므로
              자동 생성 로직 제거. (디스크에 잔재 있어도 동작에 영향 없음.)

    α=15 Chebyshev 검증 결과 (sanity_check_alpha15.py):
      - Core worst case 77τ (= 0.5/N at N=212): PASS ✓ — 핵심 가정 검증.
      - 일반 영역 (≥16τ): PASS ✓
      - Normalize boundary (≈1τ): FAIL — 예상됨, false ±로 흡수.
      - ±3τ FAIL: 안전 영역이 실측 ~16τ. hepta ARI 실측으로 영향도 판단.
    """
    if not os.path.exists(_MCP_ALPHA15_CHEB_PATH):
        print(f"[MCP] α=15 Chebyshev MCP 생성 중 (Normalize/Core/LP 공유)")
        print(f"      degrees=[7,15,15,15,27], BSGS depth=5, margin η=2^{{-17}}")
        print(f"      δ=2^{{-15}}≈3.05e-5, drift(840콜)≈0.39 < 1.0 ✓")
        comps = compute_mcp_for_label_prop_chebyshev(alpha=15, verbose=True)
        save_mcp(comps, _MCP_ALPHA15_CHEB_PATH)
        print(f"[MCP] 저장 → {_MCP_ALPHA15_CHEB_PATH}  "
              f"err={comps[-1]['error']:.4e}  t_k={comps[-1]['t_i']:.4e}")
    else:
        print(f"[MCP] {_MCP_ALPHA15_CHEB_PATH} 존재 → 스킵 (Normalize/Core/LP 공유)")


# ── GPU 메모리 유틸 ───────────────────────────────────────────────────────

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

    # ── Step 0: MCP 파일 준비 ────────────────────────────────────
    print("\n[STEP 0] MCP 파일 준비")
    _ensure_mcp_files()

    # ── Step 1: 파라미터 입력 ────────────────────────────────────
    print(f"\n▶ 데이터셋: {DATASET_PATH}")
    eps_val     = float(input("eps 값 > "))
    min_pts_val = int(input("min_pts 값 > "))
    print(f"\n▶ eps={eps_val}, min_pts={min_pts_val}\n")

    pts, true_labels = load_arff_to_pts_with_labels(DATASET_PATH)
    N         = len(pts)
    dimension = pts.shape[1]


    # ── Step 2: 정규화 ───────────────────────────────────────────
    global_min   = np.min(pts)
    global_max   = np.max(pts)
    scale_factor = (global_max - global_min) if (global_max - global_min) != 0.0 else 1.0
    normalized_pts = (pts - global_min) / scale_factor
    normalized_eps = eps_val / scale_factor

    print(f"데이터: {N}개, {dimension}차원  정규화 eps={normalized_eps:.6f}")
    if N > 100:
        print(f"⚠️  {N}개 → 긴 시간 소요 예상\n")

    # ── Step 3+4: Ball Tree 정렬 + k_max 구조 분석 (eps-이웃 조회 없음) ──
    print("\n[Ball Tree] BFS level-order 정렬 + k_max 구조 분석...")
    mode, k_max, heap_idx, inv_perm = prepare_client_ordering(
        normalized_pts, normalized_eps, int(min_pts_val), N, dimension  # ← dim 추가
    )

    if mode == 'kd_dense':
        ball_sorted_pts = normalized_pts[heap_idx]
        save_heap_order_csv(f"debug_heap_order_N{N}.csv", heap_idx, inv_perm, N)
        T_kmax = k_max * (k_max + 1) // 2
        print(f"  → k_max={k_max}  T({k_max})={T_kmax}  "
            f"{'✓ ≥' if T_kmax >= N else '⚠ <'} N={N}")
    else:
        ball_sorted_pts = normalized_pts
        print(f"  → sweep 방식 (num_sweeps={k_max})")


    # ── Step 5: 평문 DBSCAN (원래 순서 기준, 비교 기준) ──────────
    print("\n================ Plaintext ===================")
    pt_start   = time()
    transposed = list(zip(*normalized_pts.tolist()))
    columns_np = [np.array(v, dtype=np.float64) for v in transposed]
    np_final_labels, _, debug_np = send_to_server_np(
        encrypted_columns=columns_np, num_points=N,
        eps=normalized_eps, min_pts=float(min_pts_val), dimension=dimension
    )
    # ★ [2026-07] 간격 기반 판정 (정수 반올림 대체) — Client_main 과 동일 경로
    print("[평문]", end=" ")
    cluster_labels_np = assign_clusters_by_gap(np.asarray(np_final_labels[:N]), N)
    print(f"▶ Plaintext: {time()-pt_start:.2f}초\n")

    # ── Step 6: FHE DBSCAN (Heap 정렬 데이터) ────────────────────
    print(f"================ FHE ({mode}) ========================")
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
    # ★ [2026-05c] kd_dense 통일: prepare_client_ordering이 항상 'kd_dense' 반환
    #   → use_kd_propagation=True 고정. num_sweeps는 dead path지만 호환 위해 인자 유지.
    # ★ [2026-05c 작업 B] n_rounds 명시: log₂N (Server가 None시 동일 처리하나 명시성)
    fhe_final_ct, debug_fhe = send_to_server_fhe(
        engine=engine, keypack=keypack, secret_key=secret_key,
        encrypted_columns=encrypted_columns,
        num_points=N, eps=normalized_eps, min_pts=float(min_pts_val),
        k_max=k_max,                                        # Ball Tree 구조 분석 최종값
        use_kd_propagation=(mode == 'kd_dense'),
        num_sweeps=k_max if mode == 'sweep' else None,      # sweep dead path
        n_rounds=_N_ROUNDS,                                 # ★ [2026-07] Client_main._N_ROUNDS 사용
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
    output_filename = f"hepta_fhe_heap_result_eps{eps_val}_min{min_pts_val}.csv"
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
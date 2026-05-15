# test_Server_main_call_dataset.py
#
# ── 변경사항 ────────────────────────────────────────────────────────────────
# [변경 1] Heap KD-tree 정렬 + dense stride k=1..k_max (power-of-2 → dense 수정)
#   - build_kd_tree_order()로 Heap(BFS level-order) 정렬
#   - k_max = min(N//2, 3×ceil(√N))  → T(k_max)≥N → 1 sweep 수렴 보장
#   - 복호화 후 inv_perm으로 원래 순서 복원
#
# [변경 2] decide_propagation_mode: 'kd_depth' → 'kd_dense'
#   - min_pts ≥ 4 → kd_dense  (dense k=1..k_max, 비구형 cluster도 정확)
#   - min_pts < 4 → sweep
#
# [변경 3] send_to_server_fhe: log2_n → k_max 전달
#
# [변경 4] Normalize mcp_normalize_alpha12.json 사용
#   - alpha=12: margin 0.012→0.00077, false positive 32%→2.4%
# ────────────────────────────────────────────────────────────────────────────

import os
import math
import numpy as np
from time import time
import pynvml
from sklearn.metrics import adjusted_rand_score

from core.ciphertext_single.Client_main import (
    setup_fhe_engine,
    build_kd_tree_order,
    get_kd_dense_kmax,       # ★ get_depth_strides → get_kd_dense_kmax
    decide_propagation_mode,
)
from core.ciphertext_single.Server_main import send_to_server_fhe
from core.ex.plaintext.Server_main import send_to_server_np
from core.ciphertext_single.EncryptModule import DimensionalEncryptor

from core.ciphertext_single.minimax import (
    compute_mcp_for_normalize,
    compute_mcp_for_core,
    load_mcp,
    save_mcp,
)

_MCP_ALPHA11_PATH = "mcp_alpha11.json"
_MCP_NORMALIZE_ALPHA12_PATH = "mcp_normalize_alpha12.json"   # ★ Normalize 전용 alpha=12

DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/hepta.arff"


# ── MCP 파일 준비 ─────────────────────────────────────────────────────────

def _ensure_mcp_files():
    """
    α=11 (Core, LabelProp): degrees=[7,15,15,15]
    α=12 (Normalize 전용):   degrees=[15,15,15,15]  ← false positive 32%→2.4%
    """

    if not os.path.exists(_MCP_ALPHA11_PATH):
        print(f"[MCP] α=11 계산 중 (논문 Table 2: [7,15,15,15])...")
        comps11 = compute_mcp_for_core(alpha=11, verbose=True)
        save_mcp(comps11, _MCP_ALPHA11_PATH)
        print(f"[MCP] α=11 저장 → {_MCP_ALPHA11_PATH}  "
              f"err={comps11[-1]['error']:.4e}  t_k={comps11[-1]['t_i']:.4e}")
    else:
        print(f"[MCP] α=11 존재, 스킵 → {_MCP_ALPHA11_PATH}")

    # ★ Normalize 전용 alpha=12: margin 0.012→0.00077, false positive 32%→2.4%
    if not os.path.exists(_MCP_NORMALIZE_ALPHA12_PATH):
        print(f"[MCP] α=12 (Normalize 전용) 계산 중 (degrees=[15,15,15,15])...")
        comps12 = compute_mcp_for_normalize(alpha=12, verbose=True)
        save_mcp(comps12, _MCP_NORMALIZE_ALPHA12_PATH)
        print(f"[MCP] α=12 저장 → {_MCP_NORMALIZE_ALPHA12_PATH}  "
              f"err={comps12[-1]['error']:.4e}  t_k={comps12[-1]['t_i']:.4e}")
    else:
        print(f"[MCP] α=12 Normalize 존재, 스킵 → {_MCP_NORMALIZE_ALPHA12_PATH}")


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
    log2_n    = math.ceil(math.log2(N))

    # ── Step 2: 정규화 ───────────────────────────────────────────
    global_min   = np.min(pts)
    global_max   = np.max(pts)
    scale_factor = (global_max - global_min) if (global_max - global_min) != 0.0 else 1.0
    normalized_pts = (pts - global_min) / scale_factor
    normalized_eps = eps_val / scale_factor

    print(f"데이터: {N}개, {dimension}차원  정규화 eps={normalized_eps:.6f}")
    if N > 100:
        print(f"⚠️  {N}개 → 긴 시간 소요 예상\n")

    # ── Step 3: 전파 방식 결정 (O(1)) ────────────────────────────
    mode, k_max = decide_propagation_mode(min_pts_val, log2_n, N)

    # ── Step 4: Heap KD-tree 정렬 (kd_dense 선택 시) ─────────────
    if mode == 'kd_dense':
        print("\n[Heap KD-tree] BFS level-order 정렬 중...")
        heap_idx, inv_perm = build_kd_tree_order(normalized_pts)
        kd_sorted_pts      = normalized_pts[heap_idx]
        T_kmax = k_max * (k_max + 1) // 2

        old_ops = 2 * 2 * math.ceil(math.log2(N)) * (N // 2) * 2   # old sweep
        new_ops = 2 * 2 * k_max * 2                                  # dense fwd+bwd
        print(f"[Heap KD-tree] k_max={k_max}, T({k_max})={T_kmax}")
        print(f"[Heap KD-tree] fhe_max: {new_ops}회 (기존 sweep {old_ops}회, {old_ops//new_ops}배↓)")

        save_heap_order_csv(f"debug_heap_order_N{N}.csv", heap_idx, inv_perm, N)
    else:
        # sweep 방식: 정렬 없이 원래 순서
        heap_idx      = np.arange(N)
        inv_perm      = np.arange(N)
        kd_sorted_pts = normalized_pts
        print(f"[Sweep] 정렬 없음, num_sweeps={k_max}")

    # ── Step 5: 평문 DBSCAN (원래 순서 기준, 비교 기준) ──────────
    print("\n================ Plaintext ===================")
    pt_start   = time()
    transposed = list(zip(*normalized_pts.tolist()))
    columns_np = [np.array(v, dtype=np.float64) for v in transposed]
    np_final_labels, _, debug_np = send_to_server_np(
        encrypted_columns=columns_np, num_points=N,
        eps=normalized_eps, min_pts=float(min_pts_val), dimension=dimension
    )
    cluster_labels_np = []
    for x in np_final_labels[:N]:
        r = round(x)
        if r <= 0:  cluster_labels_np.append(-1)
        elif r > N: cluster_labels_np.append(N)
        else:       cluster_labels_np.append(r)
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
    encrypted_columns = encryptor.encrypt_data(kd_sorted_pts.tolist(), dimension)
    _mem_delta(f"encrypt_data Heap-sorted ({dimension}개)", before)

    # 서버 연산
    before = _gpu_used_mb()
    fhe_final_ct, debug_fhe = send_to_server_fhe(
        engine=engine, keypack=keypack, secret_key=secret_key,
        encrypted_columns=encrypted_columns,
        num_points=N, eps=normalized_eps, min_pts=float(min_pts_val),
        k_max=k_max,
        use_kd_propagation=(mode == 'kd_dense'),
        num_sweeps=k_max,
    )
    _mem_delta("send_to_server_fhe 전체", before)

    # 복호화 (Heap 순서)
    before = _gpu_used_mb()
    decrypted_heap = np.real(engine.decrypt(fhe_final_ct, secret_key))
    _mem_delta("decrypt", before)

    # ── inv_perm: Heap 순서 → 원래 순서 복원 ─────────────────────
    # heap_labels[i]: Heap 위치 i = 원래 heap_idx[i]번 점의 라벨
    # original_labels[j] = heap_labels[inv_perm[j]]
    heap_labels_raw    = decrypted_heap[:N]
    original_labels_raw = heap_labels_raw[inv_perm]

    cluster_labels_fhe = []
    for x in original_labels_raw:
        r = int(np.round(x))
        if r <= 0:  cluster_labels_fhe.append(-1)
        elif r > N: cluster_labels_fhe.append(N)
        else:       cluster_labels_fhe.append(r)

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
    print(f"[Plaintext] 클러스터 {len(pt_valid)}개")
    print(f"[FHE]       클러스터 {len(fhe_valid)}개")

    ari = adjusted_rand_score(cluster_labels_np, cluster_labels_fhe)
    print(f"\n📊 [ARI] 평문 vs FHE: {ari*100:.2f}점 / 100점")
    if   ari > 0.99: print("  => 🎉 대성공!")
    elif ari > 0.80: print("  => 👍 대부분 일치")
    else:            print("  => ❌ 군집 구조 붕괴")

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
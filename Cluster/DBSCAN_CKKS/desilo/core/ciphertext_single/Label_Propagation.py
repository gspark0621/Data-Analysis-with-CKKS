# core/ciphertext_single/Label_Propagation.py
#
# ── 변경사항 ────────────────────────────────────────────────────────────────
# [변경 1] fhe_kd_depth_propagation() 추가
#   Heap(BFS level-order) 레이아웃 전제:
#     depth d cross-boundary stride = 2^(d-1)
#     → depth_strides = [1, 2, 4, 8, ..., min(2^(log₂N-1), N//2)]
#   2-pass (bottom-up + top-down):
#     Pass1: strides 오름차순 [1,2,4,...] → 로컬 연결 먼저 해결
#     Pass2: strides 내림차순 [...,4,2,1] → transitive update 보완
#   연산량: 2ph × 2pass × log₂N = 4×log₂N fhe_max (N=212: ~64회)
#   기존 fhe_sweep_propagation: 2ph × num_sweeps × N//2 (N=212: ~1696회+)
#
# [변경 2] _propagate_one_stride_core / _propagate_one_stride_border 헬퍼 추가
#   코드 중복 제거, 두 전파 함수가 공유
#
# [유지] fhe_sweep_propagation (fallback, min_pts<4 or 하위호환)
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
import numpy as np
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp

_MCP_LABEL_PATH     = "mcp_alpha11.json"
_mcp_label_components = None

def _get_mcp_label():
    global _mcp_label_components
    if _mcp_label_components is None:
        print(f"  [LabelProp] MCP 로드: {_MCP_LABEL_PATH}")
        _mcp_label_components = load_mcp(_MCP_LABEL_PATH)
    return _mcp_label_components


def _dbg(engine, secret_key, ct, tag, num_points, show=6):
    if secret_key is None:
        return ct
    vals = np.array(engine.decrypt(ct, secret_key))[:num_points]
    print(f"  [DBG] {tag}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")
    print(f"         {np.round(vals[:show], 4).tolist()}")
    return ct


def _refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack) -> Ciphertext:
    """일반 bootstrap."""
    return engine.bootstrap(
        engine.intt(ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key,
    )


def _eval_one_component(engine, ct, comp, slot_count, keypack, relin_key):
    """단일 MCP 컴포넌트 평가. domain_b 정규화 포함."""
    coeffs   = comp["coeffs"]
    domain_b = comp.get("domain_b", 1.0)

    if abs(domain_b - 1.0) > 1e-9:
        x_sc = engine.multiply(ct, engine.encode([1.0 / domain_b] * slot_count))
    else:
        x_sc = ct

    x_sq  = engine.square(x_sc, relin_key)
    x_pow = x_sc
    result = engine.multiply(x_pow, engine.encode([coeffs[0]] * slot_count))

    for k in range(1, len(coeffs)):
        x_pow  = engine.multiply(x_pow, x_sq, relin_key)
        result = engine.add(result,
                            engine.multiply(x_pow, engine.encode([coeffs[k]] * slot_count)))
    return result


def _eval_mcp(engine, ct, components, slot_count, keypack):
    """MCP 전체 합성. 모든 컴포넌트 후 bootstrap → level=10 반환."""
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key
    current   = ct

    for step_idx, comp in enumerate(components):
        current = _eval_one_component(engine, current, comp, slot_count, keypack, relin_key)
        current = engine.intt(current)
        current = engine.bootstrap(current, relin_key, conj_key, boot_key)

    return current  # level=10


def fhe_sgn(
    engine: Engine, x_ct: Ciphertext, num_points: int, keypack: KeyPack,
    secret_key=None, tag: str = "",
) -> Ciphertext:
    """MCP + sign_bootstrap으로 sgn(x) 근사."""
    slot_count = engine.slot_count
    components = _get_mcp_label()

    result = _eval_mcp(engine, x_ct, components, slot_count, keypack)
    result = engine.sign_bootstrap(
        engine.intt(result),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )
    return result


def fhe_max(
    engine: Engine, u_ct: Ciphertext, v_ct: Ciphertext,
    num_points: int, keypack: KeyPack,
    label_scale: float = 1.0,
    secret_key=None, tag: str = "",
) -> Ciphertext:
    """max(u,v) = ((u+v) + (u-v)·sgn((u-v)/label_scale)) / 2"""
    slot_count = engine.slot_count
    relin_key  = keypack.relinearization_key

    u_minus_v = engine.subtract(u_ct, v_ct)
    u_minus_v = _refresh(engine, u_minus_v, keypack)

    if abs(label_scale - 1.0) > 1e-9:
        u_minus_v_for_sgn = engine.multiply(
            u_minus_v, engine.encode([1.0 / label_scale] * slot_count))
    else:
        u_minus_v_for_sgn = u_minus_v

    sgn_ct   = fhe_sgn(engine, u_minus_v_for_sgn, num_points, keypack,
                       secret_key=secret_key, tag=f"{tag}max/")
    diff_sgn = engine.multiply(u_minus_v, sgn_ct, relin_key)
    combined = engine.add(engine.add(u_ct, v_ct), diff_sgn)
    result   = engine.multiply(combined, engine.encode([0.5] * slot_count))
    return _refresh(engine, result, keypack)


def fhe_circular_shift(engine, ct, k, num_points, keypack):
    """원형 LEFT-shift: result[i] = ct[(i+k) mod num_points]"""
    slot_count    = engine.slot_count
    left_shifted  = engine.rotate(ct, keypack.rotation_key, -k)
    right_shifted = engine.rotate(ct, keypack.rotation_key, (num_points - k))

    mask_left  = [1.0 if i < (num_points - k) else 0.0 for i in range(slot_count)]
    mask_right = [0.0] * slot_count
    for i in range(num_points - k, num_points):
        mask_right[i] = 1.0

    return engine.add(
        engine.multiply(left_shifted,  engine.encode(mask_left)),
        engine.multiply(right_shifted, engine.encode(mask_right)),
    )


# ── 헬퍼: stride 하나 전파 (Core-Core) ────────────────────────────────────

def _propagate_one_stride_core(
    engine, keypack, adj_k_half_list,
    core_labels_ct, core_mask_ct,
    k, N, label_scale,
    secret_key=None,
):
    """
    Core-Core 전파: stride k에 대해 Forward + Backward 처리.
    adj_{N-k} = rotate(adj_k, N-k) 대칭 활용.
    """
    relin_key = keypack.relinearization_key
    adj_k     = adj_k_half_list[k - 1]

    # Forward: label[i] ← max(label[i], adj_k[i] × label[(i+k)%N])
    s_mask_k  = fhe_circular_shift(engine, core_mask_ct,   k, N, keypack)
    s_label_k = fhe_circular_shift(engine, core_labels_ct, k, N, keypack)
    neighbor_k = _refresh(engine, engine.multiply(
        engine.multiply(adj_k, s_mask_k,  relin_key),
        s_label_k, relin_key), keypack)
    core_labels_ct = fhe_max(engine, core_labels_ct, neighbor_k,
                             N, keypack, label_scale=label_scale,
                             secret_key=secret_key)
    core_labels_ct = _refresh(engine,
        engine.multiply(core_labels_ct, core_mask_ct, relin_key), keypack)

    # Backward: label[i] ← max(label[i], adj_{N-k}[i] × label[(i-k)%N])
    if 2 * k < N:
        adj_Nk     = fhe_circular_shift(engine, adj_k,         N-k, N, keypack)
        s_mask_Nk  = fhe_circular_shift(engine, core_mask_ct,  N-k, N, keypack)
        s_label_Nk = fhe_circular_shift(engine, core_labels_ct,N-k, N, keypack)
        neighbor_Nk = _refresh(engine, engine.multiply(
            engine.multiply(adj_Nk, s_mask_Nk,  relin_key),
            s_label_Nk, relin_key), keypack)
        core_labels_ct = fhe_max(engine, core_labels_ct, neighbor_Nk,
                                 N, keypack, label_scale=label_scale,
                                 secret_key=secret_key)
        core_labels_ct = _refresh(engine,
            engine.multiply(core_labels_ct, core_mask_ct, relin_key), keypack)

    return core_labels_ct


# ── 헬퍼: stride 하나 전파 (Core→Border) ─────────────────────────────────

def _propagate_one_stride_border(
    engine, keypack, adj_k_half_list,
    border_labels_ct, non_core_ct, core_labels_ct, core_mask_ct,
    k, N, label_scale,
    secret_key=None,
):
    """Core→Border 전파: stride k에 대해 Forward + Backward 처리."""
    relin_key = keypack.relinearization_key
    adj_k     = adj_k_half_list[k - 1]

    # Forward
    s_mask_k  = fhe_circular_shift(engine, core_mask_ct,   k, N, keypack)
    s_label_k = fhe_circular_shift(engine, core_labels_ct, k, N, keypack)
    candidate_k = _refresh(engine, engine.multiply(
        engine.multiply(adj_k, s_mask_k,  relin_key),
        s_label_k, relin_key), keypack)
    border_labels_ct = fhe_max(engine, border_labels_ct, candidate_k,
                               N, keypack, label_scale=label_scale,
                               secret_key=secret_key)
    border_labels_ct = _refresh(engine,
        engine.multiply(border_labels_ct, non_core_ct, relin_key), keypack)

    # Backward
    if 2 * k < N:
        adj_Nk      = fhe_circular_shift(engine, adj_k,         N-k, N, keypack)
        s_mask_Nk   = fhe_circular_shift(engine, core_mask_ct,  N-k, N, keypack)
        s_label_Nk  = fhe_circular_shift(engine, core_labels_ct,N-k, N, keypack)
        candidate_Nk = _refresh(engine, engine.multiply(
            engine.multiply(adj_Nk, s_mask_Nk,  relin_key),
            s_label_Nk, relin_key), keypack)
        border_labels_ct = fhe_max(engine, border_labels_ct, candidate_Nk,
                                   N, keypack, label_scale=label_scale,
                                   secret_key=secret_key)
        border_labels_ct = _refresh(engine,
            engine.multiply(border_labels_ct, non_core_ct, relin_key), keypack)

    return border_labels_ct


# ══════════════════════════════════════════════════════════════════════════
# 신규: KD-tree Heap depth 기반 2-pass 전파
# ══════════════════════════════════════════════════════════════════════════

def fhe_kd_dense_propagation(
    engine: Engine,
    keypack: KeyPack,
    adj_k_half_list: list,
    core_ct: Ciphertext,
    num_points: int,
    k_max: int,
    secret_key=None,
) -> Ciphertext:
    """
    KD-tree ordering + dense stride k=1..k_max 라벨 전파.

    ── 구 fhe_kd_depth_propagation 문제 ────────────────────────────────────
      depth_strides = [1,2,4,8,...,64] (power-of-2만)
      → stride=3,5,7,... 연결 영구 누락
      → 비구형 클러스터: 멤버가 stride=3,5,...에 위치 → 전파 실패

    ── 수정: dense k=1..k_max  (FHE sequential T(k_max) 정리) ──────────────
      FHE sweep에서 k=1→2→...→k_max 순서로 처리,
      각 k 처리 시 직전 k에서 업데이트된 label 사용 (체이닝):

        After k=1: label[i] = max(label[i], label[i+1])
        After k=2: label[i] = max(label[i], UPDATED label[i+2])
                             = max(label[i..i+3])  ← 4개 커버
        After k=j: T(j) = j(j+1)/2 위치까지 전파

      k_max = min(N//2, 3×ceil(√N)):  N=212 → k_max=45, T(45)=1035 >> N
      → 연속 stride ≤ k_max인 모든 클러스터를 1 forward sweep으로 수렴 ✓
      → power-of-2가 아닌 stride=3,5,7,... 연결도 모두 커버 ✓

    ── 연산량 (N=212, k_max=45) ─────────────────────────────────────────────
      fhe_max: 2phases × 2passes × 45strides × 2방향 = 360회
      (구 depth_strides 64회 대비 5.6배, ALL-sweep 3392회 대비 9.4배↓)

    label_scale = N: 기존 Server_main 호환 (정수 라벨 1..N 반환)
    """
    N = num_points
    k_max = min(k_max, N // 2)
    T_kmax = k_max * (k_max + 1) // 2

    label_scale = float(N)
    slot_count  = engine.slot_count

    print(f"\n[KD-LP] ══════════════════════════════════════════")
    print(f"[KD-LP] KD-tree dense stride 전파 (k=1..{k_max})")
    print(f"[KD-LP] N={N}, k_max={k_max}")
    print(f"[KD-LP] T({k_max})={T_kmax}  {'✓ ≥ N' if T_kmax >= N else '⚠ < N, cluster span 초과 시 추가 pass 필요'}")
    fhe_max_cnt = 2 * 2 * k_max * 2
    print(f"[KD-LP] fhe_max: {fhe_max_cnt}회 "
          f"(depth-stride 64회 대비 {fhe_max_cnt/64:.1f}배, ALL-sweep 3392회 대비 {3392/fhe_max_cnt:.1f}배↓)")
    print(f"[KD-LP] ══════════════════════════════════════════\n")

    # ── 초기화 ─────────────────────────────────────────────────
    core_mask_ct = _refresh(engine, core_ct, keypack)
    non_core_ct  = _refresh(engine,
        engine.subtract(engine.encode([1.0] * slot_count), core_mask_ct), keypack)

    id_enc = engine.encode(
        [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
    core_labels_ct = _refresh(engine,
        engine.multiply(core_mask_ct, id_enc), keypack)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N)

    # ── Phase 1: Core-Core (forward sweep + backward sweep) ────
    # forward (k=1..k_max): sequential T(k_max) 체이닝으로 우→좌 전파
    # backward (k=k_max..1): 좌→우 전파 보완
    passes = [
        ("forward",  list(range(1, k_max + 1))),
        ("backward", list(range(k_max, 0, -1))),
    ]
    for pass_idx, (pass_name, k_order) in enumerate(passes):
        print(f"[KD-LP] Phase1: Core-Core Pass{pass_idx+1}/2 ({pass_name}): k=1..{k_max}")
        for k in k_order:
            core_labels_ct = _propagate_one_stride_core(
                engine, keypack, adj_k_half_list,
                core_labels_ct, core_mask_ct,
                k, N, label_scale, secret_key=secret_key,
            )
        _dbg(engine, secret_key, core_labels_ct, f"P1-{pass_name} 완료", N)

    _dbg(engine, secret_key, core_labels_ct, "Phase1 완료 (Core-Core)", N)

    # ── Phase 2: Core→Border (forward + backward) ──────────────
    border_labels_ct = _refresh(engine,
        engine.multiply(engine.encode([0.0] * slot_count), non_core_ct), keypack)

    for pass_idx, (pass_name, k_order) in enumerate(passes):
        print(f"[KD-LP] Phase2: Core→Border Pass{pass_idx+1}/2 ({pass_name}): k=1..{k_max}")
        for k in k_order:
            border_labels_ct = _propagate_one_stride_border(
                engine, keypack, adj_k_half_list,
                border_labels_ct, non_core_ct, core_labels_ct, core_mask_ct,
                k, N, label_scale, secret_key=secret_key,
            )
        _dbg(engine, secret_key, border_labels_ct, f"P2-{pass_name} 완료", N)

    # ── 최종 합산 ─────────────────────────────────────────────
    final_ct = _refresh(engine,
        engine.add(core_labels_ct, border_labels_ct), keypack)
    _dbg(engine, secret_key, final_ct, f"최종 [0,{N}]", N)
    return final_ct



# ══════════════════════════════════════════════════════════════════════════
# 기존: ALL strides sweep (fallback, min_pts<4 시 사용)
# ══════════════════════════════════════════════════════════════════════════

def fhe_sweep_propagation(
    engine: Engine,
    keypack: KeyPack,
    adj_k_half_list: list,
    core_ct: Ciphertext,
    num_points: int,
    secret_key=None,
    num_sweeps: int = None,
) -> Ciphertext:
    """
    ALL strides sweep 기반 라벨 전파 (fallback).

    사용 시점: min_pts < 4 (chain형 클러스터 허용 시)
    num_sweeps: ceil(N / min_pts / 2) (decide_propagation_mode에서 계산)

    정확성 보장: num_sweeps ≥ ceil(diameter/2)이면 무조건 수렴
    """
    N = num_points
    if num_sweeps is None:
        num_sweeps = math.ceil(math.log2(N))

    N_half      = N // 2
    label_scale = float(N)
    slot_count  = engine.slot_count

    print(f"[LP-sweep] N={N}, N_half={N_half}, num_sweeps={num_sweeps}")
    print(f"           총 fhe_max ≈ 2ph × {num_sweeps}sw × {N_half}str × 2방향 = "
          f"{2 * num_sweeps * N_half * 2}회")

    core_mask_ct = _refresh(engine, core_ct, keypack)
    non_core_ct  = _refresh(engine,
        engine.subtract(engine.encode([1.0] * slot_count), core_mask_ct), keypack)

    id_enc = engine.encode(
        [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
    core_labels_ct = _refresh(engine,
        engine.multiply(core_mask_ct, id_enc), keypack)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N)

    # Phase 1: Core-Core
    for sweep in range(num_sweeps):
        print(f"[LP-sweep] P1 Sweep {sweep+1}/{num_sweeps}")
        for k in range(1, N_half + 1):
            core_labels_ct = _propagate_one_stride_core(
                engine, keypack, adj_k_half_list,
                core_labels_ct, core_mask_ct,
                k, N, label_scale, secret_key=secret_key,
            )
        _dbg(engine, secret_key, core_labels_ct, f"P1 Sweep{sweep+1} 완료", N)

    _dbg(engine, secret_key, core_labels_ct, "P1 완료", N)

    # Phase 2: Core→Border
    border_labels_ct = _refresh(engine,
        engine.multiply(engine.encode([0.0] * slot_count), non_core_ct), keypack)

    for sweep in range(num_sweeps):
        print(f"[LP-sweep] P2 Sweep {sweep+1}/{num_sweeps}")
        for k in range(1, N_half + 1):
            border_labels_ct = _propagate_one_stride_border(
                engine, keypack, adj_k_half_list,
                border_labels_ct, non_core_ct, core_labels_ct, core_mask_ct,
                k, N, label_scale, secret_key=secret_key,
            )
        _dbg(engine, secret_key, border_labels_ct, f"P2 Sweep{sweep+1} 완료", N)

    final_ct = _refresh(engine,
        engine.add(core_labels_ct, border_labels_ct), keypack)
    _dbg(engine, secret_key, final_ct, f"최종 [0,{N}]", N)
    return final_ct


# ── 하위 호환 alias ──────────────────────────────────────────────────────
def fhe_doubling_propagation_fhe(engine, keypack, adjacency_ct_list, core_ct,
                                  num_points, secret_key=None):
    """deprecated: fhe_sweep_propagation 사용 권장."""
    print("[WARNING] deprecated → fhe_sweep_propagation 사용하세요.")
    return fhe_sweep_propagation(
        engine, keypack, adjacency_ct_list, core_ct, num_points,
        secret_key=secret_key,
    )

fhe_doubling_propagation_fhe_debug = fhe_sweep_propagation
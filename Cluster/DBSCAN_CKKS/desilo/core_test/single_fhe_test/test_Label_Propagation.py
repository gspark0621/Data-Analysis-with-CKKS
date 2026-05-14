# core/ciphertext_single/Label_Propagation.py
from __future__ import annotations
import math
import numpy as np
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp

_MCP_LABEL_PATH       = "mcp_alpha12.json"   # ← α=12로 변경
_mcp_label_components = None

def _get_mcp_alpha12():   # 함수명 유지 (호환성)
    global _mcp_label_components
    if _mcp_label_components is None:
        print(f"  [LabelProp] Label용 MCP 로드 (α=12): {_MCP_LABEL_PATH}")
        _mcp_label_components = load_mcp(_MCP_LABEL_PATH)
    return _mcp_label_components


def _dbg(engine, secret_key, ct, tag, num_points, show=8):
    if secret_key is None:
        return ct
    vals = np.array(engine.decrypt(ct, secret_key))[:num_points]
    vmin, vmax, vmean = vals.min(), vals.max(), vals.mean()
    print(f"  [DBG] {tag}: min={vmin:.6f}  max={vmax:.6f}  mean={vmean:.6f}")
    print(f"         샘플={np.round(vals[:show], 6).tolist()}")
    return ct


def _refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack) -> Ciphertext:
    return engine.bootstrap(
        engine.intt(ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key,
    )


def _eval_one_component(engine, ct, comp, slot_count, keypack, relin_key):
    """
    comp 하나를 평가.
    margin 적용된 MCP에서 domain_b = 1 + t_{i-1} (margin 포함).
    입력을 domain_b로 나눠 [-1, 1] 내로 정규화.
    margin buffer 덕분에 실제 입력이 더 안전하게 [-1, 1] 안에 들어온다.
    """
    coeffs   = comp["coeffs"]
    domain_b = comp["domain_b"]

    inv_b  = engine.encode([1.0 / domain_b] * slot_count)
    x_sc   = engine.multiply(ct, inv_b)

    x_sq  = engine.square(x_sc, relin_key)
    x_pow = x_sc

    result = engine.multiply(x_pow, engine.encode([coeffs[0]] * slot_count))

    for k in range(1, len(coeffs)):
        x_pow  = engine.multiply(x_pow, x_sq, relin_key)
        result = engine.add(
            result,
            engine.multiply(x_pow, engine.encode([coeffs[k]] * slot_count)),
        )

    return result


def _eval_mcp(engine, ct, components, slot_count, keypack,
              secret_key=None, num_points=None, tag="mcp"):
    """MCP 전체 합성 평가."""
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key
    current   = ct

    _dbg(engine, secret_key, current,
         f"{tag}/MCP전체입력", num_points or slot_count)

    for step_idx, comp in enumerate(components):
        current = _eval_one_component(
            engine, current, comp, slot_count, keypack, relin_key
        )

        _dbg(engine, secret_key, current,
             f"{tag}/step{step_idx+1}출력(domain_b={comp['domain_b']:.4f})",
             num_points or slot_count)

        if step_idx < len(components) - 1:
            current = engine.intt(current)
            current = engine.bootstrap(current, relin_key, conj_key, boot_key)
            _dbg(engine, secret_key, current,
                 f"{tag}/step{step_idx+1}bootstrap후",
                 num_points or slot_count)

    return current


def fhe_sgn(
    engine     : Engine,
    x_ct       : Ciphertext,
    num_points : int,
    keypack    : KeyPack,
    secret_key = None,
    tag        : str = "",
) -> Ciphertext:
    """MCP로 sgn(x) 근사. margin 적용된 계수로 CKKS 노이즈에 robust."""
    slot_count = engine.slot_count
    components = _get_mcp_alpha12()
    return _eval_mcp(
        engine, x_ct, components, slot_count, keypack,
        secret_key=secret_key,
        num_points=num_points,
        tag=f"{tag}sgn",
    )


def fhe_max(
    engine     : Engine,
    u_ct       : Ciphertext,
    v_ct       : Ciphertext,
    num_points : int,
    keypack    : KeyPack,
    secret_key = None,
    tag        : str = "",
) -> Ciphertext:
    """
    논문 Algorithm 8: MinimaxMax
        max(u, v) = ((u + v) + (u − v) · sgn(u − v)) / 2

    전제: u, v ∈ [0, 1]  →  u−v ∈ [−1, 1]  →  MCP 입력 조건 충족

    [수정 1] u_minus_v 이중 refresh 제거 → 1회만 refresh
    [수정 2] 논문 Algorithm 8 식으로 통합:
             ((u+v) + (u-v)·sgn) / 2  (마지막 한 번의 /2)
    [수정 3] (u+v) 계산을 sgn 평가 이후로 이동 → 레벨 불균형 완화
    """
    slot_count = engine.slot_count
    relin_key  = keypack.relinearization_key

    # ── 1. u − v: 1회만 refresh ──────────────────────────────────
    u_minus_v = engine.subtract(u_ct, v_ct)
    u_minus_v = _refresh(engine, u_minus_v, keypack)
    _dbg(engine, secret_key, u_minus_v,
         f"{tag}max/u-v(refreshed)", num_points)

    # ── 2. sgn(u − v) ────────────────────────────────────────────
    sgn_ct = fhe_sgn(
        engine, u_minus_v, num_points, keypack,
        secret_key=secret_key, tag=f"{tag}max/",
    )
    _dbg(engine, secret_key, sgn_ct,
         f"{tag}max/sgn출력(±1 근사)", num_points)

    # ── 3. sgn 출력 refresh ──────────────────────────────────────
    sgn_ct = _refresh(engine, sgn_ct, keypack)

    # ── 4. (u − v) · sgn(u − v) ─────────────────────────────────
    # u_minus_v는 step 1에서 이미 refresh → 추가 refresh 불필요
    diff_sgn = engine.multiply(u_minus_v, sgn_ct, relin_key)
    _dbg(engine, secret_key, diff_sgn,
         f"{tag}max/(u-v)·sgn", num_points)

    # ── 5. ((u + v) + (u − v)·sgn) / 2 ──────────────────────────
    sum_uv   = engine.add(u_ct, v_ct)
    combined = engine.add(sum_uv, diff_sgn)
    half_enc = engine.encode([0.5] * slot_count)
    result   = engine.multiply(combined, half_enc)

    return _refresh(engine, result, keypack)


def fhe_circular_shift(engine, ct, k, num_points, keypack):
    slot_count    = engine.slot_count
    left_shifted  = engine.rotate(ct, keypack.rotation_key, k)
    right_shifted = engine.rotate(ct, keypack.rotation_key, -(num_points - k))

    mask_left  = [1.0 if i < (num_points - k) else 0.0 for i in range(slot_count)]
    mask_right = [0.0] * slot_count
    for i in range(num_points - k, num_points):
        mask_right[i] = 1.0

    return engine.add(
        engine.multiply(left_shifted,  engine.encode(mask_left)),
        engine.multiply(right_shifted, engine.encode(mask_right)),
    )


def fhe_doubling_propagation_fhe(
    engine            : Engine,
    keypack           : KeyPack,
    adjacency_ct_list : list,
    core_ct           : Ciphertext,
    cluster_id_pt     : list,
    num_points        : int,
    secret_key        = None,
) -> Ciphertext:
    relin_key  = keypack.relinearization_key
    slot_count = engine.slot_count
    num_rounds = math.ceil(math.log2(num_points))

    print(f"[Server][Doubling] N={num_points}, rounds={num_rounds}")

    # ── 초기화 ──────────────────────────────────────────────────────
    core_mask_ct = _refresh(engine, core_ct, keypack)
    non_core_ct  = _refresh(engine,
        engine.subtract(engine.encode([1.0] * slot_count), core_mask_ct), keypack)

    id_enc = engine.encode(cluster_id_pt + [0.0] * (slot_count - num_points))
    core_labels_ct = _refresh(engine,
        engine.multiply(core_mask_ct, id_enc), keypack)

    _dbg(engine, secret_key, core_labels_ct,
         "초기 core_labels_ct", num_points)

    # ── Phase 1: Core-Core Doubling ────────────────────────────────
    for r in range(num_rounds):
        stride = 1 << r
        if stride >= num_points:
            break
        print(f"[Server][Doubling] P1 Round {r+1}/{num_rounds} stride={stride}")

        adj_ct  = adjacency_ct_list[stride - 1]
        s_label = fhe_circular_shift(engine, core_labels_ct, stride, num_points, keypack)
        s_mask  = fhe_circular_shift(engine, core_mask_ct,   stride, num_points, keypack)

        neighbor = _refresh(engine,
            engine.multiply(
                engine.multiply(adj_ct, s_mask, relin_key),
                s_label, relin_key,
            ), keypack)
        _dbg(engine, secret_key, neighbor,
             f"P1-R{r+1}/neighbor", num_points)

        core_labels_ct = fhe_max(
            engine, core_labels_ct, neighbor,
            num_points, keypack,
            secret_key=secret_key, tag=f"P1-R{r+1}/",
        )
        _dbg(engine, secret_key, core_labels_ct,
             f"P1-R{r+1}/max후", num_points)

        core_labels_ct = _refresh(engine,
            engine.multiply(core_labels_ct, core_mask_ct, relin_key), keypack)
        _dbg(engine, secret_key, core_labels_ct,
             f"P1-R{r+1}/core_mask후", num_points)

    # ── Phase 2: Border Doubling ───────────────────────────────────
    border_labels_ct = _refresh(engine,
        engine.multiply(
            engine.encode([0.0] * slot_count), core_mask_ct
        ), keypack)

    for r in range(num_rounds):
        stride = 1 << r
        if stride >= num_points:
            break
        print(f"[Server][Doubling] P2 Round {r+1}/{num_rounds} stride={stride}")

        adj_ct  = adjacency_ct_list[stride - 1]
        s_label = fhe_circular_shift(engine, core_labels_ct, stride, num_points, keypack)
        s_mask  = fhe_circular_shift(engine, core_mask_ct,   stride, num_points, keypack)

        candidate = _refresh(engine,
            engine.multiply(
                engine.multiply(adj_ct, s_mask, relin_key),
                s_label, relin_key,
            ), keypack)
        _dbg(engine, secret_key, candidate,
             f"P2-R{r+1}/candidate", num_points)

        border_labels_ct = fhe_max(
            engine, border_labels_ct, candidate,
            num_points, keypack,
            secret_key=secret_key, tag=f"P2-R{r+1}/",
        )
        _dbg(engine, secret_key, border_labels_ct,
             f"P2-R{r+1}/max후", num_points)

        border_labels_ct = _refresh(engine,
            engine.multiply(border_labels_ct, non_core_ct, relin_key), keypack)
        _dbg(engine, secret_key, border_labels_ct,
             f"P2-R{r+1}/non_core후", num_points)

    # ── 최종 합산 ──────────────────────────────────────────────────
    final_ct = _refresh(engine,
        engine.add(core_labels_ct, border_labels_ct), keypack)
    _dbg(engine, secret_key, final_ct, "최종(core+border)", num_points)
    return final_ct


fhe_doubling_propagation_fhe_debug = fhe_doubling_propagation_fhe

def fhe_max_propagation_fhe(engine, keypack, adjacency_ct_list, core_ct,
                             cluster_id_pt, num_points, max_iter=None):
    print("[WARNING] deprecated → fhe_doubling_propagation_fhe() 사용하세요.")
    return fhe_doubling_propagation_fhe(
        engine, keypack, adjacency_ct_list, core_ct,
        cluster_id_pt, num_points,
    )
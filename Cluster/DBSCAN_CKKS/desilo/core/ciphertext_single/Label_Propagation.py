# core/ciphertext_single/Label_Propagation.py
#
# ── 변경사항 (2026-05c) ───────────────────────────────────────────────────
# [sgn 정밀화] fhe_sgn에 sign_cleaning 추가 (라벨 단조하강 버그 수정)
#   발견: sign_bootstrap 출력 sgn이 ~0.999 (정확히 ±1 아님). fhe_max=
#         ((u+v)+(u-v)·sgn)/2 에서 sgn<1이면 max가 (u-v)(1-sgn)/2 만큼 하강.
#         3780회 누적 → 라벨 211→160, 클러스터 라벨 159~165 압축 → 충돌
#         → ARI 54.7 (7→4 클러스터). 평문 이상sgn은 ARI 100.
#   수정: sign_bootstrap 후 sign_cleaning g(x)=(3/2)x-(1/2)x³ → ±1 정밀화.
#         평문검증: sgn 0.999 가정 시 cleaning 없으면 ARI 79.5(max 212→100),
#         cleaning 1회면 ARI 100(max 212 유지).
# [작업 A 보강] core_mask cleaning (0.714 감쇠 버그 수정 — 위 별도 주석)
# [작업 B] fhe_kd_dense_propagation에 n_rounds 추가
#
# ── 변경사항 (2026-05) ────────────────────────────────────────────────────
# [변경] fhe_sgn: bsgs_poly.eval_mcp_full → bsgs_chebyshev.eval_mcp_full_chebyshev
#        MCP 파일: mcp_alpha15_lp.json → mcp_alpha15_lp_cheb.json
#
#   이유: Power basis는 deg=27에서 T_27 leading coefficient = 6.7×10^7 같은
#         거대 계수 발생 → CKKS plaintext mult noise 폭발 → 라벨 drift 악화.
#         Chebyshev basis로 평가하면 계수 폭발 회피, drift 감소 기대.
#
# [기존 변경사항 유지]
# - fhe_kd_dense_propagation(), fhe_sweep_propagation()
# - _propagate_one_stride_core / _propagate_one_stride_border 헬퍼
# ─────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
import numpy as np
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev   # ★ Chebyshev
from core.ciphertext_single.cleaning import bit_cleaning, sign_cleaning   # ★ [2026-05c] mask + sgn 정밀도


_MCP_LABEL_PATH = "mcp_alpha15_lp_cheb.json"   # ★ Chebyshev basis (이전: mcp_alpha15_lp.json)
# α=15: τ=2^{-15}, drift=840×30×τ/2≈0.39 < 1.0 ✓
# degrees=[7,15,15,15,27] (논문 Table 2)
# Chebyshev BSGS dep(27)=5 → 10레벨 ≤ budget 10 ✓

# ★ [2026-05c] fhe_sgn의 sign_cleaning 반복 횟수.
#   1회면 0.999 → 0.99999. 2회면 CKKS 한계(2^-32).
# ★ [Tier 1a / 2026-06] 1 → 2.
#   진단: fhe_max가 수천 회 누적되며 sgn<1 잔차가 라벨을 압축(hepta 160~166).
#   2회로 sgn을 ±1에 기계 정밀도까지 붙여 max 하강 제거.
#   비용: fhe_max마다 호출 → 시간 증가(가장 큰 비용 지점). FHE에서 1 vs 2 측정 권장.
_SGN_CLEANING_ITERS = 2

# ★ [Tier 1a / 2026-06] 초기 core_mask/non_core cleaning 반복 횟수 (Core와 동일 2).
_INIT_CLEANING_ITERS = 2
_mcp_label_components = None


def _get_mcp_label():
    global _mcp_label_components
    if _mcp_label_components is None:
        print(f"  [LabelProp] MCP 로드: {_MCP_LABEL_PATH}")
        _mcp_label_components = load_mcp(_MCP_LABEL_PATH)
        # basis 확인
        basis = _mcp_label_components[0].get("basis", "power")
        if basis != "chebyshev":
            raise ValueError(
                f"[LabelProp] {_MCP_LABEL_PATH} has basis='{basis}', expected 'chebyshev'. "
                f"JSON 재생성 필요: compute_mcp_for_label_prop_chebyshev() 사용."
            )
    return _mcp_label_components


def _dbg(engine, secret_key, ct, tag, num_points, show=6, save_path=None):
    """라벨 상태 디버그. min/max/mean + '서로 구분되는 정수 라벨 수'(distinct)를 출력.

    ★ [2026-06] distinct 추적:
      클러스터가 압축/병합되면 round 후 distinct 수가 정답 클러스터 수보다 작아짐.
      round 단계 이전에 압축이 진행되는지 pass마다 직접 관찰 가능.
      save_path 주면 해당 pass 라벨 전체를 CSV로 저장(클라이언트 round 없이 원시값).
    """
    if secret_key is None:
        return ct
    vals = np.real(np.array(engine.decrypt(ct, secret_key)))[:num_points]
    pos  = vals[vals > 0.5]
    distinct = len(set(int(round(v)) for v in pos)) if len(pos) else 0
    print(f"  [DBG] {tag}: min={vals.min():.4f} max={vals.max():.4f} "
          f"mean={vals.mean():.4f} | 양수라벨 distinct(round)={distinct}")
    print(f"         앞{show}개={np.round(vals[:show], 4).tolist()}")
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("Heap_Position,Label_Raw,Label_Round\n")
                for i, v in enumerate(vals):
                    f.write(f"{i},{float(v):.6f},{int(round(v)) if v>0.5 else -1}\n")
            print(f"         ↳ 스냅샷 저장: {save_path}")
        except Exception as e:
            print(f"         ↳ 스냅샷 저장 실패: {e}")
    return ct


def _refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack) -> Ciphertext:
    """일반 bootstrap."""
    return engine.bootstrap(
        engine.intt(ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key,
    )


def fhe_sgn(
    engine: Engine, x_ct: Ciphertext, num_points: int, keypack: KeyPack,
    secret_key=None, tag: str = "",
) -> Ciphertext:
    """
    MCP + sign_bootstrap + sign_cleaning 으로 sgn(x) 근사. Chebyshev BSGS 기반.

    ★ [2026-05c sgn 정밀화] sign_cleaning 추가 이유:
      [발견된 버그] sign_bootstrap 출력 sgn이 정확히 ±1이 아니라 ~0.999.
        fhe_max = ((u+v)+(u-v)·sgn)/2 에서 sgn=0.999면 max가
        (u-v)·0.0005 만큼 진짜 최댓값보다 작아짐 (max인데 값 하강!).
        fhe_max 3780회 누적 → 라벨 211→160 단조 하강 (round당 ~6).
        클러스터별 라벨이 159~165로 압축 → 서로 다른 클러스터 충돌
        → ARI 54.7 (클러스터 7→4). 평문 이상적 sgn은 ARI 100.
      [해결] sign_bootstrap 출력에 sign_cleaning g(x)=(3/2)x-(1/2)x³ 적용.
        0.999 → 0.99999... 로 quadratic 수렴 → max 하강 제거.
      [비용] fhe_max마다 호출되어 시간 증가하나, 라벨 정밀도가 핵심이므로 필수.
        레벨은 cleaning.py의 자동 _ensure_level이 관리.
    """
    slot_count = engine.slot_count
    components = _get_mcp_label()

    # ★ Chebyshev BSGS 평가 (Bossuat Algorithm 1)
    result = eval_mcp_full_chebyshev(
        engine, x_ct, components, slot_count, keypack, tag="LP "
    )
    result = engine.sign_bootstrap(
        engine.intt(result),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )
    # ★ [2026-05c] sign_cleaning: sgn을 정확히 ±1로 (max 하강 제거)
    result = sign_cleaning(
        engine, result, keypack,
        n_iters=_SGN_CLEANING_ITERS, slot_count=slot_count,
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
    apply_mask: bool = True,          # ★ [Tier 1b] False면 per-stride 마스킹 생략
):
    """Core-Core 전파: stride k에 대해 Forward + Backward.

    ★ [Tier 1b] apply_mask:
      True  = 기존(per-stride): 매 max 직후 × core_mask + _refresh(=bootstrap).
      False = per-pass 모드: 여기선 마스킹 생략(호출측이 pass 끝에 1회 수행).
      정합성 근거(평문 검증 ①=②): 전파는 항상 shift(core_mask)로 *소스*를
      게이트하므로 비-core 슬롯의 오염은 다른 슬롯으로 전파되지 않음 → core 라벨
      결과 불변. per-pass는 점당 수천 회의 곱셈+bootstrap을 제거(비용↓).
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
    if apply_mask:
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
        if apply_mask:
            core_labels_ct = _refresh(engine,
                engine.multiply(core_labels_ct, core_mask_ct, relin_key), keypack)

    return core_labels_ct


# ── 헬퍼: stride 하나 전파 (Core→Border) ─────────────────────────────────

def _propagate_one_stride_border(
    engine, keypack, adj_k_half_list,
    border_labels_ct, non_core_ct, core_labels_ct, core_mask_ct,
    k, N, label_scale,
    secret_key=None,
    apply_mask: bool = True,          # ★ [Tier 1b]
):
    """Core→Border 전파: stride k에 대해 Forward + Backward.

    ★ [Tier 1b] border_labels는 *목적지 전용*(소스는 core_labels)이라 오염이
      전파되지 않음 → per-pass 마스킹으로 동일 결과(평문 검증 ①=②).
    """
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
    if apply_mask:
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
        if apply_mask:
            border_labels_ct = _refresh(engine,
                engine.multiply(border_labels_ct, non_core_ct, relin_key), keypack)

    return border_labels_ct


# ══════════════════════════════════════════════════════════════════════════
# KD-tree dense stride 라벨 전파 (kd_dense, min_pts ≥ 4)
# ══════════════════════════════════════════════════════════════════════════

def fhe_kd_dense_propagation(
    engine: Engine,
    keypack: KeyPack,
    adj_k_half_list: list,
    core_ct: Ciphertext,
    num_points: int,
    k_max: int,
    secret_key=None,
    n_rounds: int = 1,          # ★ [2026-05c 작업 B] Core-Core 전파 반복 횟수
    mask_mode: str = "per_stride",   # ★ [Tier 1b] "per_stride"(기존) | "per_pass"
    debug_snapshot_prefix: str = None,  # ★ [2026-06] 주면 pass마다 라벨 CSV 저장
) -> Ciphertext:
    """
    KD-tree ordering + dense stride k=1..k_max 라벨 전파.

    Forward + backward 2 pass × Core-Core, Core→Border 2 phase.

    ★ [2026-05c 작업 B] n_rounds:
      기존 고정 2 pass(forward+backward 1회)는 hepta(구형) 같은 짧은 지름
      클러스터에만 충분. two-moons/circles 같은 chain형(긴 지름)은 라벨이
      클러스터 끝까지 전파되려면 2 pass로 부족 (실측: circles 16 pass 필요).

      n_rounds회 반복 = 2*n_rounds pass. Client가 ⌈log₂N⌉ round로 결정
      (검증: 2*log₂N pass가 hepta/moons/circles/blobs 모두 ARI>=0.9 커버).

      Phase 1(Core-Core)만 반복. Phase 2(Core→Border)는 Core 수렴 후 1회면 충분
      (border는 인접 core 라벨을 1-hop 받기만 하므로).

      작업 A로 core_mask=1.0 (cleaning) 확보 시, round 늘려도 라벨 damping 無
      → 과대 추정해도 안전 (시간만 증가).

    fhe_max 횟수 = Phase1(2*2*k_max*n_rounds) + Phase2(2*2*k_max) ... 직접 계산은 아래.
    """
    N     = num_points
    k_max = min(k_max, N // 2)
    T_kmax = k_max * (k_max + 1) // 2

    label_scale = float(N)
    slot_count  = engine.slot_count
    per_pass    = (mask_mode == "per_pass")   # ★ [Tier 1b]
    apply_mask_in_stride = (not per_pass)

    print(f"\n[KD-LP] ══════════════════════════════════════════")
    print(f"[KD-LP] KD-tree dense stride 전파 (k=1..{k_max})")
    print(f"[KD-LP] N={N}, k_max={k_max}, n_rounds={n_rounds} (★ 작업 B)")
    print(f"[KD-LP] mask_mode={mask_mode} (★ Tier 1b)  "
          f"sgn_cleaning_iters={_SGN_CLEANING_ITERS} init_cleaning_iters={_INIT_CLEANING_ITERS} (★ Tier 1a)")
    print(f"[KD-LP] T({k_max})={T_kmax}  {'✓ ≥ N' if T_kmax >= N else '⚠ < N'}")
    fhe_max_cnt = (2 * n_rounds + 2) * k_max * 2
    # per_stride: 마스킹(곱+bootstrap)이 fhe_max와 동수 발생. per_pass: pass당 1회.
    n_mask_per_stride = fhe_max_cnt
    n_mask_per_pass   = (2 * n_rounds + 2)
    print(f"[KD-LP] fhe_max: {fhe_max_cnt}회 "
          f"(Phase1 {2*n_rounds*k_max*2} + Phase2 {2*k_max*2})")
    print(f"[KD-LP] 마스킹(곱+bootstrap) 횟수: "
          f"per_stride={n_mask_per_stride} vs per_pass={n_mask_per_pass} "
          f"→ Tier 1b 절감={n_mask_per_stride-n_mask_per_pass}회")
    print(f"[KD-LP] ══════════════════════════════════════════\n")

    # ── 초기화 ─────────────────────────────────────────────────
    # ★ [2026-05c 작업 A 보강] core_mask 정밀도 복원
    #   _refresh(일반 bootstrap)가 noise 주입 → bit_cleaning으로 1.0 재정리.
    # ★ [Tier 1a] n_iters를 _INIT_CLEANING_ITERS(=2)로 상향 → mask=1.0 기계정밀도.
    core_mask_ct = _refresh(engine, core_ct, keypack)
    core_mask_ct = bit_cleaning(engine, core_mask_ct, keypack,
                                n_iters=_INIT_CLEANING_ITERS, slot_count=slot_count)
    non_core_ct  = engine.subtract(engine.encode([1.0] * slot_count), core_mask_ct)
    non_core_ct  = _refresh(engine, non_core_ct, keypack)
    non_core_ct  = bit_cleaning(engine, non_core_ct, keypack,
                                n_iters=_INIT_CLEANING_ITERS, slot_count=slot_count)

    id_enc = engine.encode(
        [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
    core_labels_ct = _refresh(engine,
        engine.multiply(core_mask_ct, id_enc), keypack)

    def _snap(name):
        return (f"{debug_snapshot_prefix}_{name}.csv"
                if debug_snapshot_prefix else None)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N,
         save_path=_snap("init"))

    passes = [
        ("forward",  list(range(1, k_max + 1))),
        ("backward", list(range(k_max, 0, -1))),
    ]

    def _mask_core(lab):
        return _refresh(engine,
            engine.multiply(lab, core_mask_ct, keypack.relinearization_key), keypack)

    def _mask_border(lab):
        return _refresh(engine,
            engine.multiply(lab, non_core_ct, keypack.relinearization_key), keypack)

    # ── Phase 1: Core-Core (★ 작업 B: n_rounds 반복) ──────────────
    for rnd in range(n_rounds):
        for pass_idx, (pass_name, k_order) in enumerate(passes):
            print(f"[KD-LP] Phase1: Core-Core Round{rnd+1}/{n_rounds} "
                  f"Pass{pass_idx+1}/2 ({pass_name})")
            for k in k_order:
                core_labels_ct = _propagate_one_stride_core(
                    engine, keypack, adj_k_half_list,
                    core_labels_ct, core_mask_ct,
                    k, N, label_scale, secret_key=secret_key,
                    apply_mask=apply_mask_in_stride,    # ★ Tier 1b
                )
            if per_pass:                                # ★ Tier 1b: pass당 1회 마스킹
                core_labels_ct = _mask_core(core_labels_ct)
            _dbg(engine, secret_key, core_labels_ct,
                 f"P1-R{rnd+1}-{pass_name} 완료", N,
                 save_path=_snap(f"P1_R{rnd+1}_{pass_name}"))

    # per_pass 모드: Phase 끝에서 비-core 슬롯 오염 최종 정리 (안전)
    if per_pass:
        core_labels_ct = _mask_core(core_labels_ct)

    # ── Phase 2: Core→Border ───────────────────────────────────
    border_labels_ct = _refresh(engine,
        engine.multiply(engine.encode([0.0] * slot_count), non_core_ct), keypack)

    for pass_idx, (pass_name, k_order) in enumerate(passes):
        print(f"[KD-LP] Phase2: Core→Border Pass{pass_idx+1}/2 ({pass_name})")
        for k in k_order:
            border_labels_ct = _propagate_one_stride_border(
                engine, keypack, adj_k_half_list,
                border_labels_ct, non_core_ct, core_labels_ct, core_mask_ct,
                k, N, label_scale, secret_key=secret_key,
                apply_mask=apply_mask_in_stride,        # ★ Tier 1b
            )
        if per_pass:
            border_labels_ct = _mask_border(border_labels_ct)
        _dbg(engine, secret_key, border_labels_ct, f"P2-{pass_name} 완료", N,
             save_path=_snap(f"P2_{pass_name}"))

    if per_pass:
        border_labels_ct = _mask_border(border_labels_ct)

    final_ct = _refresh(engine,
        engine.add(core_labels_ct, border_labels_ct), keypack)
    _dbg(engine, secret_key, final_ct, f"최종 [0,{N}]", N, save_path=_snap("final"))
    return final_ct


# ══════════════════════════════════════════════════════════════════════════
# ALL strides sweep (fallback, min_pts < 4)
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
    """ALL strides sweep 기반 라벨 전파 (fallback)."""
    N = num_points
    if num_sweeps is None:
        num_sweeps = math.ceil(math.log2(N))

    N_half      = N // 2
    label_scale = float(N)
    slot_count  = engine.slot_count

    print(f"[LP-sweep] N={N}, N_half={N_half}, num_sweeps={num_sweeps}")

    # ★ [2026-05c 작업 A 보강] kd_dense와 동일: mask cleaning으로 감쇠 방지.
    core_mask_ct = _refresh(engine, core_ct, keypack)
    core_mask_ct = bit_cleaning(engine, core_mask_ct, keypack,
                                n_iters=1, slot_count=slot_count)
    non_core_ct  = engine.subtract(engine.encode([1.0] * slot_count), core_mask_ct)
    non_core_ct  = _refresh(engine, non_core_ct, keypack)
    non_core_ct  = bit_cleaning(engine, non_core_ct, keypack,
                                n_iters=1, slot_count=slot_count)

    id_enc = engine.encode(
        [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
    core_labels_ct = _refresh(engine,
        engine.multiply(core_mask_ct, id_enc), keypack)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N)

    for sweep in range(num_sweeps):
        print(f"[LP-sweep] P1 Sweep {sweep+1}/{num_sweeps}")
        for k in range(1, N_half + 1):
            core_labels_ct = _propagate_one_stride_core(
                engine, keypack, adj_k_half_list,
                core_labels_ct, core_mask_ct,
                k, N, label_scale, secret_key=secret_key,
            )
        _dbg(engine, secret_key, core_labels_ct, f"P1 Sweep{sweep+1} 완료", N)

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
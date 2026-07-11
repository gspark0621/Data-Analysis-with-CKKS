# core/ciphertext_single/Label_Propagation.py
#
# ── 변경사항 (2026-06 Phase 1: 라벨 노이즈 주입 경로 제거) ─────────────────
# [배경] hepta 런 분석: 라벨 raw 값이 160~166 띠로 비단조 압축 (C5 212→165.8,
#   C3 106→161.5 상승). 원인 분해:
#   (1) fhe_max당 표준 bootstrap 3~4회가 라벨 값에 직접 노이즈 주입
#       (sign_bootstrap 직후 2^-18 정밀 → 일반 bootstrap이 2^-9.3로 악화,
#        cleaning.py 헤더 실측). 특히 _refresh(u_minus_v)의 노이즈는
#        sgn(≈±1)에 곱해져 감쇠 없이 라벨에 영구 주입되던 최악 채널.
#   (2) minimax 오차는 two-sided (±2^-α 등진동) → sgn>1이면 라벨 상승,
#       sgn<1이면 하강 → 비단조 스크램블.
#
# [Phase 1 수정 4건]
#   ① fhe_max 재구성: max(u,v) = u + (v−u)·step,  step=(1+sgn)/2.
#      - d=v−u는 refresh하지 않음 (라벨 경로 노이즈 주입 제거)
#      - sgn 가지만 분리 refresh → 그 노이즈는 sign_bootstrap이 이차 squash
#        (Hong et al. Theorem 1: τ → (π²/8)τ²)
#      - ×0.5는 step에 병합 (신선한 sgn 가지에서 계산, 라벨 레벨 무소모)
#      - 최종 _refresh(result) 제거 → 레벨 관리는 lineage guard가 담당
#      → 라벨 lineage의 표준 bootstrap: max당 3~4회 → guard 발동 시에만
#   ② adjm 사전계산 (호이스팅): adj_k ⊙ shift(core_mask,k) ⊙ core_mask(dest)
#      는 라운드 불변 → Phase1 시작 시 1회 계산. destination mask까지
#      접어 넣어 post-max ×core_mask + refresh 제거.
#      메모리: 2×k_max ciphertext (k_max=97, logN=16 기준 ≈ 2GB급) — 로그 출력.
#   ③ 라벨 lineage 레벨 가드: 곱셈 전 ct.level < _LABEL_MIN_LEVEL이면 그
#      지점에서만 _refresh. "라벨이 받는 노이즈 주입 = 가드 발동 횟수"로
#      명시·감사 가능 (카운터 출력). budget=10이면 방향당 ~1회,
#      엔진을 use_bootstrap_to_17_levels로 바꾸면 ~2방향당 1회로 감소.
#   ④ label_scale = N → 1.1·N 인셋: sgn 입력 |x| ≤ 0.91로 제한.
#      x=1.0 경계 + CKKS 노이즈 ε → T_27(1+ε)≈1+729ε 폭주 →
#      sign_cleaning g(x)가 x>√3에서 음수 반전(라벨 텔레포트) 위험 차단.
#      최소 유의미 라벨 차 1/(1.1N)=4.3e-3 ≫ δ=2^-15 → 근사 보증 유지.
#
# ── 변경사항 (2026-05c) ───────────────────────────────────────────────────
# [sgn 정밀화] fhe_sgn에 sign_cleaning 추가 (라벨 단조하강 버그 수정)
#   발견: sign_bootstrap 출력 sgn이 ~0.999 (정확히 ±1 아님). fhe_max에서
#         sgn<1이면 max가 (u-v)(1-sgn)/2 만큼 하강. 3780회 누적 → 라벨
#         211→160, 클러스터 라벨 159~165 압축 → 충돌 → ARI 54.7.
#   수정: sign_bootstrap 후 sign_cleaning g(x)=(3/2)x-(1/2)x³ → ±1 정밀화.
# [작업 A 보강] core_mask cleaning (0.714 감쇠 버그 수정)
# [작업 B] fhe_kd_dense_propagation에 n_rounds 추가
#
# ── 변경사항 (2026-05) ────────────────────────────────────────────────────
# [변경] fhe_sgn: bsgs_poly.eval_mcp_full → bsgs_chebyshev.eval_mcp_full_chebyshev
#        MCP 파일: mcp_alpha15_lp.json → mcp_alpha15_lp_cheb.json
# ─────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
import numpy as np
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev   # ★ Chebyshev
from core.ciphertext_single.cleaning import bit_cleaning, sign_cleaning


_MCP_LABEL_PATH = "mcp_alpha15_lp_cheb.json"   # ★ Chebyshev basis
# α=15: τ=2^{-15}, degrees=[7,15,15,15,27] (논문 Table 2)
# Chebyshev BSGS dep(27)=5 → 10레벨 ≤ budget 10 ✓

# ★ [2026-05c] fhe_sgn의 sign_cleaning 반복 횟수.
_SGN_CLEANING_ITERS = 1

# ★ [Phase1-④] sgn 입력 도메인 인셋 계수: label_scale = _SCALE_INSET · N
#   → sgn 입력 |x| ≤ 1/_SCALE_INSET ≈ 0.909 (Chebyshev 경계 1.0에서 격리)
_SCALE_INSET = 1.1

# ★ [Phase1-③] 라벨 lineage 레벨 가드 임계.
#   max-direction 1회 소모(예상): label shift(≤2) + neighbor 곱(2) + d×step(2) ≈ 6
#   → 6 + 여유 1 = 7. budget=10 가정. ct.level 런타임 확인이라 실제
#   plaintext 곱 비용(1 또는 2레벨)에 자동 적응. budget 변경 시 이 값만 조정.
_LABEL_MIN_LEVEL = 7

# ★ [Phase1-②] adjm(adj×mask 사전계산) 사용 여부. 메모리 부족 시 False로
#   내리면 스트라이드마다 즉석 계산 (여전히 dest mask 접기는 유지 →
#   post-max ×core_mask는 어느 쪽이든 제거됨).
_HOIST_ADJ_MASKS = True

# ★ [Phase1d/메모리] adjm 캐시 최대 개수 (GPU 메모리 상한).
#   None = 전체(k_max) 캐시. 정수 M = stride k=1..M만 캐시, 나머지는 즉석 계산.
#   sign_bootstrap 작업 메모리를 위해 캐시가 GPU를 다 채우지 않도록 제한.
#   LSUN(N=400, k_max=199)에서 199개 캐시 후 sign_bootstrap OOM → 줄여서 사용.
#   권장 시작값: hepta에서 통과한 105 근방. 더 줄여야 하면 64, 32, 0 순으로.
_MAX_CACHED_ADJM = 105

_mcp_label_components = None

# ★ [Phase1-③] 라벨 lineage refresh 횟수 카운터 (감사용)
_label_refresh_count = 0


def _get_mcp_label():
    global _mcp_label_components
    if _mcp_label_components is None:
        print(f"  [LabelProp] MCP 로드: {_MCP_LABEL_PATH}")
        _mcp_label_components = load_mcp(_MCP_LABEL_PATH)
        basis = _mcp_label_components[0].get("basis", "power")
        if basis != "chebyshev":
            raise ValueError(
                f"[LabelProp] {_MCP_LABEL_PATH} has basis='{basis}', expected 'chebyshev'. "
                f"JSON 재생성 필요: compute_mcp_for_label_prop_chebyshev() 사용."
            )
    return _mcp_label_components


def _dbg(engine, secret_key, ct, tag, num_points, show=6):
    if secret_key is None:
        return ct
    vals = np.array(engine.decrypt(ct, secret_key))[:num_points]
    print(f"  [DBG] {tag}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")
    print(f"         {np.round(vals[:show], 4).tolist()}")
    return ct


def _refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack) -> Ciphertext:
    """일반 bootstrap. ★ 라벨 운반 암호문에는 _ensure_label_level 경유로만 사용."""
    return engine.bootstrap(
        engine.intt(ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key,
    )


def _ensure_label_level(engine, ct, keypack, tag=""):
    """★ [Phase1-③] 라벨 lineage 유일의 표준 bootstrap 지점.

    ct.level < _LABEL_MIN_LEVEL일 때만 refresh. 라벨이 받는 노이즈 주입
    횟수 = 이 함수의 발동 횟수 (카운터로 감사). 그 외 모든 지점의
    라벨 refresh는 Phase 1에서 제거됨.
    """
    global _label_refresh_count
    lvl = getattr(ct, "level", None)
    if lvl is not None and lvl < _LABEL_MIN_LEVEL:
        _label_refresh_count += 1
        ct = _refresh(engine, ct, keypack)
    return ct


def fhe_sgn(
    engine: Engine, x_ct: Ciphertext, num_points: int, keypack: KeyPack,
    secret_key=None, tag: str = "",
) -> Ciphertext:
    """MCP + sign_bootstrap + sign_cleaning 으로 sgn(x) 근사. Chebyshev BSGS 기반.

    ★ sign_cleaning은 sign_bootstrap 직후(레벨 신선)에만 배치.
      이후 일반 bootstrap이 따라오면 2^-18 → 2^-9.3로 악화 (실측) →
      Phase 1에서 fhe_max의 후속 일반 bootstrap을 제거했으므로
      cleaning의 이차 수렴 이득이 보존됨.
    """
    slot_count = engine.slot_count
    components = _get_mcp_label()

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
    # ★ ensure_output_level=False: cleaning 직후 표준 bootstrap이 따라오면
    #   2^-18 → 2^-9.3로 악화 (실측). SB 직후라 레벨 충분 → 후속 refresh 불필요.
    result = sign_cleaning(
        engine, result, keypack,
        n_iters=_SGN_CLEANING_ITERS, slot_count=slot_count,
        ensure_output_level=False,
    )
    return result


def fhe_max(
    engine: Engine, u_ct: Ciphertext, v_ct: Ciphertext,
    num_points: int, keypack: KeyPack,
    label_scale: float = 1.0,
    secret_key=None, tag: str = "",
) -> Ciphertext:
    """max(u,v) = u + (v−u)·step(v−u),  step = (1+sgn(v−u))/2

    ★ [Phase1-①] refresh-minimal 재구성:
      - d=v−u는 refresh하지 않음. (이전: _refresh(u_minus_v)의 bootstrap
        노이즈가 sgn≈±1에 곱해져 감쇠 없이 라벨에 직접 주입되던 최악 채널)
      - sgn 가지(d의 사본)만 분리 refresh → composite에 필요한 레벨 확보.
        이 가지의 노이즈는 sign_bootstrap이 이차 squash (Thm 1) → 무해.
      - step=(1+sgn)/2 는 신선한 sgn 가지에서 계산 (+1, ×0.5 모두
        라벨 레벨 무소모). 이전의 (u+v+(u−v)·sgn)×0.5와 수학적으로 동일.
      - 출력 refresh 없음. 라벨 lineage 레벨은 호출부 가드가 관리.

    라벨 lineage 소모: cipher 곱 1회 (d×step). 출력 레벨 ≈ min(u,v) − 곱 1회분.
    오차: |max_err| = |v−u|·|sgn_err|/2 (기존과 동일 형태).
    """
    slot_count = engine.slot_count
    relin_key  = keypack.relinearization_key

    d = engine.subtract(v_ct, u_ct)            # 레벨 소모 없음, refresh 없음

    # ── sgn 가지 (분리 사본): 여기서만 refresh ──
    d_sgn = _refresh(engine, d, keypack)
    if abs(label_scale - 1.0) > 1e-9:
        d_sgn = engine.multiply(
            d_sgn, engine.encode([1.0 / label_scale] * slot_count))

    sgn_ct = fhe_sgn(engine, d_sgn, num_points, keypack,
                     secret_key=secret_key, tag=f"{tag}max/")

    # step = (1+sgn)/2 — sgn 가지(신선 레벨)에서 계산
    step = engine.add(sgn_ct, engine.encode([1.0] * slot_count))
    step = engine.multiply(step, engine.encode([0.5] * slot_count))

    # ── 라벨 경로: 곱 1회 + 덧셈 ──
    prod = engine.multiply(d, step, relin_key)
    return engine.add(u_ct, prod)


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


# ── ★ [Phase1-②] adjm 사전계산 (라운드 불변 호이스팅) ──────────────────────

def _build_adjm_fwd(engine, keypack, adj_k, dest_mask_ct, core_mask_ct, k, N):
    """adjm_fwd = adj_k ⊙ shift(core_mask, k) ⊙ dest_mask
    source mask(送 측) + destination mask(受 측)를 인접 행렬에 접어 넣음
    → 전파 루프에서 post-max ×mask 불필요.
    build 시 1회 refresh + bit_cleaning (라운드 불변이므로 비용·노이즈 비누적).

    ★ [Phase1c/메모리] backward 마스크는 저장하지 않는다. Core-Core처럼
      source==dest==core_mask인 대칭 케이스에서는
        adjm_bwd[i] = adj_k[(i−k)%N]·core[(i−k)%N]·core[i] = shift(adjm_fwd, N−k)[i]
      이므로 사용 시점에 circular shift(회전 2회, bootstrap 없음)로 유도 가능.
      → 캐시 메모리 2×k_max → k_max (절반). LSUN N=400 CUDA OOM 해결.
      (mask가 비대칭인 Border 경로에는 이 항등식이 성립하지 않으므로 사용 금지 —
       Border는 _build_adjm_border_pair로 즉석 계산.)"""
    relin_key = keypack.relinearization_key

    s_mask_k = fhe_circular_shift(engine, core_mask_ct, k, N, keypack)
    adjm_fwd = engine.multiply(
        engine.multiply(adj_k, s_mask_k, relin_key),
        dest_mask_ct, relin_key)
    adjm_fwd = _refresh(engine, adjm_fwd, keypack)
    adjm_fwd = bit_cleaning(engine, adjm_fwd, keypack, n_iters=1,
                            slot_count=engine.slot_count,
                            ensure_output_level=False)
    return adjm_fwd


def _bwd_from_fwd(engine, keypack, adjm_fwd, k, N):
    """Core-Core 전용 (mask 대칭): adjm_bwd = shift(adjm_fwd, N−k).
    회전 2회 + plaintext 곱 2회 — bootstrap 없음, 레벨 1 소모."""
    return fhe_circular_shift(engine, adjm_fwd, N - k, N, keypack)


def _build_adjm_border_pair(engine, keypack, adj_k, non_core_ct, core_mask_ct, k, N):
    """Border 전용 (mask 비대칭 → shift 트릭 불가): fwd/bwd 명시 계산.
    Phase 2는 2 pass뿐이라 캐시하지 않음."""
    relin_key = keypack.relinearization_key

    fwd = _build_adjm_fwd(engine, keypack, adj_k, non_core_ct, core_mask_ct, k, N)

    bwd = None
    if 2 * k < N:
        adj_Nk    = fhe_circular_shift(engine, adj_k,        N - k, N, keypack)
        s_mask_Nk = fhe_circular_shift(engine, core_mask_ct, N - k, N, keypack)
        bwd = engine.multiply(
            engine.multiply(adj_Nk, s_mask_Nk, relin_key),
            non_core_ct, relin_key)
        bwd = _refresh(engine, bwd, keypack)
        bwd = bit_cleaning(engine, bwd, keypack, n_iters=1,
                           slot_count=engine.slot_count,
                           ensure_output_level=False)
    return fwd, bwd


def _build_adjm_cache(engine, keypack, adj_k_half_list, dest_mask_ct,
                      core_mask_ct, k_max, N, tag="", max_cached=None):
    """k=1..min(k_max, max_cached)의 adjm_fwd만 사전계산.
    반환: dict {k: adjm_fwd}. 캐시 안 된 k는 사용 시 즉석 계산.
    (메모리 상한: sign_bootstrap 작업 공간 확보용. bwd는 shift로 유도.)
    GPU OOM 발생 시 거기까지만 캐시하고 부분 dict 반환 (나머지 즉석)."""
    limit = k_max if max_cached is None else min(k_max, max_cached)
    print(f"[KD-LP] {tag} adjm 사전계산: {limit}/{k_max} ciphertext "
          f"(fwd만 저장, bwd=shift(fwd) 유도; k>{limit}은 즉석 계산)")
    cache = {}
    try:
        for k in range(1, limit + 1):
            cache[k] = _build_adjm_fwd(
                engine, keypack, adj_k_half_list[k - 1],
                dest_mask_ct, core_mask_ct, k, N)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[KD-LP] ⚠ adjm 캐시 중 GPU OOM (k={len(cache)+1}/{limit}) "
                  f"→ {len(cache)}개까지만 캐시, 나머지 즉석 계산")
            return cache if cache else None
        raise
    return cache


# ── 헬퍼: stride 하나 전파 (Core-Core) ────────────────────────────────────

def _propagate_one_stride_core(
    engine, keypack, adjm_cache, adj_k_half_list,
    core_labels_ct, core_mask_ct,
    k, N, label_scale,
    secret_key=None,
):
    """Core-Core 전파: stride k, Forward + Backward.

    ★ [Phase1] 변경:
      - neighbor = adjm ⊙ s_label (adjm은 사전계산, dest mask 포함)
      - neighbor refresh 제거, post-max ×core_mask + refresh 제거
      - 방향 시작 전 라벨 lineage 레벨 가드만 수행
    """
    relin_key = keypack.relinearization_key

    if adjm_cache is not None and k in adjm_cache:
        adjm_fwd = adjm_cache[k]
    else:
        adjm_fwd = _build_adjm_fwd(
            engine, keypack, adj_k_half_list[k - 1],
            core_mask_ct, core_mask_ct, k, N)
    # ★ [Phase1c] bwd는 저장하지 않고 대칭성으로 유도 (메모리 절반)
    adjm_bwd = _bwd_from_fwd(engine, keypack, adjm_fwd, k, N) if 2 * k < N else None

    # Forward: label[i] ← max(label[i], adjm_fwd[i] × label[(i+k)%N])
    core_labels_ct = _ensure_label_level(engine, core_labels_ct, keypack,
                                         tag=f"core k={k} fwd")
    s_label_k  = fhe_circular_shift(engine, core_labels_ct, k, N, keypack)
    neighbor_k = engine.multiply(adjm_fwd, s_label_k, relin_key)
    core_labels_ct = fhe_max(engine, core_labels_ct, neighbor_k,
                             N, keypack, label_scale=label_scale,
                             secret_key=secret_key)

    # Backward: label[i] ← max(label[i], adjm_bwd[i] × label[(i-k)%N])
    if adjm_bwd is not None:
        core_labels_ct = _ensure_label_level(engine, core_labels_ct, keypack,
                                             tag=f"core k={k} bwd")
        s_label_Nk  = fhe_circular_shift(engine, core_labels_ct, N - k, N, keypack)
        neighbor_Nk = engine.multiply(adjm_bwd, s_label_Nk, relin_key)
        core_labels_ct = fhe_max(engine, core_labels_ct, neighbor_Nk,
                                 N, keypack, label_scale=label_scale,
                                 secret_key=secret_key)

    return core_labels_ct


# ── 헬퍼: stride 하나 전파 (Core→Border) ─────────────────────────────────

def _propagate_one_stride_border(
    engine, keypack, adj_k_half_list,
    border_labels_ct, non_core_ct, core_labels_ct, core_mask_ct,
    k, N, label_scale,
    secret_key=None,
):
    """Core→Border 전파: stride k, Forward + Backward.

    ★ [Phase1] Phase 2는 2 pass뿐이므로 캐시 없이 즉석 계산하되,
      dest mask(non_core)를 candidate에 접어 post-max ×non_core + refresh 제거.
    """
    relin_key = keypack.relinearization_key

    adjm_fwd, adjm_bwd = _build_adjm_border_pair(
        engine, keypack, adj_k_half_list[k - 1],
        non_core_ct, core_mask_ct, k, N)

    # Forward
    border_labels_ct = _ensure_label_level(engine, border_labels_ct, keypack,
                                           tag=f"border k={k} fwd")
    s_label_k   = fhe_circular_shift(engine, core_labels_ct, k, N, keypack)
    candidate_k = engine.multiply(adjm_fwd, s_label_k, relin_key)
    border_labels_ct = fhe_max(engine, border_labels_ct, candidate_k,
                               N, keypack, label_scale=label_scale,
                               secret_key=secret_key)

    # Backward
    if adjm_bwd is not None:
        border_labels_ct = _ensure_label_level(engine, border_labels_ct, keypack,
                                               tag=f"border k={k} bwd")
        s_label_Nk   = fhe_circular_shift(engine, core_labels_ct, N - k, N, keypack)
        candidate_Nk = engine.multiply(adjm_bwd, s_label_Nk, relin_key)
        border_labels_ct = fhe_max(engine, border_labels_ct, candidate_Nk,
                                   N, keypack, label_scale=label_scale,
                                   secret_key=secret_key)

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
    n_rounds: int = 1,
    enc_init_labels=None,        # ★ 신규: 클라이언트 초기 라벨 (None이면 서버 1..N)
    init_label_max=None,         # ★ 신규: 초기 라벨 최대값 (label_scale 조정)
) -> Ciphertext:
    """KD-tree ordering + dense stride k=1..k_max 라벨 전파.

    Forward + backward 2 pass × Core-Core(n_rounds 반복), Core→Border 2 phase.

    ★ [Phase1] 라벨 노이즈 회계:
      라벨이 받는 표준 bootstrap = _ensure_label_level 발동 횟수 (종료 시 출력)
      + 최종 add 후 1회. 이전 구조의 max당 3~4회 무조건 주입은 제거됨.
    """
    global _label_refresh_count
    _label_refresh_count = 0

    N     = num_points
    k_max = min(k_max, N // 2)
    T_kmax = k_max * (k_max + 1) // 2

    _eff_max = float(N) if init_label_max is None else float(init_label_max)
    label_scale = _SCALE_INSET * _eff_max      # ★ [Phase1-④] 인셋 (init_label_max로 sgn포화 방지)
    slot_count  = engine.slot_count

    print(f"\n[KD-LP] ══════════════════════════════════════════")
    print(f"[KD-LP] KD-tree dense stride 전파 (k=1..{k_max})  [Phase1 재구성]")
    print(f"[KD-LP] N={N}, k_max={k_max}, n_rounds={n_rounds}, "
          f"label_scale={label_scale:.1f} (인셋 {_SCALE_INSET})")
    print(f"[KD-LP] T({k_max})={T_kmax}  {'✓ ≥ N' if T_kmax >= N else '⚠ < N'}")
    fhe_max_cnt = (2 * n_rounds + 2) * k_max * 2
    print(f"[KD-LP] fhe_max: {fhe_max_cnt}회 "
          f"(Phase1 {2*n_rounds*k_max*2} + Phase2 {2*k_max*2})")
    print(f"[KD-LP] 라벨 레벨 가드 임계: {_LABEL_MIN_LEVEL} (budget=10 가정)")
    print(f"[KD-LP] ══════════════════════════════════════════\n")

    # ── 초기화 ─────────────────────────────────────────────────
    # core_mask 정밀도 복원 (작업 A): refresh로 레벨 확보 후 bit_cleaning.
    core_mask_ct = _refresh(engine, core_ct, keypack)
    core_mask_ct = bit_cleaning(engine, core_mask_ct, keypack,
                                n_iters=1, slot_count=slot_count,
                                ensure_output_level=False)
    non_core_ct  = engine.subtract(engine.encode([1.0] * slot_count), core_mask_ct)
    non_core_ct  = _refresh(engine, non_core_ct, keypack)
    non_core_ct  = bit_cleaning(engine, non_core_ct, keypack,
                                n_iters=1, slot_count=slot_count,
                                ensure_output_level=False)

    if enc_init_labels is None:
        id_enc = engine.encode(
            [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
        core_labels_ct = _refresh(engine,
            engine.multiply(core_mask_ct, id_enc), keypack)
    else:
        # ★ 클라이언트 제공 초기 라벨 (암호문) × core_mask
        #   ct×ct 곱이므로 relinearization_key 필수 (3-poly→2-poly).
        core_labels_ct = _refresh(engine,
            engine.multiply(core_mask_ct, enc_init_labels,
                            keypack.relinearization_key), keypack)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N)

    # ── ★ [Phase1-②] adjm 사전계산 (Core-Core용, 라운드 불변) ──
    adjm_cache = None
    if _HOIST_ADJ_MASKS:
        adjm_cache = _build_adjm_cache(
            engine, keypack, adj_k_half_list, core_mask_ct,
            core_mask_ct, k_max, N, tag="Core-Core",
            max_cached=_MAX_CACHED_ADJM)

    passes = [
        ("forward",  list(range(1, k_max + 1))),
        ("backward", list(range(k_max, 0, -1))),
    ]

    # ── Phase 1: Core-Core (n_rounds 반복) ─────────────────────
    for rnd in range(n_rounds):
        for pass_idx, (pass_name, k_order) in enumerate(passes):
            print(f"[KD-LP] Phase1: Core-Core Round{rnd+1}/{n_rounds} "
                  f"Pass{pass_idx+1}/2 ({pass_name})")
            for k in k_order:
                core_labels_ct = _propagate_one_stride_core(
                    engine, keypack, adjm_cache, adj_k_half_list,
                    core_labels_ct, core_mask_ct,
                    k, N, label_scale, secret_key=secret_key,
                )
            print(f"[KD-LP]   (라벨 refresh 누계: {_label_refresh_count})")
            _dbg(engine, secret_key, core_labels_ct,
                 f"P1-R{rnd+1}-{pass_name} 완료", N)

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
            )
        print(f"[KD-LP]   (라벨 refresh 누계: {_label_refresh_count})")
        _dbg(engine, secret_key, border_labels_ct, f"P2-{pass_name} 완료", N)

    final_ct = _refresh(engine,
        engine.add(core_labels_ct, border_labels_ct), keypack)
    print(f"[KD-LP] 라벨 lineage 표준 bootstrap 총계: {_label_refresh_count} "
          f"(+ 최종 1회)  ← 이전 구조 추정치 {fhe_max_cnt * 3}~{fhe_max_cnt * 4}회")
    _dbg(engine, secret_key, final_ct, f"최종 [0,{N}]", N)
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
    """ALL strides sweep 기반 라벨 전파 (fallback). [Phase1 동일 재구성 적용]"""
    global _label_refresh_count
    _label_refresh_count = 0

    N = num_points
    if num_sweeps is None:
        num_sweeps = math.ceil(math.log2(N))

    N_half      = N // 2
    label_scale = _SCALE_INSET * float(N)      # ★ [Phase1-④]
    slot_count  = engine.slot_count

    print(f"[LP-sweep] N={N}, N_half={N_half}, num_sweeps={num_sweeps}, "
          f"label_scale={label_scale:.1f}  [Phase1 재구성]")

    core_mask_ct = _refresh(engine, core_ct, keypack)
    core_mask_ct = bit_cleaning(engine, core_mask_ct, keypack,
                                n_iters=1, slot_count=slot_count,
                                ensure_output_level=False)
    non_core_ct  = engine.subtract(engine.encode([1.0] * slot_count), core_mask_ct)
    non_core_ct  = _refresh(engine, non_core_ct, keypack)
    non_core_ct  = bit_cleaning(engine, non_core_ct, keypack,
                                n_iters=1, slot_count=slot_count,
                                ensure_output_level=False)

    if enc_init_labels is None:
        id_enc = engine.encode(
            [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
        core_labels_ct = _refresh(engine,
            engine.multiply(core_mask_ct, id_enc), keypack)
    else:
        # ★ 클라이언트 제공 초기 라벨 (암호문) × core_mask
        #   ct×ct 곱이므로 relinearization_key 필수 (3-poly→2-poly).
        core_labels_ct = _refresh(engine,
            engine.multiply(core_mask_ct, enc_init_labels,
                            keypack.relinearization_key), keypack)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N)

    adjm_cache = None
    if _HOIST_ADJ_MASKS:
        adjm_cache = _build_adjm_cache(
            engine, keypack, adj_k_half_list, core_mask_ct,
            core_mask_ct, N_half, N, tag="sweep Core-Core",
            max_cached=_MAX_CACHED_ADJM)

    for sweep in range(num_sweeps):
        print(f"[LP-sweep] P1 Sweep {sweep+1}/{num_sweeps}")
        for k in range(1, N_half + 1):
            core_labels_ct = _propagate_one_stride_core(
                engine, keypack, adjm_cache, adj_k_half_list,
                core_labels_ct, core_mask_ct,
                k, N, label_scale, secret_key=secret_key,
            )
        print(f"[LP-sweep]   (라벨 refresh 누계: {_label_refresh_count})")
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
        print(f"[LP-sweep]   (라벨 refresh 누계: {_label_refresh_count})")
        _dbg(engine, secret_key, border_labels_ct, f"P2 Sweep{sweep+1} 완료", N)

    final_ct = _refresh(engine,
        engine.add(core_labels_ct, border_labels_ct), keypack)
    print(f"[LP-sweep] 라벨 lineage 표준 bootstrap 총계: {_label_refresh_count} (+ 최종 1회)")
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
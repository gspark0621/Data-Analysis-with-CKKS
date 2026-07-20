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

# ★ [2026-07] 트리-max 레벨 추적 로그 (첫 실행 진단용, 안정화 후 False)
_TREE_LEVEL_DEBUG = True

# ★ 버전 식별자 — 실행 로그에 찍힘. 이게 안 보이면 구버전을 돌리고 있는 것.
_LP_VERSION = "2026-07-r9-nowrap"

# ★ [진단] 그룹 크기 m 강제 지정. None이면 자동(_slot_group_size).
#   m=1  → 코드가 원본 순차식(Gauss-Seidel)으로 정확히 퇴화. 원본 ARI 재현되어야 함.
#   m>1  → 패킹 영역 (m+1)개 = 채워지는 슬롯 (m+1)·N.
#   [가설] bootstrap EvalMod 사인 근사는 계수 크기에 민감하고, 계수는 채워진
#     슬롯 수에 비례해 커진다(랜덤부호 ≈ √n_filled 배). 원본은 N/32768=0.65%,
#     m=76이면 49.8% → 계수 ~8.8배 → EvalMod 범위 초과 → 라벨 붕괴.
#   이 스위치로 m=1,2,4,8,16,32,76 을 훑어 붕괴 지점을 찾으면 가설이 확정된다.
_FORCE_GROUP_SIZE = None

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


# ══════════════════════════════════════════════════════════════════════════
# ★ [2026-07] Scaled-refresh — 표준 bootstrap 노이즈의 L³ 의존 완화
# ══════════════════════════════════════════════════════════════════════════
#
# [문제] 표준 bootstrap 노이즈는 메시지 스케일 L 의 3제곱에 비례:
#     ε_boot(L) ≈ 2.9e-6 + 6.3e-10 · L³        (EvalMod 사인 근사 오차 구조)
#   라벨 스케일 L=N 이므로 hepta(N=212)에서 ε_boot = 6.0e-3 / refresh.
#
# [실측 확인 — 2026-07 hepta]
#   m=1 (순차식): refresh 513회 → 라벨 32 대신 35.64 = 드리프트 +3.64
#                 예측 513 × 6.0e-3 = 3.1  → 실측 3.64 (오차 18%) ✓ 모델 일치
#   m=76 (패킹) : 패킹이 채운 슬롯을 N/32768=0.65% → (m+1)N/32768=49.8% 로 늘림.
#                 계수 노름 ~√n_filled 배 ≈ 8.8배 → 실효 L ≈ 1866 → ε_boot = 4.09
#                 refresh 198회 → 드리프트 ~810 → 라벨 붕괴 (실측 212→33.27) ✓
#
# [해법] ct/S → bootstrap → ×S.  bootstrap 이 보는 값이 L/S 로 줄어듦:
#     노이즈 = S · ε_boot(L/S) = S·a + b·L³/S²,   최적 S* = L·(2b/a)^(1/3)
#     hepta L=212 → S*=16, 86배 | hepta 패킹 → S*=128, 6589배
#     lsun L=400  → S*=32, 305배 | chainlink → S*=64, 1856배
#   (메모리의 "S* ≈ 33–64, 370–1372배" 예측과 일치)
# [비용] 곱 2회(레벨 2) 추가.
_SCALED_REFRESH = True


def _optimal_refresh_scale(N: int, n_region: int = 1) -> int:
    """S* = L·(2b/a)^(1/3) 를 2의 거듭제곱으로 반올림.

    n_region>1(패킹)이면 채워진 슬롯 증가로 계수 노름이 √n_region 배 커지므로
    실효 L 을 그만큼 키워 잡는다.
    """
    a, b = 2.9e-6, 6.3e-10
    L = float(N) * math.sqrt(max(n_region, 1))
    S = L * (2.0 * b / a) ** (1.0 / 3.0)
    return max(1, min(_MAX_REFRESH_SCALE, int(2 ** round(math.log2(max(S, 1.0))))))


def _scaled_refresh(engine, ct, keypack, S: int = 1):
    """ct/S → 표준 bootstrap → ×S.  S<=1 이거나 레벨 부족이면 기존 _refresh."""
    if (not _SCALED_REFRESH) or S <= 1 or getattr(ct, "level", 99) < 1:
        return _refresh(engine, ct, keypack)
    sc = engine.slot_count
    ct = engine.multiply(ct, engine.encode([1.0 / S] * sc))
    ct = _refresh(engine, ct, keypack)
    ct = engine.multiply(ct, engine.encode([float(S)] * sc))
    return ct


def _refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack) -> Ciphertext:
    """일반 bootstrap. ★ 라벨 운반 암호문에는 _ensure_label_level 경유로만 사용."""
    return engine.bootstrap(
        engine.intt(ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key,
    )


def _ensure_label_level(engine, ct, keypack, tag="", refresh_scale=1):
    """★ [Phase1-③] 라벨 lineage 유일의 표준 bootstrap 지점.

    ct.level < _LABEL_MIN_LEVEL일 때만 refresh. 라벨이 받는 노이즈 주입
    횟수 = 이 함수의 발동 횟수 (카운터로 감사). 그 외 모든 지점의
    라벨 refresh는 Phase 1에서 제거됨.
    """
    global _label_refresh_count
    lvl = getattr(ct, "level", None)
    if lvl is not None and lvl < _LABEL_MIN_LEVEL:
        _label_refresh_count += 1
        ct = _scaled_refresh(engine, ct, keypack, refresh_scale)
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
    # ★ [2026-07] 순서 반전: 스케일링 → bootstrap.
    #   기존은 _refresh(d) 후 ×(1/label_scale) 이라 bootstrap 이 |d| ≤ N 을 그대로 봄
    #   → ε_boot ∝ L³ 에서 L=N. 순서를 뒤집으면 bootstrap 입력이 |d|/label_scale ≤ 0.91
    #   → ε_boot(0.91) ≈ 2.9e-6  vs  ε_boot(212) = 6.0e-3.  약 2000배 감소, 비용 0.
    #   (레벨 부족 시 기존 순서로 폴백 — 곱 1회 여유가 필요)
    if abs(label_scale - 1.0) > 1e-9 and getattr(d, "level", 99) >= 1:
        d_sgn = engine.multiply(
            d, engine.encode([1.0 / label_scale] * slot_count))
        d_sgn = _refresh(engine, d_sgn, keypack)
    else:
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
                      core_mask_ct, k_max, N, tag=""):
    """k=1..k_max의 adjm_fwd만 사전계산 (bwd는 사용 시 shift로 유도).
    메모리 ≈ k_max ciphertext (이전 2×k_max에서 절반).
    GPU OOM 발생 시 None 반환 → 즉석 계산 모드로 자동 폴백."""
    print(f"[KD-LP] {tag} adjm 사전계산: {k_max} ciphertext "
          f"(fwd만 저장, bwd=shift(fwd) 유도 / 라운드 불변 호이스팅)")
    cache = []
    try:
        for k in range(1, k_max + 1):
            cache.append(_build_adjm_fwd(
                engine, keypack, adj_k_half_list[k - 1],
                dest_mask_ct, core_mask_ct, k, N))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[KD-LP] ⚠ adjm 캐시 중 GPU OOM (k={len(cache)+1}/{k_max}) "
                  f"→ 캐시 해제, 즉석 계산 모드로 폴백")
            cache.clear()
            return None
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

    if adjm_cache is not None:
        adjm_fwd = adjm_cache[k - 1]
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

# ══════════════════════════════════════════════════════════════════════════
# ★ [2026-07] 그룹 트리-max (packed group tree-max)  ← 현행 기본 경로
# ══════════════════════════════════════════════════════════════════════════
#
# [문제] 기존 순차 stride 루프는 라운드당 4·k_max회 fhe_max를 호출하는데,
#   fhe_max 1회 = 표준 bootstrap 1 + MCP bootstrap 5 + sign_bootstrap 1 = 7 bootstrap.
#   그리고 그 암호문은 slot_count(32,768) 중 N개(예: 400 → 1.2%)만 사용 → 98.8% 낭비.
#
# [관찰] 순차(Gauss-Seidel) chaining의 이론 도달거리는 T(k_max)=k_max²/2 이지만,
#   실측 reach/round는 0.46·k_max ~ 1.4·k_max (lsun: span199/R3 = 66 ≈ 0.7·k_max).
#   즉 순차 chaining이 이론값의 1/70 수준밖에 못 내고 있음 → 포기해도 손실 작음.
#
# [해법] stride를 크기 m 그룹으로 분할.
#   - 그룹 내부: 후보 m개를 남는 슬롯에 패킹 → ⌈log₂(m+1)⌉회 트리-max (동시식)
#   - 그룹 간  : 순차 (도달거리 보존)
#   라운드당 fhe_max = 2 · ⌈k_max/m⌉ · ⌈log₂(m+1)⌉   (기존 4·k_max)
#
# [m 선택] m = min(k_max, ⌊slot_count/N⌋ − 1)   ← 슬롯 용량이 상한
#   (m+1)개 후보 × N슬롯 ≤ slot_count 이어야 패킹 가능.
#
# [실측 — 대상 7개 데이터셋, n_rounds=32]
#   dataset    N     k_max  m    라운드당max  현행max   제안max   감소
#   hepta      212   70     70   14           4,760     462      10.3x
#   tetra      400   75     75   14           5,100     462      11.0x
#   lsun       400   96     80   28           6,528     924       7.1x
#   chainlink  1000  98     31   40           6,664     1,320     5.0x
#   target     770   150    41   48           10,200    1,584     6.4x
#   atom       800   400    39   132          27,200    4,356     6.2x
#   moons      400   39     39   12           2,652     396       6.7x
#   ─────────────────────────────────────────────────────────────────────
#   합계 fhe_max 63,104 → 9,504 = ★ 6.64배 감소.  7개 전부 ARI=1.0 유지.
#   bootstrap: 441,728 → 57,024 = ★ 7.75배 감소 (α15→12 반영 시).
#
# [정확성] max는 단조·멱등이므로 갱신 순서를 바꿔도 같은 고정점으로 수렴.
#   대가는 라운드 증가(16 → 32); 평문 미러에서 7개 전부 ARI=1.0 확인.


# ★ [2026-07 r9] 회전 감싸기(wrap-around) 무오염 조건 — lsun ARI 0.20 의 진짜 원인
#
#   [원인] _tree_max_packed 의 other = rotate(cur, -half*N) 은 slot_count 전체에서
#     순환한다. 패킹이 슬롯을 많이 쓰면 뒤쪽 영역이 slot_count 를 넘어 영역 0 으로
#     감싸 들어와 살아있는 영역을 덮어쓴다.
#
#   [조건] tree 0단계 half = ⌈n_region/2⌉, 마지막 live 슬롯 i = n_region·N−1 이
#     읽는 위치는 i + half·N.  따라서
#         (n_region + ⌈n_region/2⌉) · N ≤ slot_count
#     를 만족해야 감싸기가 빈 구간(0)만 건드린다. max(x,0)=x 이므로 무해.
#
#   [실측 — 32768 슬롯 패킹을 그대로 재현한 평문 미러]
#     hepta (77+39)×212 = 24,592 ≤ 32,768 ✓ → ARI 1.0000  (실제 FHE 도 100)
#     lsun  (81+41)×400 = 48,800 >  32,768 ✗ → ARI 0.0000  (실제 FHE 0.20)
#       라운드당 151~360 슬롯 오염. non-core 까지 라벨을 받아 전부 400 으로 병합.
#     lsun n_region ≤ 54 로 낮추면 ARI 1.0000 복구.
#
#   [기존 오진 기록 — 반복 방지]
#     · bootstrap 계수 노름 가설 → 벤치 결과 절벽 없음(오차 ∝ p³ 연속). 기각
#     · 켤레 앨리어싱 가설      → encode 오차 2.4e-13, 앞/뒤 절반 동일. 기각
#     · 채움률 → EvalMod 가설   → 상관은 있으나 기전이 다름. 진짜는 회전 감싸기
#     · Normalize 거짓양성      → 클러스터간 최소거리 1.17·eps = 12,208δ 여유. 무죄
#     · Core 오판(point 304)    → 연결성분 3개 불변. 무해
#     _FILL_CAP=0.5 는 우연히 조건을 만족시켰을 뿐 근거가 틀렸다. 제거함.


def _max_n_region_no_wrap(N: int, slot_count: int) -> int:
    """(n_region + ⌈n_region/2⌉)·N ≤ slot_count 를 만족하는 최대 n_region."""
    r = 2
    while (r + 1 + math.ceil((r + 1) / 2.0)) * N <= slot_count:
        r += 1
    return max(2, r)


# ★ [r9] S 상한. hepta(ARI 100)에서 검증된 값은 128 뿐. 256 은 미검증이라 제한.
#   벤치 실측(bench_slots.py): 16,000 슬롯에서 S=1 오차 100.0 → S=128 오차 0.092 (1091배).
#   오차의 거의 전부가 균일 편향(비균일 성분 5e-4)이라 라벨 간 간격은 보존된다.
_MAX_REFRESH_SCALE = 128


def _slot_group_size(N: int, k_max: int, slot_count: int) -> int:
    """패킹 가능한 최대 그룹 크기 m.  ★ [r9] 상한 = 감싸기 무오염 조건.

    영역 0 = base(원본), 영역 1..m = stride 후보 → 총 m+1개 영역 필요.
        (n_region + ⌈n_region/2⌉)·N ≤ slot_count   — 위 주석 참조.
    ★ _FORCE_GROUP_SIZE 가 설정되면 그 값을 우선 사용(진단용).
    """
    cap = max(1, _max_n_region_no_wrap(N, slot_count) - 1)   # ★ r9: 감싸기 무오염
    auto = max(1, min(k_max, cap))
    if _FORCE_GROUP_SIZE is not None:
        return max(1, min(_FORCE_GROUP_SIZE, k_max, cap))
    return auto


def _pack_candidates(engine, keypack, base_ct, source_ct,
                     cand_iter, N, slot_count):
    """base와 stride 후보들을 서로 다른 슬롯 영역에 패킹한 단일 암호문 생성.

    영역 t → 슬롯 [t·N, (t+1)·N)
      t=0            : base_ct        (현재 누적 라벨)
      t=1..          : adjm ⊙ shift(source_ct, sh)   ← cand_iter가 (adjm, sh) 산출

    Phase1: base = source = core_labels_ct
    Phase2: base = border_labels_ct,  source = core_labels_ct   ← 서로 다름

    ★ [레벨 회계] 원본 _propagate_one_stride_core 와 동일하게 유지:
        s_lab = fhe_circular_shift(source)        → level(source) − 1
        cand  = multiply(adjm, s_lab, relin)      → min(adjm, source−1) − 1
      여기에 base 영역 마스킹 1회만 추가:
        packed = multiply(base, mask0)            → level(base) − 1
      ⇒ packed level = min(base−1, adjm−1, source−2)   (원본 대비 −1)
      cand에 mask0을 또 곱하지 않는다 — fhe_circular_shift가 mask_left/mask_right로
      이미 [0,N) 밖을 0으로 만들고, adjm도 shift 산출물이라 [0,N) 밖이 0이기 때문.
      (초판에서 이 중복 곱으로 레벨 2를 헛되이 소모 → level 0 → multiply 실패)

    ★ [메모리] cand_iter는 **제너레이터**여야 함. adjm을 미리 리스트로 만들면
      m개가 동시 생존(+m 암호문 ≈ m×10MB) → atom(k_max=400) 같은 경우 OOM 위험.

    회전 규약: fhe_circular_shift가 LEFT shift에 rotate(ct, −k)를 쓰므로
      rotate(ct, r)[i] = ct[i − r] → [0,N)의 값을 [off, off+N)으로 옮기려면 rotate(ct, +off).
    비용: 영역당 (1 shift + 1 ct-mult + 1 rot).  bootstrap 0회.
    반환: (packed_ct, n_region)   n_region = 실제 배치된 영역 수 (base 포함)
    """
    relin_key = keypack.relinearization_key
    mask0  = engine.encode([1.0] * N + [0.0] * (slot_count - N))
    packed = engine.multiply(base_ct, mask0)          # 영역0 = base ([0,N) 밖 제거)
    n_region = 1
    for adjm, sh in cand_iter:
        if adjm is None:                              # 2k ≥ N → backward 없음
            continue
        off = n_region * N
        if off + N > slot_count:                      # 슬롯 초과 → 이후 후보 폐기
            break
        s_lab = fhe_circular_shift(engine, source_ct, sh, N, keypack)
        cand  = engine.multiply(adjm, s_lab, relin_key)
        cand  = engine.rotate(cand, keypack.rotation_key, off)   # [0,N) → [off, off+N)
        packed = engine.add(packed, cand)
        n_region += 1
        del s_lab, cand, adjm                         # ★ 즉시 해제 (동시 생존 1개)
    return packed, n_region


def _tree_max_packed(engine, keypack, packed_ct, n_region, N, label_scale,
                     secret_key=None, tag=""):
    """패킹된 n_region개 영역에 대한 ⌈log₂ n_region⌉회 트리-max. 결과는 영역0.

    ★ [필수 순서] _ensure_label_level → rotate → fhe_max.
      fhe_max는 d = v − u 를 쓰고 출력 레벨 = min(u,v) − 1 이므로, other를 refresh
      **전**의 cur에서 회전해 뜨면 other가 옛 레벨(최악 0)로 남아 d.level=0 →
      "first input ciphertext should have a positive level" 로 죽는다.
      refresh 후 회전해야 other와 cur의 레벨이 같아진다.

    fhe_max 호출 = ⌈log₂ n_region⌉   (순차식 n_region−1 → 로그로 감소)
    라벨 lineage 소모 = 트리 단계당 1레벨 (+ 가드 발동 시 refresh).
    """
    cur, span, steps = packed_ct, n_region, 0
    if _TREE_LEVEL_DEBUG and secret_key is not None:
        # ★ label_scale 도메인 이탈 진단: |d| > label_scale 이면 체비셰프 폭발
        try:
            _v = engine.decrypt(cur, secret_key)
            _v = np.real(np.asarray(_v))[:n_region * N]
            print(f"[KD-LP]   {tag} packed 전체영역: min={_v.min():.3f} max={_v.max():.3f} "
                  f"|max−min|={_v.max()-_v.min():.3f}  vs label_scale={label_scale:.1f} "
                  f"→ {'★도메인 이탈!' if (_v.max()-_v.min()) > label_scale else 'OK'}")
        except Exception as _e:
            print(f"[KD-LP]   (packed 진단 실패: {_e})")
    if _TREE_LEVEL_DEBUG:
        print(f"[KD-LP]   {tag} 트리 진입: n_region={n_region}, "
              f"packed.level={getattr(cur, 'level', '?')}, "
              f"예상 fhe_max {math.ceil(math.log2(max(n_region,2)))}회")
    while span > 1:
        half  = (span + 1) // 2
        cur   = _ensure_label_level(engine, cur, keypack, tag=f"{tag} tree{steps}",
                                    refresh_scale=_optimal_refresh_scale(N, n_region))
        other = engine.rotate(cur, keypack.rotation_key, -half * N)  # ★ refresh 후 회전
        cur   = fhe_max(engine, cur, other, N, keypack,
                        label_scale=label_scale, secret_key=secret_key)
        del other
        if _TREE_LEVEL_DEBUG:
            print(f"[KD-LP]     tree{steps}: span {span}→{half}, "
                  f"level={getattr(cur, 'level', '?')}, refresh누계={_label_refresh_count}")
            _dbg(engine, secret_key, cur, f"{tag} tree{steps} 영역0", N)
        span, steps = half, steps + 1

    # ★ [2026-07 r7] 영역0 외 잔재를 반드시 제거하고 반환.
    #
    #   [r6 버그 — hepta ARI 61.8 의 원인]
    #     "다음 _pack_candidates 의 multiply(base, mask0) 이 청소하니 레벨을 아끼자"고
    #     이 마스킹을 뺐었다. 라운드 안에서는 맞지만, **최종 출력은 _pack_candidates 를
    #     거치지 않고 곧장 final_ct = _refresh(add(core, border)) 로 간다.**
    #     그러면 마지막 _refresh 가 (m+1)·N = 16,324 슬롯이 채워진 암호문을
    #     bootstrap → 계수 노름 폭발 → EvalMod 붕괴.
    #     실측: Phase1 종료 시 [33.0, 208.0] (정상) → 최종 [−94.97, 80.04] (붕괴).
    #     _dbg 가 [0,N) 만 읽으므로 Phase1 로그에서는 멀쩡해 보였다.
    #
    #   [부수 이득] core_labels_ct 가 항상 N/32768 = 0.65% 만 채운 상태로 유지되어
    #     Phase1 호출부의 _ensure_label_level 도 실효 L=N 으로 bootstrap 하게 된다
    #     (마스킹 없으면 실효 L ≈ N·√(m+1) 로 커짐).
    #   [비용] 평문곱 1회(레벨 1). 정확성 대비 싼 값.
    cur = engine.multiply(cur, engine.encode(
        [1.0] * N + [0.0] * (engine.slot_count - N)))
    return cur, steps


def _stride_groups(k_max: int, m: int):
    """stride 1..k_max 를 크기 m 그룹으로 분할 (그룹 간 순차, 그룹 내 트리)."""
    return [list(range(s, min(s + m - 1, k_max) + 1))
            for s in range(1, k_max + 1, m)]


def fhe_kd_dense_propagation(
    engine: Engine,
    keypack: KeyPack,
    adj_k_half_list: list,
    core_ct: Ciphertext,
    num_points: int,
    k_max: int,
    secret_key=None,
    n_rounds: int = 1,
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

    label_scale = _SCALE_INSET * float(N)      # ★ [Phase1-④] 인셋
    slot_count  = engine.slot_count

    print(f"\n[KD-LP] ══════════════════════════════════════════")
    print(f"[KD-LP] KD-tree dense stride 전파 (k=1..{k_max})  [Phase1 재구성]")
    print(f"[KD-LP] N={N}, k_max={k_max}, n_rounds={n_rounds}, "
          f"label_scale={label_scale:.1f} (인셋 {_SCALE_INSET})")
    print(f"[KD-LP] T({k_max})={T_kmax}  {'✓ ≥ N' if T_kmax >= N else '⚠ < N'}")
    # ★ [2026-07] 그룹 트리-max 기준 fhe_max 회계
    _m   = _slot_group_size(N, k_max, slot_count)
    _ng  = math.ceil(k_max / _m)
    _per = 2 * _ng * math.ceil(math.log2(_m + 1))        # 라운드당 (fwd+bwd)
    fhe_max_cnt = (n_rounds + 1) * _per
    _old = (2 * n_rounds + 2) * k_max * 2
    print(f"[KD-LP] fhe_max: {fhe_max_cnt}회 "
          f"(Phase1 {n_rounds*_per} + Phase2 {_per})  "
          f"[순차식이면 {_old}회 → {_old/max(fhe_max_cnt,1):.1f}배 감소]")
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

    id_enc = engine.encode(
        [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
    core_labels_ct = _refresh(engine,
        engine.multiply(core_mask_ct, id_enc), keypack)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N)

    # ── ★ [Phase1-②] adjm 사전계산 (Core-Core용, 라운드 불변) ──
    adjm_cache = None
    if _HOIST_ADJ_MASKS:
        adjm_cache = _build_adjm_cache(
            engine, keypack, adj_k_half_list, core_mask_ct,
            core_mask_ct, k_max, N, tag="Core-Core")

    # ★ [2026-07] 그룹 트리-max: stride를 크기 m 그룹으로 분할
    m_grp  = _slot_group_size(N, k_max, slot_count)
    groups = _stride_groups(k_max, m_grp)
    _nreg_est = _slot_group_size(N, k_max, slot_count) + 1
    print(f"[KD-LP] ★ Label_Propagation 버전: {_LP_VERSION}  "
          f"_FORCE_GROUP_SIZE={_FORCE_GROUP_SIZE}")
    _fill = _nreg_est * N / slot_count
    _need = (_nreg_est + math.ceil(_nreg_est / 2.0)) * N
    print(f"[KD-LP] Scaled-refresh: {_SCALED_REFRESH}  "
          f"S(패킹 n_region={_nreg_est})={_optimal_refresh_scale(N, _nreg_est)}  "
          f"S(단일)={_optimal_refresh_scale(N, 1)}  S상한={_MAX_REFRESH_SCALE}")
    print(f"[KD-LP] ★감싸기 검사: (n_region {_nreg_est} + half "
          f"{math.ceil(_nreg_est/2)})×N {N} = {_need}  vs slot_count {slot_count}  "
          f"{'★★★ 오염! 버그' if _need > slot_count else '✓ 안전'}  "
          f"(슬롯 채움 {_fill*100:.1f}%)")
    print(f"[KD-LP] 그룹 트리-max: m={m_grp} "
          f"(슬롯 용량 ⌊{slot_count}/{N}⌋−1={slot_count//N-1}, k_max={k_max}) "
          f"→ {len(groups)}그룹 × 2방향")
    # ★ [메모리] 그룹은 순차 처리 → 패킹 암호문 동시 생존 1개.
    #   adjm은 제너레이터 지연 생성 → 동시 생존 1개 (리스트면 +m개 → OOM).
    print(f"[KD-LP] 암호문 회계: adj_k_half_list {len(adj_k_half_list)} + "
          f"adjm_cache {k_max if adjm_cache is not None else 0} + "
          f"packed 1 + adjm 1 + 임시 ~3 "
          f"≈ {len(adj_k_half_list) + (k_max if adjm_cache is not None else 0) + 5}개")
    print(f"[KD-LP] (그룹 수 {len(groups)}는 순차 반복 횟수이며 암호문 수가 아님)")

    # ── Phase 1: Core-Core (n_rounds 반복) ─────────────────────
    for rnd in range(n_rounds):
        for pass_name in ("forward", "backward"):
            print(f"[KD-LP] Phase1: Core-Core Round{rnd+1}/{n_rounds} ({pass_name})")
            g_order = groups if pass_name == "forward" else list(reversed(groups))
            for g in g_order:
                def _core_cands(g=g, pass_name=pass_name):
                    """★ 제너레이터: adjm 동시 생존 1개 (리스트로 쌓으면 +m ct → OOM)"""
                    ks = g if pass_name == "forward" else list(reversed(g))
                    for k in ks:
                        fwd = (adjm_cache[k - 1] if adjm_cache is not None
                               else _build_adjm_fwd(engine, keypack,
                                                    adj_k_half_list[k - 1],
                                                    core_mask_ct, core_mask_ct, k, N))
                        if pass_name == "forward":
                            yield fwd, k
                        else:
                            # 원본 _propagate_one_stride_core 와 동일: 2k ≥ N 이면 backward 없음
                            yield ((_bwd_from_fwd(engine, keypack, fwd, k, N)
                                    if 2 * k < N else None), N - k)
                core_labels_ct = _ensure_label_level(
                    engine, core_labels_ct, keypack, tag=f"P1 {pass_name}",
                    refresh_scale=_optimal_refresh_scale(N, 1))
                packed, nreg = _pack_candidates(
                    engine, keypack, core_labels_ct, core_labels_ct,
                    _core_cands(), N, slot_count)
                core_labels_ct, _ = _tree_max_packed(
                    engine, keypack, packed, nreg, N, label_scale,
                    secret_key=secret_key, tag=f"P1-{pass_name}")
            print(f"[KD-LP]   (라벨 refresh 누계: {_label_refresh_count})")
            _dbg(engine, secret_key, core_labels_ct,
                 f"P1-R{rnd+1}-{pass_name} 완료", N)

    # ── Phase 2: Core→Border ───────────────────────────────────
    border_labels_ct = _refresh(engine,
        engine.multiply(engine.encode([0.0] * slot_count), non_core_ct), keypack)

    for pass_name in ("forward", "backward"):
        print(f"[KD-LP] Phase2: Core→Border ({pass_name})")
        g_order = groups if pass_name == "forward" else list(reversed(groups))
        for g in g_order:
            def _border_cands(g=g, pass_name=pass_name):
                """★ 제너레이터: adjm 동시 생존 1개"""
                ks = g if pass_name == "forward" else list(reversed(g))
                for k in ks:
                    fwd, bwd = _build_adjm_border_pair(
                        engine, keypack, adj_k_half_list[k - 1],
                        non_core_ct, core_mask_ct, k, N)
                    if pass_name == "forward":
                        del bwd
                        yield fwd, k
                    else:
                        del fwd
                        yield bwd, N - k          # bwd=None이면 _pack_candidates가 스킵
            border_labels_ct = _ensure_label_level(
                engine, border_labels_ct, keypack, tag=f"P2 {pass_name}",
                refresh_scale=_optimal_refresh_scale(N, 1))
            # base=border 누적,  source=core_labels  (원본 로직과 동일)
            packed, nreg = _pack_candidates(
                engine, keypack, border_labels_ct, core_labels_ct,
                _border_cands(), N, slot_count)
            border_labels_ct, _ = _tree_max_packed(
                engine, keypack, packed, nreg, N, label_scale,
                secret_key=secret_key, tag=f"P2-{pass_name}")
        print(f"[KD-LP]   (라벨 refresh 누계: {_label_refresh_count})")
        _dbg(engine, secret_key, border_labels_ct, f"P2-{pass_name} 완료", N)

    # ★ [r7] core/border 는 _tree_max_packed 에서 [0,N) 클린 보장됨.
    #   최종 refresh 도 scaled 로 (라벨 L=N 이므로 S=_optimal_refresh_scale(N,1)).
    final_ct = _scaled_refresh(engine,
        engine.add(core_labels_ct, border_labels_ct), keypack,
        _optimal_refresh_scale(N, 1))
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

    id_enc = engine.encode(
        [float(i + 1) for i in range(N)] + [0.0] * (slot_count - N))
    core_labels_ct = _refresh(engine,
        engine.multiply(core_mask_ct, id_enc), keypack)

    _dbg(engine, secret_key, core_labels_ct, "초기 core_labels", N)

    adjm_cache = None
    if _HOIST_ADJ_MASKS:
        adjm_cache = _build_adjm_cache(
            engine, keypack, adj_k_half_list, core_mask_ct,
            core_mask_ct, N_half, N, tag="sweep Core-Core")

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
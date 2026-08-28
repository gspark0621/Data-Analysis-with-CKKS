# soft_tree/tree_core.py
#
# FHE oblivious soft decision tree — 서버측 CKKS 코어.
#   forward(추론=시나리오 B)  : 구체 구현.
#   backward                  : block-순서 · 나눗셈 없는 정답 구현
#                               (평문 유한차분 검증 완료, 오차 ~1e-10).
#   train_step                : forward + backward + momentum + bootstrap 스케줄.
#
# ── 인코딩 규약 (DBSCAN 동일: 컬럼 단위, 샘플 SIMD 병렬) ────────────────────
#   X[f] : feature f 값들을 슬롯 [0..N-1] 에 담은 암호문 (F 개).
#   모델(시나리오 B): W[i][f], b[i], v[ℓ] 모두 broadcast 암호문(스칼라).
#
# ── backward 유도 (검증됨) ─────────────────────────────────────────────────
#   forward: prob_{i+1} = concat([prob_i·(1-g_i), prob_i·g_i])   (block 순서)
#   역전파(i 를 역순):
#     left, right = grad_prob 의 앞/뒤 절반
#     grad_g_i    = Σ_k (right[k]-left[k])·pb_i[k]        ← 나눗셈 없음
#     grad_z_i    = grad_g_i · σ'(z_i) · zscale,  σ'(z)=g(1-g)
#     grad_W[i][f]= Σ_slots grad_z_i · X[f]
#     grad_b[i]   = Σ_slots grad_z_i
#     grad_prob   = left·(1-g_i) + right·g_i              (다음 레벨로)
#   grad_v[ℓ]     = Σ_slots dL_df · prob[ℓ]
#
# ── momentum (FHE 친화) : vel=μ·vel-lr·grad ; θ+=vel. [pt*ct]+[add], 나눗셈/sqrt 없음.
# ── 레벨 예산(d=4, 추정): forward~7, 1스텝~17 → refresh(small-key bootstrap) 삽입.

from __future__ import annotations
import gc
_MEM_FREE = True   # ★ DBSCAN 식 적극적 메모리 해제 (OOM 방지). False 면 해제 안 함.
from typing import List
import numpy as np
try:
    from desilofhe import Engine, Ciphertext
except Exception:
    Engine = Ciphertext = object   # 평문 참조/문법검사용 폴백
from Soft_Tree.desilo.util.keypack import KeyPack, refresh
from Soft_Tree.desilo.gate_eval import gate_forward, gate_derivative_poly


class EncSoftTree:
    def __init__(self, W, b, v, cheb_coeffs, dcheb_coeffs, band, zscale):
        self.W = W                    # W[i][f] (depth × F) broadcast 암호문
        self.b = b                    # b[i]
        self.v = v                    # v[ℓ]  (2^depth)
        self.cheb_coeffs = cheb_coeffs; self.dcheb_coeffs = dcheb_coeffs
        self.band = float(band); self.zscale = float(zscale)
        self.depth = len(W); self.F = len(W[0])


# ── 헬퍼 ──
def _c(engine, s):       return engine.encode([float(s)] * engine.slot_count)
def _zeros(engine, kp):  return engine.encrypt([0.0] * engine.slot_count, kp.public_key)


def _ensure_level(engine, ct, keypack, need, tag="", verbose=False):
    """
    레벨 확인 후 refresh (방향 B). ct.level 이 need 미만이면 값 보존 bootstrap.
    forward/backward 중간의 레벨 소진 지점에 삽입 — small bootstrap(회복 낮음) 대응.
    """
    try:
        lvl = ct.level
    except Exception:
        return ct
    if lvl < need:
        if verbose:
            print(f"    [BOOT-mid] {tag}: level={lvl} < {need} → refresh")
        return refresh(engine, ct, keypack)
    return ct


def _slot_sum(engine, ct, keypack, n=None):
    """
    전체 슬롯 합을 모든 슬롯에 복제 (rotate-and-add, log2 회).
    데이터는 슬롯 [0..N-1] 에 있고 나머지 0 이므로 결과 = N개 샘플 합.
    ★ engine.sum 이 슬롯 전체 합을 안 하므로(검증), 명시적 rotate-and-sum 사용.
      (DBSCAN 의 fhe_circular_shift 기반 합과 동일 방식)
    ★ engine.rotate(ct, rotation_key, delta) — Desilo 확인된 시그니처.
    n 을 주면 그 크기의 다음 2의 거듭제곱까지만 회전(비용 절감). 기본은 slot_count.
    """
    sc = engine.slot_count
    limit = sc if n is None else 1 << (int(np.ceil(np.log2(max(2, n)))))
    out = ct
    step = 1
    while step < limit:
        out = engine.add(out, engine.rotate(out, keypack.rotation_key, step))
        step *= 2
    return out


# ============================================================================
# ### CKKS-CORE : forward (추론). pbs(레벨 진입 전 prob) 캐시.
# ============================================================================
def forward(engine, tree, X, keypack, return_cache=False, refresh_below=None,
            refresh_params_below=None, verbose=False):
    rlk = keypack.relinearization_key
    d, F = tree.depth, tree.F
    one_pt = _c(engine, 1.0)
    zscale_pt = _c(engine, tree.zscale)

    # 게이트 g_i = σ(zscale·(Σ_f W[i][f]·X[f] + b[i])),  z_scaled 캐시(도함수용)
    G = []; Zc = []
    for i in range(d):
        # ★ 예측 모드(refresh_params_below): 파라미터를 그때그때 refresh→사용→즉시 해제
        #   (140개를 한꺼번에 refresh 해 동시 생존시키던 _refresh_tree_params 의 OOM 회피)
        def _w(f):
            if refresh_params_below is not None:
                return _ensure_level(engine, tree.W[i][f], keypack, refresh_params_below,
                                     tag=f'pW[{i}][{f}]', verbose=verbose)
            return tree.W[i][f]
        w0 = _w(0)
        acc = engine.multiply(w0, X[0], rlk)                    # [ct*ct]
        if refresh_params_below is not None and _MEM_FREE: del w0
        for f in range(1, F):
            wf = _w(f)
            term = engine.multiply(wf, X[f], rlk)
            acc = engine.add(acc, term)
            if refresh_params_below is not None and _MEM_FREE: del wf, term
        b_i = (_ensure_level(engine, tree.b[i], keypack, refresh_params_below,
                             tag=f'pb[{i}]', verbose=verbose)
               if refresh_params_below is not None else tree.b[i])
        z = engine.add(acc, b_i)
        z = engine.multiply(z, zscale_pt)                       # [pt*ct]
        if refresh_params_below is not None and _MEM_FREE:
            gc.collect()
        g_i, z_scaled_i = gate_forward(engine, z, tree.cheb_coeffs, tree.band,
                                       keypack, return_zscaled=True)
        if refresh_below is not None:   # ★ 게이트 출력만 refresh (경로곱 소진 방지)
            g_i = _ensure_level(engine, g_i, keypack, refresh_below, tag=f'g[{i}]', verbose=verbose)
        G.append(g_i); Zc.append(z_scaled_i)

    # 경로확률 확장 (block 순서). pbs[i] = 레벨 i 진입 전 prob 리스트.
    prob = [engine.encrypt([1.0] * engine.slot_count, keypack.public_key)]
    pbs = []
    for i in range(d):
        pbs.append(list(prob))                                  # ★ 진입 전 prob 저장
        g = G[i]; one_minus_g = engine.subtract(one_pt, g)
        left  = [engine.multiply(p, one_minus_g, rlk) for p in prob]   # 왼쪽 블록
        right = [engine.multiply(p, g, rlk)           for p in prob]   # 오른쪽 블록
        prob  = left + right                                     # block: [왼쪽전체, 오른쪽전체]

    def _v(l):
        if refresh_params_below is not None:
            return _ensure_level(engine, tree.v[l], keypack, refresh_params_below,
                                 tag=f'pv[{l}]', verbose=verbose)
        return tree.v[l]
    f = engine.multiply(_v(0), prob[0], rlk)
    for l in range(1, len(prob)):
        vl = _v(l)
        term = engine.multiply(vl, prob[l], rlk)
        f = engine.add(f, term)
        if refresh_params_below is not None and _MEM_FREE: del vl, term
    if return_cache:
        return f, {"G": G, "Zc": Zc, "prob": prob, "pbs": pbs}
    return f


# ============================================================================
# ### CKKS-CORE : backward (block-순서, 나눗셈 없음 — 유한차분 검증됨)
# ============================================================================
def backward(engine, tree, X, cache, dL_df, keypack, refresh_below=None, verbose=False):
    rlk = keypack.relinearization_key
    d, F = tree.depth, tree.F
    G, Zc, prob, pbs = cache["G"], cache["Zc"], cache["prob"], cache["pbs"]
    one_pt = _c(engine, 1.0); zscale_pt = _c(engine, tree.zscale)
    # ★ 방향 B: 근원(dL_df) 1회 refresh → 하류 전체(grad_v, grad_prob, ...) 연쇄로 살아남.
    #   (grad_prob 2^d개 각각 refresh 하던 것 = OOM 원인 → 1회로 대체)
    if refresh_below is not None:
        dL_df = _ensure_level(engine, dL_df, keypack, refresh_below, tag='dL_df', verbose=verbose)

    # grad_v[ℓ] = Σ_slots dL_df · prob[ℓ]
    grad_v = []
    for l in range(len(prob)):
        gv_prod = engine.multiply(dL_df, prob[l], rlk)
        grad_v.append(_slot_sum(engine, gv_prod, keypack))
        if _MEM_FREE: del gv_prod

    # grad_prob[ℓ] = dL_df · v[ℓ]   (연쇄 시작, dL_df 이미 refresh됨)
    grad_prob = [engine.multiply(dL_df, tree.v[l], rlk) for l in range(len(prob))]

    grad_W = [[None] * F for _ in range(d)]; grad_b = [None] * d
    for i in reversed(range(d)):
        num = 2 ** i
        left, right = grad_prob[:num], grad_prob[num:]          # block 분할
        g = G[i]; one_minus_g = engine.subtract(one_pt, g)
        dsig = gate_derivative_poly(engine, Zc[i], tree.dcheb_coeffs,
                                    tree.band, keypack)          # 정확한 도함수

        # grad_g_i = Σ_k (right[k]-left[k]) · pb_i[k]   (나눗셈 없음)
        grad_g = None
        for k in range(num):
            diff = engine.subtract(right[k], left[k])
            term = engine.multiply(diff, pbs[i][k], rlk)
            grad_g = term if grad_g is None else engine.add(grad_g, term)
            if _MEM_FREE: del diff, term                        # ★ 즉시 해제

        # grad_z_i = grad_g · σ'(z) · zscale
        grad_z = engine.multiply(grad_g, dsig, rlk)
        if _MEM_FREE: del grad_g, dsig                          # ★
        grad_z = engine.multiply(grad_z, zscale_pt)
        if refresh_below is not None:   # ★ grad_W 합 전 refresh (X 곱 소진 방지)
            grad_z = _ensure_level(engine, grad_z, keypack, refresh_below, tag=f'gz[{i}]', verbose=verbose)

        # grad_W[i][f] = Σ_slots grad_z · X[f] ;  grad_b[i] = Σ_slots grad_z
        for f in range(F):
            prod = engine.multiply(grad_z, X[f], rlk)
            grad_W[i][f] = _slot_sum(engine, prod, keypack)
            if _MEM_FREE: del prod                              # ★
        grad_b[i] = _slot_sum(engine, grad_z, keypack)
        if _MEM_FREE: del grad_z

        # grad_prob(다음 레벨) = left·(1-g) + right·g
        new_gp = []
        for k in range(num):
            lt = engine.multiply(left[k], one_minus_g, rlk)
            rt = engine.multiply(right[k], g, rlk)
            new_gp.append(engine.add(lt, rt))
            if _MEM_FREE: del lt, rt
        if _MEM_FREE:
            del grad_prob, left, right, one_minus_g
        grad_prob = new_gp
        if _MEM_FREE and (i % 1 == 0): gc.collect()             # 레벨마다 gc
    if _MEM_FREE: del grad_prob
    return grad_W, grad_b, grad_v


# ============================================================================
# 레벨 관리: 소진 시 refresh (small-key bootstrap). 실측 로깅으로 교체할 훅.
# ============================================================================
def _maybe_refresh(engine, ct, keypack, min_level, tag="", verbose=False):
    """ct.level 이 min_level 미만이면 값 보존 bootstrap. DBSCAN 패턴(레벨 확인 후 refresh)."""
    try:
        lvl = ct.level
    except Exception:
        return ct
    if lvl < min_level:
        if verbose:
            print(f"    [BOOT] {tag}: level={lvl} < {min_level} → refresh")
        return refresh(engine, ct, keypack)
    return ct


# ============================================================================
# ### CKKS-CORE : 학습 1스텝
# ============================================================================
def train_step(engine, tree, X, y, keypack, lr, mu=0.9,
               vel_W=None, vel_b=None, vel_v=None, sample_weight_pt=None,
               refresh_below=8, mid_refresh_below=6, verbose_level=False):
    # refresh_below       : step 시작 시 파라미터 refresh 임계 (파라미터가 이보다 낮으면 bootstrap)
    # mid_refresh_below   : forward/backward 중간 refresh 임계.
    #   ★ 반드시 small bootstrap 회복레벨보다 낮게! (안 그러면 refresh 직후 또 걸려 무한반복)
    #     최대 단일연산 소비(게이트/도함수 ~4)보다는 커야 언더플로 전에 refresh 됨. 권장 4~6.
    rlk = keypack.relinearization_key
    d, F = tree.depth, tree.F

    # (0) ★ 레벨 확인 후 refresh (forward 전) — momentum 갱신으로 파라미터가
    #     매 스텝 gradient 레벨로 떨어지므로, forward 가 게이트 평가에 필요한
    #     레벨을 확보하도록 refresh_below 미만이면 값 보존 bootstrap.
    for i in range(d):
        for f_ in range(F):
            tree.W[i][f_] = _maybe_refresh(engine, tree.W[i][f_], keypack,
                                           refresh_below, tag=f'W[{i}][{f_}]', verbose=verbose_level)
        tree.b[i] = _maybe_refresh(engine, tree.b[i], keypack, refresh_below,
                                   tag=f'b[{i}]', verbose=verbose_level)
    for l in range(len(tree.v)):
        tree.v[l] = _maybe_refresh(engine, tree.v[l], keypack, refresh_below,
                                   tag=f'v[{l}]', verbose=verbose_level)
    if verbose_level:
        try: print(f'    [LEVEL] W[0][0]={tree.W[0][0].level}, v[0]={tree.v[0].level} (forward 전)')
        except Exception: pass

    # (1) forward (방향 B: 중간 refresh 활성화)
    f, cache = forward(engine, tree, X, keypack, return_cache=True,
                       refresh_below=mid_refresh_below, verbose=verbose_level)
    # (2) 제곱손실 + 1/N 을 lr 에 흡수 (대수적 변형):
    #   dL/df = (f - y)/N ∝ (f - y).  ★ forward 마지막 σ(f) 링크 제거(depth ~2 절약,
    #   평문 검증 AUC ±0.006).  ★ 1/N 은 공개 상수라 lr 로 흡수(pt*ct 곱 1회도 제거).
    #   → 호출부에서 lr_effective = lr / N 로 넘길 것.
    dL_df = engine.subtract(f, y)                     # f - y  (비교/σ/나눗셈 없음)
    # ★ bootstrap: 샘플별 공개 난수 가중치(복원추출 횟수)를 gradient 에 곱함 → 진짜 bagging.
    #   sample_weight_pt 는 공개 평문(N 슬롯) → pt*ct 곱 1회. FHE-RF 의 핵심.
    if sample_weight_pt is not None:
        dL_df = engine.multiply(dL_df, sample_weight_pt)   # [pt*ct]
    # (3) backward (방향 B: 중간 refresh 활성화)
    grad_W, grad_b, grad_v = backward(engine, tree, X, cache, dL_df, keypack,
                                      refresh_below=mid_refresh_below, verbose=verbose_level)

    # (4) momentum 갱신 (전부 [pt*ct]+[add])
    if vel_W is None:
        vel_W = [[_zeros(engine, keypack) for _ in range(F)] for _ in range(d)]
        vel_b = [_zeros(engine, keypack) for _ in range(d)]
        vel_v = [_zeros(engine, keypack) for _ in range(len(tree.v))]
    mu_pt, lr_pt = _c(engine, mu), _c(engine, lr)
    for i in range(d):
        for f_ in range(F):
            m_ = engine.multiply(vel_W[i][f_], mu_pt)
            g_ = engine.multiply(grad_W[i][f_], lr_pt)
            vel_W[i][f_] = engine.subtract(m_, g_)
            tree.W[i][f_] = engine.add(tree.W[i][f_], vel_W[i][f_])
            if _MEM_FREE:
                del m_, g_
                grad_W[i][f_] = None    # ★ 참조만 끊음(리스트 reindex 방지)
        m_ = engine.multiply(vel_b[i], mu_pt); g_ = engine.multiply(grad_b[i], lr_pt)
        vel_b[i] = engine.subtract(m_, g_)
        tree.b[i] = engine.add(tree.b[i], vel_b[i])
        if _MEM_FREE:
            del m_, g_
            grad_b[i] = None
        if _MEM_FREE: gc.collect()                      # 레벨마다 gc
    for l in range(len(tree.v)):
        m_ = engine.multiply(vel_v[l], mu_pt); g_ = engine.multiply(grad_v[l], lr_pt)
        vel_v[l] = engine.subtract(m_, g_)
        tree.v[l] = engine.add(tree.v[l], vel_v[l])
        if _MEM_FREE: del m_, g_

    # (5) 중간 ciphertext 명시적 해제 → GPU 메모리 회수 (OOM 완화)
    if _MEM_FREE:
        del grad_W, grad_b, grad_v, cache, dL_df, f, mu_pt, lr_pt
        gc.collect()
    return tree, vel_W, vel_b, vel_v
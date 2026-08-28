# soft_tree/desilo/gate_eval.py
"""
게이트 다항식 평가 — composite minimax 기반 (Cheon-Kim-Kim / Lee et al. 계열).

 [교체 이력]
   (구) sigma(z) 를 Chebyshev 최소제곱 deg3 으로 근사, band=8
        → 대역내 최대오차 11.4%, 대역 밖에서 발산 (|g|=4.6 @|z|=16, 18.9 @|z|=24)
        → 게이트 출력 도달 범위가 [0.114, 0.886] 으로 0/1 에 도달 불가
   (신) g(z) = 0.5*(1 + f_1(z/B)),   f_1(x) = (3x - x^3)/2
        → f_1 은 minimax composite sign 근사의 기저 다항식 (Lee et al., IEEE TDSC 2022)
        → f_1(±1) = ±1,  f_1'(x) = (3-3x^2)/2 >= 0 (단조),  max|f_1| on [-1,1] = 1.000000 (정확)
        → f_1([-2,2]) ⊆ [-1,1]  → |z| <= 2B 까지 g ∈ [0,1] 유계
        → 게이트 출력이 정확히 [0,1] 에 도달 (hard tree 극한 표현 가능)
        → 깊이는 동일 (둘 다 3차 다항식)

 [핵심 성질 — 실측]
   * (B, zscale) 은 자유도가 하나. beta_eff = 3*zscale/B 만이 성능을 결정.
     (B=8,zs=1.5) 와 (B=12,zs=2.25) 와 (B=48,zs=9.0) 은 완전히 동일한 결과.
     따라서 "B 를 키워 안전 여유를 확보"하는 것은 무의미 (모델이 |z| 를 같이 밀어올림).
   * 권고 beta_eff ~= 0.56.  안전범위 |z|/B <= 2 초과 비율은 실측 0.08~0.22% 로 무해.
     (sum_l P_l = 1 이 대수적으로 보존되어 |P|max<=0.78, |f|max<=1.23 로 하류가 O(1))
   * beta 를 0.28 로 낮추면 초과 0% 이지만 반례 AUC 가 -0.077 → 초과를 허용하는 것이 옳음.

 [DesiloFHE 1.9 API]
   engine.evaluate_chebyshev_polynomial(ct_nonNTT, coefficients, relin_key)
   → 입력은 non-NTT(coefficient) 형태여야 함 (intt 로 변환).

 [기저 선택 근거]
   power basis 는 절대 쓰지 말 것. deg15/band48 기준 계수 동적범위가 2.4e+17 로
   CKKS 스케일(2^40 ~ 1e12)을 초과해 정밀도가 파괴됨. Chebyshev 기저는 18.7 수준.
   (단, doubling map 처럼 순한 함수 조각들은 power 여도 계수가 O(1) 이라 안전)
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial import chebyshev as _C
try:
    from desilofhe import Engine, Ciphertext
except Exception:
    Engine = Ciphertext = object   # 평문 참조/문법검사용 폴백
from Soft_Tree.desilo.util.keypack import KeyPack


# ─────────────────────────────────────────────────────────────────────────
# f_n : Cheon-Kim-Kim / Lee et al. 의 composite sign 기저 다항식
#   f_n(x) = sum_{i=0}^{n} (1/4^i) C(2i,i) x (1-x^2)^i,   deg = 2n+1
#   n=1 → f_1(x) = (3x - x^3)/2         (deg 3, 권장)
#   n=2 → f_2(x) = ... (deg 5)          — 실측상 구간 밖 거동이 f_1 보다 나쁨. 비권장.
# ─────────────────────────────────────────────────────────────────────────
def _fn_power_coeffs(n: int) -> np.ndarray:
    """f_n 의 power basis 계수 (낮은 차수부터). 홀수차만 비영."""
    from math import comb
    p = np.zeros(2 * n + 2)
    for i in range(n + 1):
        c = comb(2 * i, i) / (4.0 ** i)
        for j in range(i + 1):
            p[2 * j + 1] += c * comb(i, j) * ((-1.0) ** j)
    return p


def fit_gate_composite(B: float = 8.0, n: int = 1):
    """
    게이트 g(z) = 0.5*(1 + f_n(z/B)) 의 Chebyshev 계수를 반환.

    Returns
    -------
    coeffs  : list[float]   chebval(z/B, coeffs) == g(z)
    dcoeffs : list[float]   chebval(z/B, dcoeffs) == dg/dz     ★ 1/B 가 이미 흡수됨
    B       : float

    n=1 기준 이론값:
      coeffs  = [0.5, 0.5625, 0.0, -0.0625]        (= 0.5 + 0.5*((9/8)T1 - (1/8)T3))
      dcoeffs = [0.375, 0.0, -0.375] / B
    """
    pw = _fn_power_coeffs(n)
    f_cheb = _C.poly2cheb(pw)                 # f_n 의 Chebyshev 계수
    g_cheb = 0.5 * f_cheb
    g_cheb[0] += 0.5                          # g = 0.5*(1 + f_n)
    dg_cheb = _C.chebder(g_cheb) / float(B)   # dg/dz = (dg/du)*(1/B)
    return g_cheb.tolist(), dg_cheb.tolist(), float(B)


# 하위 호환 별칭 — 기존 호출부(fit_sigmoid_chebyshev)를 그대로 두고 싶을 때 사용.
# ★ deg 인자는 무시된다 (f_1 고정). band 는 B 로 해석된다.
def fit_sigmoid_chebyshev(deg: int = 3, band: float = 8.0, n_grid: int = 0):
    """[DEPRECATED 이름 유지] 실제로는 composite minimax f_1 게이트를 반환."""
    return fit_gate_composite(B=band, n=1)


# ─────────────────────────────────────────────────────────────────────────
# 서버 암호문: 게이트 forward
# ─────────────────────────────────────────────────────────────────────────
def gate_forward(engine: Engine, z: Ciphertext, cheb_coeffs, band: float,
                 keypack: KeyPack, return_zscaled: bool = False,
                 prescaled: bool = False):
    """
    g = 0.5*(1 + f_1(z/B)).
    prescaled=True 이면 z 가 이미 z/B 로 스케일된 상태라고 보고 1/B 곱을 건너뛴다.
      ★ packed_core 가 combo_pt(= smask · zscale · 1/B) 로 미리 접어서 넘기므로
        평문곱 1회(레벨 1)를 절약한다.
    return_zscaled=True 이면 (g, u_nonNTT) 반환 — backward 도함수 평가에 재사용.
    깊이: prescaled 면 2(deg3 다항), 아니면 3.
    """
    if prescaled:
        u = z
    else:
        inv_band_pt = engine.encode([1.0 / band] * engine.slot_count)
        u = engine.multiply(z, inv_band_pt)
    u = engine.intt(u)                                            # NTT → non-NTT
    g = engine.evaluate_chebyshev_polynomial(
        u, cheb_coeffs, keypack.relinearization_key)
    if return_zscaled:
        return g, u
    return g


# ─────────────────────────────────────────────────────────────────────────
# 서버 암호문: 게이트 도함수 (정확한 다항 도함수)
# ─────────────────────────────────────────────────────────────────────────
def gate_derivative_poly(engine: Engine, z_scaled_nonNTT: Ciphertext,
                         dcheb_coeffs, band: float, keypack: KeyPack) -> Ciphertext:
    """
    dg/dz = 0.5 * f_1'(z/B) / B.  forward 에서 캐시한 z_scaled(non-NTT) 재사용.
    ★ 1/B 는 dcheb_coeffs 에 이미 흡수되어 있으므로 추가 곱셈이 없다 (구버전 대비 깊이 -1).
    ★ band 인자는 시그니처 호환을 위해 남겨둔 것이며 사용되지 않는다.
    """
    return engine.evaluate_chebyshev_polynomial(
        z_scaled_nonNTT, dcheb_coeffs, keypack.relinearization_key)


# ─────────────────────────────────────────────────────────────────────────
# 마스킹용 상수 — 곱셈(1레벨) 대신 평문 뺄셈(0레벨)으로 처리하기 위한 값
#   combo_pt 에 smask 가 들어 있으므로 비샘플 슬롯에서는 u = 0 이고,
#   따라서 게이트/도함수 출력이 데이터와 무관한 "상수" 가 된다.
#     g(0)      = coeffs[0] - coeffs[2] + ...  (Chebyshev: T0(0)=1,T1(0)=0,T2(0)=-1,T3(0)=0)
#     dg/dz(0)  = 같은 규칙을 dcoeffs 에 적용
#   이 상수를 담은 평문을 빼주면 곱셈 없이 마스킹된다.
# ─────────────────────────────────────────────────────────────────────────
def gate_offset_values(B: float, n: int = 1):
    """(g(0), dg/dz(0)) — 비샘플 슬롯에서 제거해야 할 상수. n=1 기준 (0.5, 0.75/B)."""
    cc, dcc, _ = fit_gate_composite(B=B, n=n)
    g0 = float(_C.chebval(0.0, cc))
    d0 = float(_C.chebval(0.0, dcc))
    return g0, d0


# ─────────────────────────────────────────────────────────────────────────
# 평문 참조 (검증용) — FHE 와 완전히 동일한 함수. ★클리핑 없음★
# ─────────────────────────────────────────────────────────────────────────
def gate_plain(z, coeffs, band):
    """FHE 의 gate_forward 와 수학적으로 동일. np.clip 을 절대 넣지 말 것."""
    return _C.chebval(np.asarray(z) / band, coeffs)


def gate_deriv_plain(z, dcoeffs, band):
    """FHE 의 gate_derivative_poly 와 동일 (1/B 흡수됨)."""
    return _C.chebval(np.asarray(z) / band, dcoeffs)


# ─────────────────────────────────────────────────────────────────────────
# 공개 난수 feature 마스크 (column subsampling, Breiman RF 의 mtry)
#   ★ 데이터와 무관한 순수 난수 → 평문으로 두어도 leakage 없음
#   ★ 평문 참조와 FHE 가 반드시 같은 마스크를 써야 하므로 여기서 일원화
# ─────────────────────────────────────────────────────────────────────────
MASK_SEED_BASE = 50_000


def feature_mask(F: int, tree_k: int, level_i: int, mtry=None) -> np.ndarray:
    """트리 k, 깊이 i 의 feature 마스크 (길이 F, 0/1). mtry=None 이면 전체 1."""
    m = np.zeros(F)
    if mtry is None or mtry >= F:
        m[:] = 1.0
        return m
    rng = np.random.default_rng(MASK_SEED_BASE + 1000 * tree_k + level_i)
    m[rng.choice(F, size=int(mtry), replace=False)] = 1.0
    return m
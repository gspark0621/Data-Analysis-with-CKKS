# core/ciphertext_single/cleaning.py
"""
Discrete-CKKS cleaning functions for noise reduction after sign_bootstrap.

배경 (측정으로 확정된 문제):
  Core/Normalize는 sign_bootstrap 직후 (±1, 측정치 0.999996, ~2^-18 정밀)
  그 뒤 일반 engine.bootstrap이 noise를 *주입*하여 0.99839 (~2^-9.3)로 악화.
  (논문: Hong et al., "A White-Box Bootstrapping Approach...", §1.3 —
   "일반 bootstrap은 noise를 추가한다")

  이 0.99839가 mask로 누적 곱해지며 라벨 damping / total_neighbors bias를 유발.

해결:
  일반 bootstrap을 cleaning 함수로 대체. cleaning은 noise를 주입하지 않고
  discrete 값(0/1 또는 ±1)으로 quadratic 수렴시킴.

두 가지 cleaning (도메인에 따라 구분):
  - bit_cleaning  h(x) = 3x² - 2x³        : {0,1} 도메인 (Drucker et al., Lemma 1)
                                              Core core_mask, Normalize adj_k 용
  - sign_cleaning g(x) = (3/2)x - (1/2)x³  : {-1,+1} 도메인 (Cheon et al. / 본 논문 §4.2)
                                              LP fhe_sgn 용

레벨 소비:
  두 cleaning 모두 1회당 2 multiplicative level 소비 (x² 1회 + x²·x 1회).
  n_iters회 반복 시 2*n_iters 레벨.

정밀도 (평문 검증, 입력 0.99839 기준):
  bit_cleaning  1회: 0.99839 → 0.9999923   (2^-17)
                2회:         → ~1.0          (2^-32, CKKS 한계 근접)
  sign_cleaning 1회: 0.99839 → 0.99999      (유사)

n_iters 권장:
  1회면 2^-17 (충분), 2회면 CKKS 정밀도 한계 도달. 3회 이상은 무의미.
  레벨 예산에 따라 1 또는 2 선택. 기본값 1 (레벨 절약).

레벨 안전장치 (★ 2026-05c 추가):
  cleaning은 일반 bootstrap과 달리 level을 *복구하지 않고 소비*(2*n_iters).
  따라서 두 지점에서 ciphertext.level을 확인하여 부족 시 _refresh(bootstrap):
    (1) 각 iter 진입 전: level >= 2 보장 (cleaning 자체 실행 가능하도록)
    (2) cleaning 완료 후: level >= _MIN_SAFE_LEVEL(=3, sign_bootstrap 요구치) 보장
       (후속 연산/다음 모듈의 _refresh 입력 level 확보)
  이로써 "sign_bootstrap 후 잔여 level이 cleaning에 부족"한 경우를 자동 처리.
"""

from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack


# 각 cleaning 1회가 소비하는 multiplicative level (square 1 + multiply 1).
_LEVEL_PER_ITER = 2

# sign_bootstrap 등 후속 연산이 요구하는 최소 입력 level (사용자 확인: 3).
# cleaning 완료 후 이 미만이면 _refresh로 복구.
_MIN_SAFE_LEVEL = 3


def _refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack) -> Ciphertext:
    """일반 bootstrap으로 level 복구 (Label_Propagation._refresh와 동일)."""
    return engine.bootstrap(
        engine.intt(ct),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.bootstrap_key,
    )


def _ensure_level(engine, ct, keypack, need_level, tag=""):
    """
    ct.level이 need_level 미만이면 _refresh로 복구.

    ciphertext.level로 현재 level 확인 (사용자 확인).
    AttributeError 등으로 level 조회 실패 시 보수적으로 _refresh 안 함
    (대부분 환경에서 level 속성 존재; 실패 시 호출측 흐름 유지).
    """
    try:
        lvl = ct.level
    except AttributeError:
        return ct  # level 조회 불가 환경 → 그대로 진행
    if lvl < need_level:
        print(f"    [cleaning{(' '+tag) if tag else ''}] level={lvl} < {need_level} "
              f"→ _refresh (bootstrap)")
        return _refresh(engine, ct, keypack)
    return ct


def bit_cleaning(
    engine: Engine,
    ct: Ciphertext,
    keypack: KeyPack,
    n_iters: int = 1,
    slot_count: int = None,
    ensure_output_level: bool = True,
) -> Ciphertext:
    """
    {0,1} 도메인 cleaning: h(x) = 3x² - 2x³ = x²(3 - 2x).

    noisy bit (예: 0.99839 또는 0.00161)을 깨끗한 {0,1}로 quadratic 수렴.
    Drucker et al. Lemma 1: |h(m+τ) - m| ≤ 5|τ|² for m∈{0,1}, |τ|≤1.

    레벨 소비: 2 * n_iters. 각 iter 진입 전 level >= 2 확인, 부족 시 _refresh.
    ensure_output_level=True 면 완료 후 level < _MIN_SAFE_LEVEL 일 때 _refresh.
    """
    if slot_count is None:
        slot_count = engine.slot_count
    relin_key = keypack.relinearization_key

    three_pt = engine.encode([3.0] * slot_count)
    two_pt   = engine.encode([2.0] * slot_count)

    x = ct
    for it in range(n_iters):
        # 이 iter가 level 2를 소비하므로 진입 전 보장
        x = _ensure_level(engine, x, keypack, _LEVEL_PER_ITER, tag=f"bit iter{it+1}")
        x_sq   = engine.square(x, relin_key)              # x²       (level -1)
        two_x  = engine.multiply(x, two_pt)               # 2x       (scalar)
        inner  = engine.subtract(three_pt, two_x)         # 3 - 2x
        x      = engine.multiply(x_sq, inner, relin_key)  # x²(3-2x) (level -1)

    if ensure_output_level:
        x = _ensure_level(engine, x, keypack, _MIN_SAFE_LEVEL, tag="bit out")
    return x


def sign_cleaning(
    engine: Engine,
    ct: Ciphertext,
    keypack: KeyPack,
    n_iters: int = 1,
    slot_count: int = None,
    ensure_output_level: bool = True,
) -> Ciphertext:
    """
    {-1,+1} 도메인 cleaning: g(x) = (3/2)x - (1/2)x³ = x(3 - x²)/2.

    noisy sign 값을 깨끗한 ±1로 quadratic 수렴.
    Cheon et al. (ASIACRYPT'20) clean₁, 본 논문 §4.2에서 hybrid에 사용.

    레벨 소비: 2 * n_iters. 각 iter 진입 전 level >= 2 확인, 부족 시 _refresh.
    """
    if slot_count is None:
        slot_count = engine.slot_count
    relin_key = keypack.relinearization_key

    three_pt = engine.encode([3.0] * slot_count)
    half_pt  = engine.encode([0.5] * slot_count)

    x = ct
    for it in range(n_iters):
        x = _ensure_level(engine, x, keypack, _LEVEL_PER_ITER, tag=f"sign iter{it+1}")
        x_sq   = engine.square(x, relin_key)            # x²
        inner  = engine.subtract(three_pt, x_sq)        # 3 - x²
        prod   = engine.multiply(x, inner, relin_key)   # x(3 - x²)
        x      = engine.multiply(prod, half_pt)         # x(3-x²)/2

    if ensure_output_level:
        x = _ensure_level(engine, x, keypack, _MIN_SAFE_LEVEL, tag="sign out")
    return x
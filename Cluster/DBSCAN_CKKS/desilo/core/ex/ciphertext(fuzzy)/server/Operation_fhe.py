"""
FHE 기본 연산 모음.
모든 함수는 Ciphertext를 받고 Ciphertext를 반환.

Numpy 대응표:
  np.roll(arr, -k)         → fhe_rotate(engine, ct, rk, k)
  (arr > 0)                → fhe_heaviside(...)
  np.maximum(a, b)         → fhe_max(...)
  (dist_sq <= eps_sq)      → fhe_check_neighbor(...)
  (total_n >= min_pts)     → fhe_core_mask(...)
"""
from core.ciphertext.shared.keypack import KeyPack
from typing import Any


def fhe_rotate(engine, ct: Any, rotation_key: Any, k: int) -> Any:
    """
    np.roll(arr, -k) 와 동일: result[i] = arr[(i+k) % N]
    desilofhe rotate(delta=k) = 오른쪽 k칸 이동 = result[i] = arr[i-k]
    따라서 왼쪽 k칸 이동은 delta=-k
    """
    return engine.rotate(ct, rotation_key, delta=-k)


def fhe_sign_poly(engine, ct: Any, kp: KeyPack, degree: int = 3) -> Any:
    """
    ① sign 다항식 근사: 임의 x → (-1, +1) 범위로 압축
    입력 범위: [-1, 1] 정규화 필수

    현재: 단순 odd 다항식 (1.5x - 0.5x³)  — 1회당 2레벨 소모
    추후: Minimax Composite Polynomial 논문 방식으로 교체 예정

    degree=3 → 1회
    degree=5 → 2회 합성
    """
    # 1.5x - 0.5x³  (Chebyshev 기반 1차 근사)
    x_sq  = engine.multiply(ct, ct, kp.relinearization_key)      # x²   -1 level
    x_cu  = engine.multiply(x_sq, ct, kp.relinearization_key)    # x³   -1 level
    term1 = engine.multiply(ct, 1.5)                              # 1.5x
    term2 = engine.multiply(x_cu, -0.5)                          # -0.5x³
    return engine.add(term1, term2)                               # ≈ sign(x), 2 level 소모


def fhe_sign(engine, ct: Any, kp: KeyPack) -> Any:
    """
    ② 완전한 sign: 다항식 근사 → sign_bootstrap
    입력: 임의 값 (정규화된 [-1, 1] 범위)
    출력: ≈ -1 또는 +1, 레벨 16 - stage_count 로 복원
    """
    approx = fhe_sign_poly(engine, ct, kp)          # x → -1/+1 근사 (2 level 소모)
    return engine.sign_bootstrap(                    # snap + 레벨 복원
        approx,
        kp.relinearization_key,
        kp.conjugation_key,
        kp.lossy_bootstrap_key,                      # lossy_bootstrap_key 사용
    )

def fhe_heaviside(engine, ct: Any, kp: KeyPack) -> Any:
    """
    Heaviside: approx 1 if ct>0, else approx 0.
    = (1 + sign(ct)) / 2

    레벨 소모: sign_bootstrap 1회
    """
    sign_ct = fhe_sign(engine, ct, kp)
    # (1 + sign) * 0.5
    result = engine.multiply(engine.add(sign_ct, 1.0), 0.5)
    return result


def fhe_check_neighbor(engine,
                       dist_sq_ct: Any,
                       eps_norm_sq: float,
                       kp: KeyPack) -> Any:
    """
    adj_k[i] ≈ 1 if dist²[i] ≤ eps², else ≈ 0.
    = Heaviside(eps² - dist²)
    = Heaviside(1 - dist²/eps²)   ← eps²으로 정규화해 sign 정밀도 유지

    Numpy 대응: check_neighbor_closed_interval(dist_sq, eps_norm_sq)

    레벨 소모:
      multiply (정규화): -1
      sign_bootstrap:   refresh
    """
    # 1 - dist² / eps²
    dist_sq_norm = engine.multiply(dist_sq_ct, -1.0 / eps_norm_sq)   # -dist²/eps²
    ct_to_sign   = engine.add(dist_sq_norm, 1.0)                      # 1 - dist²/eps²
    return fhe_heaviside(engine, ct_to_sign, kp)


def fhe_fuzzy_neighbor_same_cell(engine,
                                  dist_sq_ct: Any,
                                  eps_norm_sq: float) -> Any:
    """
    [fuzzy 이웃 기여값 — 같은 cell 내부 전용]

    전제: grid cell 한 변 = eps/sqrt(d) 이면
          같은 cell 안의 두 점은 반드시 d(x,y) <= eps.
          -> sign_bootstrap 0회 절약

    FN-DBSCAN: N_x(y) = 1 - d^2/eps^2

    레벨 소모:
      multiply (plaintext scalar): -1
      sign_bootstrap:               0  <- 절약!
    """
    dist_sq_norm = engine.multiply(dist_sq_ct, 1.0 / eps_norm_sq)      # d^2/eps^2
    return engine.add(engine.multiply(dist_sq_norm, -1.0), 1.0)        # 1 - d^2/eps^2


def fhe_fuzzy_neighbor_intra(engine,
                               dist_sq_ct: Any,
                               eps_norm_sq: float,
                               adj_k: Any,
                               kp: KeyPack) -> Any:
    """
    [fuzzy 이웃 기여값 — 인접 cell 전용]

    이미 계산된 crisp adj_k (0/1)를 clamp로 재활용.
    -> 추가 sign_bootstrap 불필요

    FN-DBSCAN: N_x(y) = max(0, 1 - d^2/eps^2)
                       = (1 - d^2/eps^2) * adj_k

    레벨 소모:
      multiply (plaintext scalar): -1
      multiply (x adj_k, relin):   -1
      sign_bootstrap:               0  (adj_k 계산 시 이미 소모)
    """
    dist_sq_norm = engine.multiply(dist_sq_ct, 1.0 / eps_norm_sq)
    raw_contrib  = engine.add(engine.multiply(dist_sq_norm, -1.0), 1.0)  # 1 - d^2/eps^2
    return engine.multiply(raw_contrib, adj_k, kp.relinearization_key)

def fhe_valid_mask(engine, coord_ct: Any, kp: KeyPack) -> Any:
    """
    valid_mask[i] ≈ 1 if coord[i] < 1.5 (real point)
                   ≈ 0 if coord[i] = 2.0 (dummy point)
    = Heaviside(1.5 - coord[i])

    레벨 소모: sign_bootstrap 1회
    """
    # 1.5 - coord → add(-1.0*coord, 1.5)
    neg_coord  = engine.multiply(coord_ct, -1.0)
    ct_to_sign = engine.add(neg_coord, 1.5)
    return fhe_heaviside(engine, ct_to_sign, kp)


def fhe_core_mask(engine,
                  total_neighbors_ct: Any,
                  min_pts: int,
                  N_real: int,
                  kp: KeyPack) -> Any:
    """
    core_mask[i] ≈ 1 if total_neighbors[i] >= min_pts, else ≈ 0
    = Heaviside(total_neighbors - (min_pts - 0.5))

    N_real로 정규화해 sign 정밀도 유지.

    레벨 소모: sign_bootstrap 1회
    """
    threshold = (min_pts - 0.5) / N_real
    # total_neighbors/N_real - threshold
    normed    = engine.multiply(total_neighbors_ct, 1.0 / N_real)
    ct_to_sign = engine.add(normed, -threshold)
    return fhe_heaviside(engine, ct_to_sign, kp)


def fhe_max(engine, a: Any, b: Any, kp: KeyPack) -> Any:
    """
    max(a, b) = (a + b + |a - b|) / 2
              = (a + b + (a-b)*sign(a-b)) / 2

    레벨 소모:
      subtract:         0
      sign_bootstrap:   refresh
      multiply (relin): -1
      add/multiply:     0
    """
    diff     = engine.subtract(a, b)
    sign_d   = fhe_sign(engine, diff, kp)
    abs_diff = engine.multiply(sign_d, diff, kp.relin_key)   # |a-b|
    return engine.multiply(
        engine.add(engine.add(a, b), abs_diff),
        0.5
    )
# core/ciphertext/server/Normalize.py
"""
이웃 판별 함수.

변경 내용:
  기존: sign 다항식(1.5x-0.5x³) required_depth 회 반복 → 일반 bootstrap
  변경: 스케일링 → lifting_to_pm1 → sign_bootstrap (fhe_sign_lifted)

핵심 개선:
  이웃 케이스(dist² << eps²)는 스케일 후 0 근방(-0.023 수준)에 몰리는데,
  기존 방식은 이 범위에서 수렴이 매우 느렸음.
  lifting 으로 먼저 ±1 근방으로 밀어낸 뒤 sign_bootstrap 으로 정확히 snap.
"""

import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext.server.SignUtils import (
    compute_lifting_depth,
    fhe_heaviside_lifted,
)


def check_neighbor_closed_interval(
    engine:    Engine,
    dist_sq_ct: Ciphertext,
    eps_sq:    float,
    keypack:   KeyPack,
    dimension: int,
    margin:    float = 0.05,
    tau:       float = 0.1,
) -> Ciphertext:
    """
    이웃 판별: dist² ≤ eps²  →  ≈ 1,  아니면  ≈ 0.
    (내부 로직 변경 없음 — 이 함수는 원래 bootstrap 을 직접 호출하지 않음)
    """
    slot_count = engine.slot_count

    threshold    = eps_sq + margin
    threshold_pt = engine.encode([threshold] * slot_count)
    x            = engine.subtract(dist_sq_ct, threshold_pt)

    max_dist_sq = float(dimension)
    lower_abs   = threshold
    upper_abs   = max_dist_sq - threshold
    bound       = max(lower_abs, upper_abs) * 1.1

    scale_pt  = engine.encode([1.0 / bound] * slot_count)
    x_scaled  = engine.multiply(x, scale_pt)

    min_abs_input = margin / bound
    depth = compute_lifting_depth(min_abs_input, tau=tau)

    neg_x_scaled = engine.multiply(x_scaled, -1.0)
    return fhe_heaviside_lifted(engine, neg_x_scaled, keypack, depth)
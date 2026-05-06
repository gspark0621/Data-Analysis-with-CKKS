# core/ciphertext/server/Core.py
"""
Core point 판별 함수.

변경 내용:
  기존: sign 다항식 required_depth 회 반복 → 일반 bootstrap
  변경: 스케일링 → lifting → sign_bootstrap (fhe_sign_lifted)
"""

import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext.server.SignUtils import (
    compute_lifting_depth,
    fhe_heaviside_lifted,
)


def identify_core_points_fhe_converted(
    engine:            Engine,
    neighbor_count_ct: Ciphertext,
    min_pts:           float,
    N:                 int,
    keypack:           KeyPack,
    margin:            float = 0.5,
    tau:               float = 0.1,
) -> Ciphertext:
    """
    Core point 판별: neighbor_count ≥ min_pts  →  ≈ 1, else ≈ 0.
    (내부 로직 변경 없음 — 이 함수는 원래 bootstrap 을 직접 호출하지 않음)
    """
    slot_count = engine.slot_count

    threshold = min_pts - margin
    thresh_pt = engine.encode([threshold] * slot_count)
    x         = engine.subtract(neighbor_count_ct, thresh_pt)

    scale_pt  = engine.encode([1.0 / float(N)] * slot_count)
    x_scaled  = engine.multiply(x, scale_pt)

    min_abs_input = margin / float(N)
    depth = compute_lifting_depth(min_abs_input, tau=tau)
    print(f"[Core] lifting depth={depth}  (margin={margin}, N={N})")

    return fhe_heaviside_lifted(engine, x_scaled, keypack, depth)
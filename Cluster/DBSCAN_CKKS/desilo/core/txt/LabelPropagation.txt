# core/ciphertext/server/LabelPropagation.py
"""
라벨 전파 함수.

변경 내용:
  기존: fhe_sign_unit (1.5x-0.5x³ 반복 + 일반 bootstrap)
        fhe_hard_mask_01 (동일 방식)
  변경: fhe_sign_lifted / fhe_heaviside_lifted (lifting + sign_bootstrap)

  fhe_fast_max_unit 의 내부 sign 도 동일하게 교체.
"""
# core/ciphertext/server/LabelPropagation.py

import math
from typing import List, Tuple
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext.server.SignUtils import (
    compute_lifting_depth,
    fhe_sign_lifted,
    fhe_heaviside_lifted,
    refresh_via_sign,
)

_MAX_LIFTING_DEPTH_DEFAULT = 8
_MASK_LIFTING_DEPTH_DEFAULT = 4

# 라벨 정규화 범위: (0, 1] → scale=1.0 으로 refresh_via_sign 적용
_LABEL_SCALE = 1.0
# adj/core/mask: 0 or 1 → scale=1.0
_BINARY_SCALE = 1.0


def refresh(engine: Engine, ct: Ciphertext, keypack: KeyPack,
            scale: float = _BINARY_SCALE) -> Ciphertext:
    """
    sign_bootstrap 기반 레벨 복원 (일반 bootstrap 완전 제거).

    scale: 슬롯값 절대최대. 기본 1.0 (이진/정규화 라벨 용도).
    ⚠️ 부호만 보존 → 연속 절대값이 필요한 암호문에는 직접 사용 금지.
    """
    return refresh_via_sign(engine, ct, keypack, scale=scale)


def fhe_circular_shift(engine, ct, k, numpoints, keypack):
    """컬럼-메이저 슬롯 기준 순환 이동 (변경 없음)."""
    slot_count    = engine.slot_count
    left_shifted  = engine.rotate(ct, keypack.rotation_key, -k)
    right_shifted = engine.rotate(ct, keypack.rotation_key, numpoints - k)
    mask_left_pt  = [1.0 if i < numpoints - k else 0.0
                     for i in range(slot_count)]
    mask_right_pt = [0.0] * slot_count
    for i in range(numpoints - k, numpoints):
        mask_right_pt[i] = 1.0
    left_clean  = engine.multiply(left_shifted, engine.encode(mask_left_pt))
    right_clean = engine.multiply(right_shifted, engine.encode(mask_right_pt))
    return engine.add(left_clean, right_clean)


def fhe_fast_max_unit(
    engine:    Engine,
    A_ct:      Ciphertext,
    B_ct:      Ciphertext,
    numpoints: int,
    keypack:   KeyPack,
    depth:     int = _MAX_LIFTING_DEPTH_DEFAULT,
) -> Ciphertext:
    """
    max(A, B) = A + ReLU(B - A)

    변경: 중간 refresh 를 sign_bootstrap 기반 refresh() 로 교체.
    diff = B - A ∈ [-1, 1] → scale=1.0 으로 refresh 가능.
    """
    relin   = keypack.relinearization_key
    half    = engine.encode([0.5] * numpoints)

    diff_ct = engine.subtract(B_ct, A_ct)
    diff_ct = refresh(engine, diff_ct, keypack, scale=_LABEL_SCALE)

    sign_ct = fhe_sign_lifted(engine, diff_ct, keypack, depth)

    diff_ref = refresh(engine, diff_ct, keypack, scale=_LABEL_SCALE)
    abs_diff = engine.multiply(diff_ref, sign_ct, relin)
    abs_diff = refresh(engine, abs_diff, keypack, scale=_LABEL_SCALE)

    relu_ct = engine.multiply(engine.add(diff_ref, abs_diff), half)
    return refresh(engine, engine.add(A_ct, relu_ct), keypack, scale=_LABEL_SCALE)


def fhe_hard_mask_01(
    engine:    Engine,
    x_ct:      Ciphertext,
    numpoints: int,
    keypack:   KeyPack,
    depth:     int = _MASK_LIFTING_DEPTH_DEFAULT,
) -> Ciphertext:
    """
    0/1 근사값을 딱딱한 0 또는 1로 snap.

    변경: 중간 refresh 를 sign_bootstrap 기반으로 교체.
    (x - 0.5) ∈ [-0.5, 0.5] → scale=0.5 로 refresh.
    """
    half     = engine.encode([0.5] * numpoints)
    centered = engine.subtract(x_ct, half)
    centered = refresh(engine, centered, keypack, scale=0.5)

    sign_ct  = fhe_sign_lifted(engine, centered, keypack, depth)
    out_ct   = engine.add(engine.multiply(sign_ct, half), half)
    return refresh(engine, out_ct, keypack, scale=_BINARY_SCALE)


def fhe_max_propagation_fhe(
    engine:        Engine,
    keypack:       KeyPack,
    adj_ct_pairs:  List[Tuple[int, Ciphertext]],
    core_ct:       Ciphertext,
    cluster_id_pt: list,
    numpoints:     int,
    maxiter:       int = None,
    max_depth:     int = _MAX_LIFTING_DEPTH_DEFAULT,
    mask_depth:    int = _MASK_LIFTING_DEPTH_DEFAULT,
) -> Ciphertext:
    """
    FHE Label Propagation.

    변경: 모든 refresh 를 sign_bootstrap 기반으로 통일.
    """
    relin          = keypack.relinearization_key
    cluster_id_enc = engine.encode(cluster_id_pt)
    if maxiter is None:
        maxiter = numpoints - 1

    # ── adj_ct 전처리 ────────────────────────────────────────────
    clean_adj_pairs: List[Tuple[int, Ciphertext]] = []
    for k, adj_ct in adj_ct_pairs:
        adj_ct = refresh(engine, adj_ct, keypack, scale=_BINARY_SCALE)
        adj_ct = fhe_hard_mask_01(engine, adj_ct, numpoints, keypack,
                                   depth=mask_depth)
        clean_adj_pairs.append((k, adj_ct))

    # ── Core / NonCore 마스크 ────────────────────────────────────
    core_mask_ct     = refresh(engine, core_ct, keypack, scale=_BINARY_SCALE)
    core_mask_ct     = fhe_hard_mask_01(engine, core_mask_ct,
                                         numpoints, keypack, depth=mask_depth)
    non_core_mask_ct = engine.subtract(1.0, core_mask_ct)
    non_core_mask_ct = fhe_hard_mask_01(engine, non_core_mask_ct,
                                         numpoints, keypack, depth=mask_depth)

    # ── 초기 Core 라벨 ───────────────────────────────────────────
    core_labels_ct = engine.multiply(core_mask_ct, cluster_id_enc)
    core_labels_ct = refresh(engine, core_labels_ct, keypack,
                              scale=_LABEL_SCALE)
    zero_ct        = engine.subtract(core_labels_ct, core_labels_ct)

    # ── 반복 전파 ────────────────────────────────────────────────
    for iteration in range(maxiter):

        # Phase 1: Core ↔ Core 라벨 병합
        for k, adj_ct in clean_adj_pairs:
            sh_labels = fhe_circular_shift(engine, core_labels_ct,
                                           k, numpoints, keypack)
            sh_mask   = fhe_circular_shift(engine, core_mask_ct,
                                           k, numpoints, keypack)
            sh_labels = refresh(engine, sh_labels, keypack, scale=_LABEL_SCALE)
            sh_mask   = refresh(engine, sh_mask,   keypack, scale=_BINARY_SCALE)

            edge_mask = engine.multiply(adj_ct,    core_mask_ct, relin)
            edge_mask = engine.multiply(edge_mask, sh_mask,      relin)
            edge_mask = refresh(engine, edge_mask, keypack, scale=_BINARY_SCALE)

            cand = engine.multiply(edge_mask, sh_labels, relin)
            cand = refresh(engine, cand, keypack, scale=_LABEL_SCALE)

            core_labels_ct = fhe_fast_max_unit(
                engine, core_labels_ct, cand, numpoints, keypack,
                depth=max_depth,
            )
            core_labels_ct = engine.multiply(
                core_labels_ct, core_mask_ct, relin
            )
            core_labels_ct = refresh(engine, core_labels_ct, keypack,
                                      scale=_LABEL_SCALE)

        # Phase 2: Core → Border 라벨 전파
        border_labels_ct = zero_ct
        assigned_mask_ct = zero_ct

        for k, adj_ct in clean_adj_pairs:
            sh_labels = fhe_circular_shift(engine, core_labels_ct,
                                           k, numpoints, keypack)
            sh_mask   = fhe_circular_shift(engine, core_mask_ct,
                                           k, numpoints, keypack)
            sh_labels = refresh(engine, sh_labels, keypack, scale=_LABEL_SCALE)
            sh_mask   = refresh(engine, sh_mask,   keypack, scale=_BINARY_SCALE)

            cand_mask = engine.multiply(adj_ct,    sh_mask,          relin)
            cand_mask = engine.multiply(cand_mask, non_core_mask_ct, relin)
            cand_mask = refresh(engine, cand_mask, keypack, scale=_BINARY_SCALE)
            cand_mask = fhe_hard_mask_01(engine, cand_mask,
                                          numpoints, keypack, depth=mask_depth)

            empty_mask  = engine.subtract(1.0, assigned_mask_ct)
            empty_mask  = fhe_hard_mask_01(engine, empty_mask,
                                            numpoints, keypack, depth=mask_depth)
            accept_mask = engine.multiply(cand_mask, empty_mask, relin)
            accept_mask = refresh(engine, accept_mask, keypack, scale=_BINARY_SCALE)
            accept_mask = fhe_hard_mask_01(engine, accept_mask,
                                            numpoints, keypack, depth=mask_depth)

            accepted         = engine.multiply(accept_mask, sh_labels, relin)
            border_labels_ct = engine.add(border_labels_ct, accepted)
            border_labels_ct = refresh(engine, border_labels_ct, keypack,
                                        scale=_LABEL_SCALE)

            assigned_mask_ct = fhe_fast_max_unit(
                engine, assigned_mask_ct, accept_mask, numpoints, keypack,
                depth=max_depth,
            )
            assigned_mask_ct = fhe_hard_mask_01(
                engine, assigned_mask_ct, numpoints, keypack, depth=mask_depth
            )

    final_norm_ct = engine.add(core_labels_ct, border_labels_ct)
    return refresh(engine, final_norm_ct, keypack, scale=_LABEL_SCALE)
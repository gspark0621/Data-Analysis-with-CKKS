# core/ciphertext_single/bsgs_poly.py
#
# Odd Baby-Step Giant-Step (BSGS) polynomial evaluation for FHE.
#
# 참조: Lee et al., "Minimax Approximation of Sign Function by Composite
#       Polynomial for Homomorphic Comparison", IEEE TDSC 2022
#
# 핵심 아이디어:
#   odd polynomial p(x) = Σ_{k=0}^{m-1} c_k × x^{2k+1}
#   = Σ_j G^j × IS_j   (G = x^{2b}, IS_j = inner sum of b baby steps)
#
#   Baby steps:   {x^1, x^3, ..., x^{2b-1}} via sq_tree + addition chain
#   Giant step:   G = x^{2b}, powers G^2, G^4, ... via binary tree
#   Inner sums:   scalar mult + add (depth 0)
#   Tree combine: binary tree → depth ceil(log2(g))
#
# 레벨 예산 분석 (DesiloFHE lazy-rescaling: 각 non-scalar mult = 2 레벨):
#   dep(d) = 논문 Table 1 값 = BSGS 최소 depth
#   레벨 소비 = dep(d) × 2
#   bootstrap → level=10 → dep 최대 5 → degree 최대 27
#
#   deg=7  (m=4):  b=2, dep=3, 레벨 소비=6,  남은=4  ✓
#   deg=15 (m=8):  b=2, dep=4, 레벨 소비=8,  남은=2  ✓
#   deg=27 (m=14): b=4, dep=5, 레벨 소비=10, 남은=0  ✓ (경계)
#
# naive 루프 대비:
#   naive deg=27: 14회 non-scalar mult → 14 레벨 → budget=10 초과 → ✗
#   BSGS  deg=27: 10회 non-scalar mult → dep=5  → budget=10 정확 → ✓

from __future__ import annotations
import math


# ─────────────────────────────────────────────────────────────────────────────
# Baby step depth lookup (b=1..8)
# baby[i] = x^{2i+1}: sq_tree + addition chain으로 달성 가능한 최소 depth
# e.g., b=4: baby={x,x^3,x^5,x^7}
#   x^3 = x^2×x   (depth 2)
#   x^5 = x^4×x   (depth 3)
#   x^7 = x^4×x^3 (depth 3)  ← NOT x^5×x^2 (depth 4, naive보다 1 절약)
# ─────────────────────────────────────────────────────────────────────────────
_BABY_DEPTH = {1: 0, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4}


def _baby_step_depth(b: int) -> int:
    return _BABY_DEPTH.get(b, math.ceil(math.log2(max(2, 2 * b))))


def choose_bsgs_b(m: int) -> int:
    """
    최적 baby step 크기 b 선택 (depth 최소화).
    동점 시 b≈sqrt(m) 선택 (mult 균형).

    검증 (논문 Table 1):
      deg=7  (m=4):  b=2 → g=2, depth=3  ✓
      deg=15 (m=8):  b=2 → g=4, depth=4  ✓
      deg=27 (m=14): b=4 → g=4, depth=5  ✓
    """
    best_b, best_depth = 1, float('inf')
    sqrt_m = math.sqrt(m)
    for b in range(1, m + 1):
        g          = math.ceil(m / b)
        sq_depth   = math.ceil(math.log2(max(2, 2 * b)))  # depth for G=x^{2b}
        comb_depth = math.ceil(math.log2(max(1, g)))       # tree combine depth
        total      = max(_baby_step_depth(b), sq_depth) + comb_depth
        if (total < best_depth or
                (total == best_depth and abs(b - sqrt_m) < abs(best_b - sqrt_m))):
            best_depth = total
            best_b     = b
    return best_b


def build_baby_and_giant(engine, x_ct, b: int, relin_key):
    """
    Baby steps {x, x^3, ..., x^{2b-1}} 와 Giant base G=x^{2b} 계산.

    방법: Squaring tree + addition chain
      sq_tree[k] = x^{2^k}  (repeated squaring, depth=k)
      x^{2i+1} = sq_tree[p] × baby[r_idx]  (p+r=2i+1, r odd, r_idx=(r-1)//2)
      e.g., x^7 = x^4 × x^3  (depth max(2,2)+1=3, naive x^5×x^2=depth 4보다 1 depth 절약)
      G = x^{2b}: 2b가 2의 거듭제곱이면 sq_tree 직접 사용, 아니면 분해

    Returns
    -------
    baby : list[Ciphertext]  baby[i] = x^{2i+1}, i=0..b-1
    G    : Ciphertext        G = x^{2b}
    """
    if b == 1:
        # baby = {x}, G = x^2
        return [x_ct], engine.square(x_ct, relin_key)

    # ── sq_tree: x^{2^k} for k=1..K ──────────────────────────────────────
    K  = math.ceil(math.log2(max(2, 2 * b)))
    sq = {1: engine.square(x_ct, relin_key)}   # sq[1] = x^2
    for k in range(2, K + 1):
        sq[k] = engine.square(sq[k - 1], relin_key)  # sq[k] = x^{2^k}

    pow_cache = {2 ** k: sq[k] for k in range(1, K + 1)}
    pow_cache[1] = x_ct                                  # x^1 = x

    # ── Baby steps ────────────────────────────────────────────────────────
    baby = [None] * b
    baby[0] = x_ct  # x^1
    for i in range(1, b):
        odd_exp = 2 * i + 1
        placed  = False
        for p in sorted(pow_cache.keys(), reverse=True):  # 큰 p부터 탐색
            r = odd_exp - p
            if r >= 1 and r % 2 == 1:
                r_idx = (r - 1) // 2
                if 0 <= r_idx < i and baby[r_idx] is not None:
                    baby[i] = engine.multiply(pow_cache[p], baby[r_idx], relin_key)
                    placed   = True
                    break
        if not placed:
            # fallback: x^{2i+1} = x^{2i-1} × x^2 (1 depth 추가, 희귀)
            baby[i] = engine.multiply(baby[i - 1], sq[1], relin_key)

    # ── Giant step base: G = x^{2b} ───────────────────────────────────────
    two_b = 2 * b
    if two_b in pow_cache:
        G = pow_cache[two_b]                   # 2b가 2의 거듭제곱이면 직접
    else:
        G = None
        for a in sorted(pow_cache.keys(), reverse=True):
            c = two_b - a
            if c > 0 and c in pow_cache:
                G = engine.multiply(pow_cache[a], pow_cache[c], relin_key)
                break
        if G is None:
            G = engine.multiply(baby[-1], x_ct, relin_key)  # x^{2b-1} × x = x^{2b}

    return baby, G


def bsgs_tree_combine(engine, inner_sums: list, G, relin_key):
    """
    Binary tree combination:
      result = IS[0] + G×IS[1] + G^2×IS[2] + ...

    Round k: G_power = G^{2^{k-1}} 사용, 쌍 (IS_{2j}, IS_{2j+1}) 결합.
    G_power는 각 라운드 종료 시 제곱으로 갱신.

    예시 (g=4):
      Round 1 (G^1):   T0=IS0+G×IS1,  T1=IS2+G×IS3
      Round 2 (G^2):   result=T0+G^2×T1
    """
    current = list(inner_sums)
    G_power = G
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                next_level.append(
                    engine.add(current[i],
                               engine.multiply(G_power, current[i + 1], relin_key))
                )
            else:
                next_level.append(current[i])
        current = next_level
        if len(current) > 1:
            G_power = engine.square(G_power, relin_key)
    return current[0]


def eval_mcp_component(engine, ct, comp, slot_count, keypack):
    """
    단일 MCP 컴포넌트를 BSGS 알고리즘으로 평가.
    Core, Normalize, LabelProp 공용.

    수식:
      p_i(x) = Σ_{k=0}^{m-1} c_k × (x/domain_b)^{2k+1}
      where coeffs=[c_0, c_1, ..., c_{m-1}] (odd terms only)

    Parameters
    ----------
    comp : dict
        {"degree": d, "coeffs": [...], "domain_b": b, ...}

    Returns
    -------
    Ciphertext   (level = input_level - dep(degree)×2)
    """
    relin_key = keypack.relinearization_key
    coeffs    = comp["coeffs"]
    domain_b  = comp.get("domain_b", 1.0)
    m         = len(coeffs)   # number of odd terms

    # domain_b 정규화: x → x/domain_b
    if abs(domain_b - 1.0) > 1e-9:
        x_sc = engine.multiply(ct, engine.encode([1.0 / domain_b] * slot_count))
    else:
        x_sc = ct

    # degree=1 (m=1) 특수 처리
    if m == 1:
        return engine.multiply(x_sc, engine.encode([coeffs[0]] * slot_count))

    # 최적 b 선택
    b = choose_bsgs_b(m)
    g = math.ceil(m / b)

    # Baby steps + Giant base
    baby, G = build_baby_and_giant(engine, x_sc, b, relin_key)

    # Inner sums (scalar linear combination: depth 0)
    inner_sums = []
    for j in range(g):
        base_idx        = j * b
        terms_in_group  = min(b, m - base_idx)
        if terms_in_group <= 0:
            break

        is_j = engine.multiply(baby[0], engine.encode([coeffs[base_idx]] * slot_count))
        for i in range(1, terms_in_group):
            coeff_idx = base_idx + i
            is_j = engine.add(
                is_j,
                engine.multiply(baby[i], engine.encode([coeffs[coeff_idx]] * slot_count))
            )
        inner_sums.append(is_j)

    # Binary tree combination
    return bsgs_tree_combine(engine, inner_sums, G, relin_key)


def eval_mcp_full(engine, ct, components, slot_count, keypack, tag: str = "") -> object:
    """
    MCP 전체 평가: 각 컴포넌트 BSGS 평가 + 매 컴포넌트 후 bootstrap.

    Returns
    -------
    Ciphertext  level=10 (bootstrap 후)
    """
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key
    current   = ct

    for _, comp in enumerate(components):
        current = eval_mcp_component(engine, current, comp, slot_count, keypack)
        current = engine.intt(current)
        current = engine.bootstrap(current, relin_key, conj_key, boot_key)

    return current  # level=10
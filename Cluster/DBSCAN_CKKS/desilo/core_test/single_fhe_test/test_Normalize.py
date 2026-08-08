# core/ciphertext_single/normalize_packed.py
#
# ★ [#5 2026-07] Normalize 패킹 sign-eval.
#   기존: stride k=1..k_max 마다 check_neighbor_closed_interval() 를 1회씩
#         → sign-eval(MCP 5 SB + sign_bootstrap 1 + bit_cleaning) k_max 회.
#         각 암호문은 N/slot_count 만 채운 희소 상태.
#   변경: m 개 stride 의 dist_sq 를 서로 다른 슬롯 영역에 패킹 → sign-eval 1회 →
#         영역별로 되읽어(rotate+mask) adj_k 복원.  sign-eval 이 k_max → ⌈k_max/m⌉.
#
#   ★ LP 트리-max 패킹과의 결정적 차이: 여기엔 '영역 간 회전(tree rotate)'이 없다.
#     패킹 → sign-eval 1회(슬롯별) → 언패킹. 즉 lsun ARI 0.20 을 유발한
#     wrap-around 오염이 발생할 자리가 없다(cross-region 회전이 없으므로).
#     유일한 부작용은 슬롯 채움↑ → MCP 내부 SB 계수노름↑ 인데, "채움→EvalMod 절벽"
#     가설은 벤치로 이미 기각됨(Label_Propagation 헤더 r9 주석) + scaled-refresh 로 관리.
#
#   ★ 인덱스 정합성은 평문(numpy) 미러로 검증됨:
#       rotate(ct, off)[i] = ct[i-off]  → [0,N) 값을 [off, off+N) 로 배치(패킹)
#       rotate(ct, -off)[i] = ct[i+off] → 영역 t 를 [0,N) 로 복원(언패킹)
#     (검증: 4 stride, m=⌊slot/N⌋-1, 언패킹 adj == 개별 sign-eval adj 완전 일치.)
#   ※ FHE 실행 정확도(정밀도/레벨)는 sanity 및 total_neighbors 일치로 재확인 필요.

import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev
from core.ciphertext_single.cleaning import bit_cleaning


def normalize_group_size(N: int, slot_count: int) -> int:
    """패킹 가능한 그룹 크기 m = ⌊slot_count / N⌋.
    ★ LP 와 달리 tree-max 회전이 없으므로 wrap 무오염 조건 불필요 →
      슬롯 용량 전체를 쓸 수 있다(⌊slot/N⌋). 예: 32768/212=154.
    """
    return max(1, slot_count // N)


def _sign_eval_core_packed(engine, packed_ct, eps_sq, keypack, dimension,
                           mcp_path, slot_count):
    """check_neighbor_closed_interval 의 sign-eval 코어를 '패킹 암호문'에 1회 적용.
    threshold/bound 는 eps_sq·dimension 에만 의존하므로 그룹 내 모든 stride 공통.
    반환: 패킹된 adj (영역별 {0,1}), 슬롯별 동일 파이프라인.
    """
    components = load_mcp(mcp_path)
    if components[0].get("basis", "power") != "chebyshev":
        raise ValueError(f"[normalize_packed] {mcp_path} basis != chebyshev")

    mcp_delta    = components[0]["domain_a"]
    max_dist_sq  = float(dimension)
    approx_bound = max_dist_sq * 1.05
    margin_val   = mcp_delta * approx_bound

    threshold_pt = engine.encode([eps_sq + margin_val] * slot_count)
    x = engine.subtract(packed_ct, threshold_pt)

    x_min_abs = eps_sq + margin_val
    x_max_abs = max_dist_sq - (eps_sq + margin_val)
    bound     = max(x_min_abs, x_max_abs) * 1.05
    current_x = engine.multiply(x, engine.encode([1.0 / bound] * slot_count))

    # ── MCP sign 근사 (슬롯별) ──
    current_x = eval_mcp_full_chebyshev(
        engine, current_x, components, slot_count, keypack, tag="Norm-packed ")
    current_x = engine.sign_bootstrap(
        engine.intt(current_x),
        keypack.relinearization_key, keypack.conjugation_key,
        keypack.rotation_key, keypack.smallbootstrap_key,
    )
    # (-sign + 1)/2 → {1:이웃, 0:비이웃}
    result = engine.add(
        engine.multiply(current_x, engine.encode([-0.5] * slot_count)),
        engine.encode([0.5] * slot_count))
    result = bit_cleaning(engine, result, keypack, n_iters=1, slot_count=slot_count)
    return result


def check_neighbors_group_packed(
    engine: Engine,
    dist_sq_by_k: list,          # [(k, dist_sq_ct), ...]  한 그룹(≤ m개)
    eps_sq: float,
    keypack: KeyPack,
    dimension: int,
    N: int,
    slot_count: int,
    mcp_path: str = "mcp_alpha15_lp_cheb.json",
) -> dict:
    """그룹 내 여러 stride 의 dist_sq 를 패킹 → sign-eval 1회 → 언패킹.
    반환: {k: adj_k_ct}  각 adj_k 는 [0,N) 에 배치된 독립 암호문 ({0,1}).

    비용: 패킹(영역당 rotate 1) + sign-eval 1회 + 언패킹(영역당 rotate+mask 1).
          → sign_bootstrap 회수: len(group) → 1.
    """
    relin_key = keypack.relinearization_key
    ks = [k for k, _ in dist_sq_by_k]
    if len(ks) > slot_count // N:
        raise ValueError(f"[normalize_packed] 그룹 크기 {len(ks)} > 슬롯 용량 {slot_count//N}")

    # ── 1. 패킹: 영역 t=0.. 에 dist_sq_(k) 배치 ──
    #   빈 영역(그룹 미만)은 0 → 어차피 언패킹하지 않음.
    packed = None
    region_of = {}
    for t, (k, dsq_ct) in enumerate(dist_sq_by_k):
        off = t * N
        # dsq_ct 는 [0,N) 에 값이 있다고 가정(Server 루프의 dist_sq_k 그대로).
        # [0,N) 밖을 0 으로 만들고(mask), [off,off+N) 로 회전 배치.
        masked = engine.multiply(
            dsq_ct, engine.encode([1.0] * N + [0.0] * (slot_count - N)))
        placed = (masked if off == 0
                  else engine.rotate(masked, keypack.rotation_key, off))
        packed = placed if packed is None else engine.add(packed, placed)
        region_of[k] = t

    # ── 2. sign-eval 1회 (슬롯별) ──
    packed_adj = _sign_eval_core_packed(
        engine, packed, eps_sq, keypack, dimension, mcp_path, slot_count)

    # ── 3. 언패킹: 영역 t 를 [0,N) 로 복원 ──
    mask0 = engine.encode([1.0] * N + [0.0] * (slot_count - N))
    out = {}
    for k in ks:
        t = region_of[k]
        off = t * N
        reg = (packed_adj if off == 0
               else engine.rotate(packed_adj, keypack.rotation_key, -off))
        out[k] = engine.multiply(reg, mask0)   # [0,N) 만 남김
    return out


# ── Server_main.py Step 1 통합 예시 (교체 스케치) ─────────────────────────────
#
#   m = normalize_group_size(N, engine.slot_count)      # 예: hepta 154 ≥ k_max=70 → 1그룹
#   groups = [list(range(s, min(s + m, _k_upper + 1))) for s in range(1, _k_upper + 1, m)]
#   for g in groups:
#       dist_sq_by_k = []
#       for k in g:
#           dist_sq_k = None
#           for d in range(dim):
#               rot = fhe_circular_shift(engine, encrypted_columns[d], k, N, keypack)
#               diff = engine.subtract(encrypted_columns[d], rot)
#               sq   = engine.square(diff, keypack.relinearization_key)
#               dist_sq_k = sq if dist_sq_k is None else engine.add(dist_sq_k, sq)
#           dist_sq_by_k.append((k, dist_sq_k))
#       adj_group = check_neighbors_group_packed(
#           engine, dist_sq_by_k, eps**2, keypack, dim, N, engine.slot_count,
#           mcp_path="mcp_alpha15_lp_cheb.json")
#       for k in g:
#           adj_k = adj_group[k]
#           adj_k_list.append(adj_k)
#           # total_neighbors 누적(기존과 동일: adj_k + rotate(adj_k, N-k) 등)
#
#   ※ dist_sq 제곱(dim×k_max, bootstrap 0)은 여전히 stride별이지만 값이 싸다.
#     비싼 sign-eval(부트스트랩 다수)만 그룹당 1회로 접힌다.
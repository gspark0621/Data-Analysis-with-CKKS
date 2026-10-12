# core/ciphertext_single/GridRounds.py
#
# ── 클라이언트 측 grid 기반 LP Phase-1 라운드 수 산출 (2026-10 최종안) ─────────
#
#   n_rounds = max(1, max_g ⌈(Λ_g + s·√d) / ε_n⌉)          (상한 F 로 자르지 않음)
#
#     s    = ε_n / 2                    : 셀 한 변 (반열린 구간 [k·s, (k+1)·s))
#     Λ_g  = 셀 묶음 g 의 측지 지름      : 셀 그래프(이웃 셀, 가중치 = 셀 중심 거리)에서
#                                         가장 먼 두 셀 사이 최단 거리 — 정확값
#     s√d  = 양 끝 점 → 각자 셀 중심 거리(각 s√d/2)의 합
#
# ── 계산 절차 ────────────────────────────────────────────────────────────────
#   ① 셀 번호      cell(p) = ⌊p / s⌋                                   O(N·d)
#   ② 저장         점이 있는 셀만 dict 에 저장                          O(N) 기대
#   ③ 필터         자기 셀 + 이웃 셀 점 수 < min_pts 인 셀 제외          O(C·5^d·d)
#                  (이웃 셀 = 각 축 ±2칸 중 Σ max(|o_i|−1,0)² ≤ 4; d ≤ 3 이면 5^d−1 칸 전부)
#   ④ 묶기         남은 셀을 이웃 셀끼리 BFS → 셀 묶음                    O(C·5^d·d)
#   ⑤ 셀 그래프    묶음마다 이웃 셀 쌍을 간선, 가중치 = s·‖offset‖        O(C·5^d·d)
#   ⑥ 측지 지름    가지치기 Dijkstra (BoundingDiameters, Takes & Kosters)
#                    - 셀마다 이심률 하한 eL·상한 eU 를 삼각부등식으로 갱신
#                        eL(w) ← max(eL, d(v,w), e−d(v,w)),  eU(w) ← min(eU, e+d(v,w))
#                    - r(DL) == r(DU) 이면 정지 (라운드는 정수)          [정리 S1]
#                    - eU ≤ DL 이거나 eL == eU 인 셀은 후보에서 제외      [정리 S2]
#                    - r(DU_g) ≤ 지금까지 최댓값 인 묶음은 건너뜀          [정리 S3]
#                  1회 O(C_g·5^d·log C_g), 최악 C_g 회 → 최악 O(C²·5^d·log C)
#   ⑦ n_rounds = max(1, max_g r(Λ_g)),  r(x) = ⌈(x + s√d)/ε_n⌉
#
#   · 점 사이 거리를 ε 와 비교하는 연산(DBSCAN 범위 질의)은 하지 않는다.
#   · 서버 내부 파라미터(α, margin)는 쓰지 않는다.
#
# ── 보장 (증명: 2026-10 증명 시리즈 1~7편) ───────────────────────────────────
#   G1   평문 core 성분은 셀 묶음 하나에 포함
#   주정리  가정 A(라벨은 grid 영역 안 최단 경로를 따라 hop 마다 ε_n 전진) 아래에서,
#           FHE 가 평문과 똑같이 재현한 모든 클러스터는 n_rounds 안에 전파 완료
#   S1~S3  가지치기 결과 = 모든 셀에서 Dijkstra 를 돌린 결과 (정확)
#
# ── 누출 ─────────────────────────────────────────────────────────────────────
#   n_rounds 정수 하나가 서버에 드러난다 (가장 큰 셀 묶음의 측지 지름 / ε_n 수준).
# ─────────────────────────────────────────────────────────────────────────────

import math
import heapq
import itertools
from collections import deque, defaultdict
from time import time

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
    _HAS_SCIPY = True
except Exception:          # scipy 미설치 환경: 순수 파이썬 Dijkstra 로 대체 (결과 동일)
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# 셀 이웃
# ─────────────────────────────────────────────────────────────────────────────
def neighbor_offsets(d: int):
    """셀 한 변 s = ε_n/2 일 때의 이웃 셀 offset (원점 제외).

    두 셀(반열린 구간)의 거리 하한 = s · sqrt(Σ max(|o_i|−1, 0)²).
    거리 ≤ ε_n = 2s 인 점 쌍이 있을 수 있는 셀만 이웃: Σ max(|o_i|−1,0)² ≤ 4.
    |o_i| ≥ 3 이면 그 축 하나로 거리 > 2s (반열린 구간이라 엄격) → 범위 {−2..2}^d.
    d ≤ 3 에서는 조건이 항상 참이라 5^d − 1 칸 전부 (2D 24, 3D 124).
    """
    offs = []
    for o in itertools.product(range(-2, 3), repeat=d):
        if not any(o):
            continue
        if sum(max(abs(v) - 1, 0) ** 2 for v in o) <= 4:
            offs.append(o)
    return offs


# ─────────────────────────────────────────────────────────────────────────────
# 셀 그래프 위 단일 출발 Dijkstra
# ─────────────────────────────────────────────────────────────────────────────
class _CellGraph:
    """묶음 하나의 셀 그래프. 간선 = 이웃 셀, 가중치 = s·‖offset‖."""

    def __init__(self, cells, offs, s):
        self.n = len(cells)
        pos = {c: i for i, c in enumerate(cells)}
        rows, cols, w = [], [], []
        for c in cells:
            i = pos[c]
            for o in offs:
                nb = tuple(a + b for a, b in zip(c, o))
                j = pos.get(nb)
                if j is not None:
                    rows.append(i); cols.append(j)
                    w.append(s * math.sqrt(sum(v * v for v in o)))
        self.pos = pos
        if _HAS_SCIPY:
            self.A = csr_matrix((w, (rows, cols)), shape=(self.n, self.n))
        else:
            self.adj = [[] for _ in range(self.n)]
            for i, j, ww in zip(rows, cols, w):
                self.adj[i].append((j, ww))

    def sssp(self, src: int) -> np.ndarray:
        if _HAS_SCIPY:
            return _sp_dijkstra(self.A, directed=False, indices=src)
        dist = np.full(self.n, np.inf); dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > dist[u]:
                continue
            for v, ww in self.adj[u]:
                nd = du + ww
                if nd < dist[v]:
                    dist[v] = nd; heapq.heappush(pq, (nd, v))
        return dist


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def compute_grid_rounds(norm_pts, eps_norm: float, min_pts: int,
                        pc1=None, verbose: bool = True):
    """
    클라이언트가 정규화 좌표로 LP Phase-1 라운드 수를 산출한다.

    Parameters
    ----------
    norm_pts : (N, d) 정규화 좌표 (기존 정규화 결과 그대로)
    eps_norm : ε_n = ε / scale
    min_pts  : DBSCAN min_pts (자기 자신 포함)
    pc1      : (N,) 정규화 PC1 사영값 (norm_pts 와 같은 순서). 각 묶음에서 첫 Dijkstra 를
               PC1 최대 점(전파 소스 근처)의 셀에서 시작하는 데만 쓴다 — 속도에만 영향,
               결과(정확한 라운드)에는 영향 없음. None 이면 각 묶음의 첫 셀에서 시작.

    Returns
    -------
    n_rounds : int
    info     : dict (n_cells, n_cells_kept, n_groups, largest_group, lambda_max,
                     n_dijkstra, n_dijkstra_allpairs, time_ms, ...)
    """
    t0 = time()
    P = np.asarray(norm_pts, dtype=np.float64)
    N, d = P.shape
    s = eps_norm / 2.0                                   # 셀 한 변 = ε_n / 2
    slack = s * math.sqrt(d)                             # 양 끝 점 → 셀 중심

    def r(x: float) -> int:                              # 라운드 함수 (단조 증가)
        return math.ceil((x + slack) / eps_norm)

    # ① 셀 번호  ② 저장
    keys = np.floor(P / s).astype(np.int64)
    cell = defaultdict(list)
    for i, k in enumerate(map(tuple, keys)):
        cell[k].append(i)
    cnt = {k: len(v) for k, v in cell.items()}

    # ③ 필터
    offs = neighbor_offsets(d)

    def nbrs(k):
        return (tuple(a + b for a, b in zip(k, o)) for o in offs)

    kept = set()
    for k in cell:
        tot = cnt[k]
        if tot < min_pts:
            for nb in nbrs(k):
                tot += cnt.get(nb, 0)
                if tot >= min_pts:
                    break
        if tot >= min_pts:
            kept.add(k)

    # ④ 묶기 (BFS)
    groups, seen = [], set()
    for k0 in kept:
        if k0 in seen:
            continue
        seen.add(k0); q = deque([k0]); mem = [k0]
        while q:
            k = q.popleft()
            for nb in nbrs(k):
                if nb in kept and nb not in seen:
                    seen.add(nb); mem.append(nb); q.append(nb)
        groups.append(mem)

    # ⑤⑥ 셀 그래프 + 가지치기 Dijkstra
    n_dij = 0
    R_best = 0
    lam_best = 0.0
    items = []
    for mem in groups:
        if len(mem) == 1:                                 # 셀 하나: Λ_g = 0
            R_best = max(R_best, r(0.0))
            continue
        G = _CellGraph(mem, offs, s)
        if pc1 is not None:
            pts = [i for c in mem for i in cell[c]]
            src_pt = max(pts, key=lambda i: pc1[i])
            src = G.pos[tuple(keys[src_pt])]
        else:
            src = 0
        d0 = G.sssp(src); n_dij += 1
        e0 = float(d0.max())                              # 묶음은 연결 → 모두 유한
        items.append((r(2.0 * e0), G, src, d0, e0))       # Λ_g ≤ 2·ecc(src)

    items.sort(key=lambda t: -t[0])                       # 상한 큰 묶음부터   [S3]
    for ub_round, G, src, d0, e0 in items:
        if ub_round <= R_best:                            # 최댓값을 바꿀 수 없음 [S3]
            continue
        n = G.n
        eL = np.zeros(n)
        eU = np.full(n, np.inf)
        cand = np.ones(n, dtype=bool)

        def update(dist, e, v):
            np.maximum(eL, np.maximum(dist, e - dist), out=eL)
            np.minimum(eU, e + dist, out=eU)
            eL[v] = eU[v] = e

        update(d0, e0, src)
        turn = 0
        while True:
            DL = float(eL.max()); DU = float(eU.max())
            rL, rU = r(DL), r(DU)
            if rU <= R_best:                              # 최댓값을 바꿀 수 없음 [S3]
                break
            if rL == rU:                                  # 정수 라운드 확정      [S1]
                R_best, lam_best = rL, max(lam_best, DL)
                break
            cand &= ~(eU <= DL)                           # 지름을 올릴 수 없는 셀 [S2]
            cand &= ~(eL == eU)                           # 이심률을 이미 아는 셀  [S2]
            if not cand.any():                            # 후보 소진 → DL = Λ_g   [S2]
                R_best, lam_best = max(R_best, rL), max(lam_best, DL)
                break
            idx = np.flatnonzero(cand)
            v = int(idx[np.argmax(eU[idx])] if turn % 2 == 0 else idx[np.argmin(eL[idx])])
            turn += 1
            dist = G.sssp(v); n_dij += 1
            update(dist, float(dist.max()), v)

    n_rounds = max(1, R_best)
    info = dict(n_rounds=n_rounds, cell_side=s, n_cells=len(cell), n_cells_kept=len(kept),
                n_groups=len(groups), largest_group=max((len(m) for m in groups), default=0),
                lambda_max=lam_best, n_dijkstra=n_dij,
                n_dijkstra_allpairs=sum(len(m) for m in groups if len(m) > 1),
                F_public=math.ceil(1.0 / eps_norm), time_ms=(time() - t0) * 1000.0)
    if verbose:
        print(f"[Client/grid] cell=ε_n/2  cells={info['n_cells']} (kept {info['n_cells_kept']})  "
              f"groups={info['n_groups']} (max {info['largest_group']} cells)  "
              f"Dijkstra={n_dij}/{info['n_dijkstra_allpairs']}  "
              f"→ n_rounds={n_rounds}  (참고 ⌈1/ε_n⌉={info['F_public']}, {info['time_ms']:.1f} ms)")
    return n_rounds, info


def compute_grid_rounds_allpairs(norm_pts, eps_norm: float, min_pts: int):
    """검증용: 모든 셀에서 Dijkstra 를 돌린 정확값. compute_grid_rounds 와 같아야 한다."""
    P = np.asarray(norm_pts, dtype=np.float64); N, d = P.shape
    s = eps_norm / 2.0; slack = s * math.sqrt(d)
    keys = np.floor(P / s).astype(np.int64); cell = defaultdict(list)
    for i, k in enumerate(map(tuple, keys)):
        cell[k].append(i)
    cnt = {k: len(v) for k, v in cell.items()}; offs = neighbor_offsets(d)
    nbrs = lambda k: (tuple(a + b for a, b in zip(k, o)) for o in offs)
    kept = {k for k in cell if cnt[k] + sum(cnt.get(nb, 0) for nb in nbrs(k)) >= min_pts}
    best, seen = 0.0, set()
    for k0 in kept:
        if k0 in seen:
            continue
        seen.add(k0); q = deque([k0]); mem = [k0]
        while q:
            k = q.popleft()
            for nb in nbrs(k):
                if nb in kept and nb not in seen:
                    seen.add(nb); mem.append(nb); q.append(nb)
        if len(mem) > 1:
            G = _CellGraph(mem, offs, s)
            best = max(best, max(float(G.sssp(v).max()) for v in range(G.n)))
    return max(1, math.ceil((best + slack) / eps_norm))
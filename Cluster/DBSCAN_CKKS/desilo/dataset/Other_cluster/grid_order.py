#!/usr/bin/env python3
"""
그리드 기반 클라이언트 순서 — 평문 프로토타입 & 검증

목적: 클라이언트 전처리를 O(N log N)으로 낮추면서 (pairwise 없이),
서버 stride 전파로 평문 DBSCAN과 동일 파티션(ARI=1)이 나오는지 검증.

클라이언트가 하는 일 (전부 O(N log N) 이하, pairwise 없음):
  1) min-max 정규화                    O(N·d)
  2) 셀 배정 cell = floor(x/(eps/√d))    O(N·d)
  3) 셀 coord-lex 정렬                   O(N log N)
  4) (암호화) — 프로토타입에선 생략

측정:
  - k_max_grid: 정렬 순서에서 eps-간선(전 그래프)의 최대 circular stride
    → 이게 작아야 서버가 값싸다. ball tree DFS와 대조.
  - n_rounds 상한: 셀 히스토그램 기반 클로즈드폼 (pairwise 없이 O(N))
  - ARI: 그리드 순서 stride 전파 결과 vs 평문 DBSCAN
  - 서버 fhe_max 추정: active stride 시뮬 (스케줄 없는 보수 라운드)
"""
import numpy as np
from collections import defaultdict, deque

# ─────────────────────────────────────────────────────────────
def load_arff(fp):
    pts, lab, data = [], [], False
    for line in open(fp):
        s = line.strip()
        if not s or s.startswith('%'): continue
        if s.lower().startswith('@data'): data = True; continue
        if s.lower().startswith('@'): continue
        if data:
            v = s.replace(',', ' ').split()
            pts.append([float(x) for x in v[:-1]])
            lab.append(v[-1])
    return np.array(pts), np.array(lab)

def plaintext_dbscan(P, eps, min_pts):
    """참조용 평문 DBSCAN (O(N²) 구현이지만 정답 산출용)."""
    N = len(P)
    D2 = ((P[:,None,:]-P[None,:,:])**2).sum(2)
    nbr = D2 <= eps*eps
    core = nbr.sum(1) >= min_pts
    labels = -np.ones(N, int)
    cid = 0
    for i in range(N):
        if not core[i] or labels[i] != -1: continue
        labels[i] = cid
        stack = deque([i])
        while stack:
            u = stack.pop()
            for v in np.where(nbr[u])[0]:
                if labels[v] == -1:
                    labels[v] = cid
                    if core[v]: stack.append(v)
                elif labels[v] != cid and core[v] and not core[u]:
                    pass
        cid += 1
    return labels, core

def ari(a, b):
    from math import comb
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    contab = defaultdict(int)
    ca, cb = defaultdict(int), defaultdict(int)
    for x, y in zip(a, b):
        contab[(x,y)] += 1; ca[x] += 1; cb[y] += 1
    sum_ij = sum(comb(v,2) for v in contab.values())
    sum_a = sum(comb(v,2) for v in ca.values())
    sum_b = sum(comb(v,2) for v in cb.values())
    exp = sum_a*sum_b/comb(n,2)
    mx = (sum_a+sum_b)/2
    return (sum_ij-exp)/(mx-exp) if mx != exp else 1.0

# ── 클라이언트: 그리드 순서 (O(N log N), pairwise 없음) ──────────
def grid_order(norm, eps_norm, d):
    """coord-lex 셀 정렬. 반환: order(heap→orig), cell_of_heap, side."""
    side = eps_norm / np.sqrt(d)               # 셀 한 변 = ε/√d
    cell = np.floor(norm / side).astype(int)   # O(N·d)
    # coord-lex: 셀 좌표 사전식 정렬 → 같은 셀 연속, 인접 셀 근접
    keys = [cell[:, j] for j in range(d-1, -1, -1)]
    order = np.lexsort(keys)                    # O(N log N)
    cell_sorted = cell[order]
    # 셀 인덱스(연속 정수) 부여
    uniq, inv = np.unique(cell_sorted, axis=0, return_inverse=True)
    return order, cell_sorted, inv, side, uniq

# ── n_rounds 클로즈드폼 상한 (셀 인접 그래프, pairwise 없음) ──────
def rounds_upper_from_cells(uniq_cells, d):
    """occupied 셀들의 인접(3^d 이웃) 그래프 지름 → 라운드 상한.
    비용: O(#cells · 3^d) — pairwise 아님. #cells ≤ N."""
    cellset = {tuple(c): i for i, c in enumerate(uniq_cells)}
    import itertools
    offs = [o for o in itertools.product([-1,0,1], repeat=d) if any(o)]
    M = len(uniq_cells)
    adj = [[] for _ in range(M)]
    for c, i in cellset.items():
        for o in offs:
            nb = tuple(np.array(c)+o)
            if nb in cellset:
                adj[i].append(cellset[nb])
    # 셀 그래프 성분별 BFS 지름(2×BFS 근사)
    seen = [False]*M
    diam = 0
    for s in range(M):
        if seen[s]: continue
        comp = []
        dq = deque([s]); seen[s]=True
        while dq:
            u=dq.popleft(); comp.append(u)
            for v in adj[u]:
                if not seen[v]: seen[v]=True; dq.append(v)
        # 2×BFS: 임의점→최원점 a, a→최원점 지름
        def bfs_far(src):
            dist={src:0}; dq=deque([src]); far=src
            while dq:
                u=dq.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v]=dist[u]+1
                        if dist[v]>dist[far]: far=v
                        dq.append(v)
            return far, dist[far]
        a,_ = bfs_far(s); b,dd = bfs_far(a)
        diam = max(diam, dd)
    return diam   # 셀 단위 지름 (홉)

# ── 서버 시뮬: 그리드 순서 stride 전파 (평문 미러) ────────────────
def simulate_grid_propagation(norm, order, eps_norm, min_pts, N, d,
                              n_rounds):
    """그리드 순서에서 stride 전파 → FHE 서버가 낼 파티션 (평문 미러).
    측정: active stride 집합, 최대 stride, 실제 수렴 라운드, fhe_max 추정."""
    P = norm[order]
    eps2 = eps_norm*eps_norm
    D2 = ((P[:,None,:]-P[None,:,:])**2).sum(2)   # 시뮬만 — 서버가 암호로 함
    within = D2 <= eps2
    core = within.sum(1) >= min_pts
    idx = np.arange(N)
    # active core-core stride
    S_cc = []
    kmax_grid = 0
    for k in range(1, N//2+1):
        j = (idx+k)%N
        m = within[idx,j] & core & core[j]
        if m.any():
            S_cc.append(k); kmax_grid = k
    # 전파 (위치 라벨 1..N, max)
    lab = np.arange(1, N+1, dtype=float) * core
    def apply_round(lab):
        changed=False
        for ks in (S_cc, S_cc[::-1]):
            for k in ks:
                j=(idx+k)%N
                m = within[idx,j] & core & core[j]
                # fwd
                cand=np.zeros(N); cand[m]=lab[j[m]]
                new=np.maximum(lab,cand); changed|=(new>lab+1e-9).any(); lab=new
                # bwd
                s=(idx-k)%N
                mb = within[idx,s] & core & core[s]
                cand=np.zeros(N); cand[mb]=lab[s[mb]]
                new=np.maximum(lab,cand); changed|=(new>lab+1e-9).any(); lab=new
        return lab, changed
    r_star=0
    for r in range(256):
        lab, ch = apply_round(lab)
        if ch: r_star=r+1
        else: break
    # border 1-hop
    blab = lab.copy()
    for k in S_cc:
        for sgn in (k, -k):
            j=(idx+sgn)%N
            m = within[idx,j] & (~core) & core[j]
            cand=np.zeros(N); cand[m]=lab[j[m]]
            blab=np.where((cand>blab)&(~core), cand, blab)
    # 원순서 복원 후 라벨 정규화
    final = np.where(core, lab, blab)
    inv=np.empty(N,int); inv[order]=np.arange(N)
    return final[inv], core[inv], S_cc, kmax_grid, r_star

# ─────────────────────────────────────────────────────────────
def run(name, path, eps, min_pts):
    P, truth = load_arff(path)
    N, d = P.shape
    g_min, g_max = P.min(), P.max()
    norm = (P-g_min)/(g_max-g_min)
    eps_n = eps/(g_max-g_min)

    ref_lab, ref_core = plaintext_dbscan(norm, eps_n, min_pts)

    order, cell_sorted, cell_inv, side, uniq = grid_order(norm, eps_n, d)
    cell_diam = rounds_upper_from_cells(uniq, d)
    # 라운드 상한: 셀 지름 홉 × (셀당 최대 stride 없이 보수). ecc식 미러.
    n_rounds_ub = int(np.ceil(cell_diam/2.0)+1) if cell_diam>0 else 2

    grid_lab, grid_core, S_cc, kmax_grid, r_star = \
        simulate_grid_propagation(norm, order, eps_n, min_pts, N, d,
                                  n_rounds_ub)

    a_ref = ari(ref_lab, grid_lab)
    # fhe_max 추정: 스케줄 없는 보수 = (2·n_rounds+2)·|active용 전 stride 1..kmax|·2
    #   (현행 A 시나리오와 동일 공식, k=kmax_grid)
    fhe_A_grid = (2*n_rounds_ub+2)*kmax_grid*2
    fhe_D_grid = 2*(r_star+1)*len(S_cc)*2   # 참고: active만+정확라운드(스케줄상당)

    print(f"\n{'='*66}\n{name}: N={N} d={d} eps_n={eps_n:.4f} min_pts={min_pts}")
    print(f"  [클라이언트] 셀변={side:.4f} occupied셀={len(uniq)}개 "
          f"(정렬 O(N log N), pairwise 없음)")
    print(f"  [셀지름] {cell_diam}홉 → n_rounds 상한=⌈{cell_diam}/2⌉+1={n_rounds_ub} "
          f"(셀 히스토그램만, O(N))")
    print(f"  [그리드 stride] |S_cc|={len(S_cc)} kmax_grid={kmax_grid} "
          f"(=N/2? {kmax_grid==N//2}) 실측수렴 r*={r_star}")
    print(f"  [ARI] 평문 vs 그리드전파 = {a_ref*100:.2f}")
    ref_k = len(set(ref_lab[ref_lab>=0])); grid_k=len(set(grid_lab[grid_lab>0]))
    print(f"  [클러스터수] 평문={ref_k} 그리드={grid_k} core(평문/그리드)="
          f"{ref_core.sum()}/{grid_core.sum()}")
    print(f"  [서버 fhe_max] 그리드-보수(A식)={fhe_A_grid}  "
          f"그리드-active(D상당)={fhe_D_grid}")
    return dict(name=name, N=N, kmax_grid=kmax_grid, S_cc=len(S_cc),
                r_star=r_star, n_rounds_ub=n_rounds_ub,
                ari=a_ref, fhe_A=fhe_A_grid, fhe_D=fhe_D_grid,
                cells=len(uniq), cell_diam=cell_diam)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        run(sys.argv[1].split('/')[-1], sys.argv[1],
            float(sys.argv[2]), int(sys.argv[3]))
    else:
        # 합성 데이터로 로직 검증
        for nm, pth, e, mp in [("blobs","synth_blobs.arff",0.55,4),
                                ("moons","synth_moons.arff",0.16,4)]:
            run(nm, pth, e, mp)
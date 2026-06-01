#!/usr/bin/env python3
# tools/verify_convergence.py
"""
[2026-05c 작업 C] 평문 라벨 전파 수렴 검증 도구 (FHE 불필요)

목적:
  새 데이터셋(Lsun, iris, breast_cancer 등)에서 FHE 전체(수 시간)를 돌리지 않고,
  평문으로 라벨 전파를 모사하여:
    1. 필요 round 수 (ARI>=target 도달) 측정
    2. 작업 B 공식 (n_rounds = ⌈log₂N⌉) 이 충분한지 확인
    3. mask=1.0 (작업 A 효과) 가정 하의 이상적 수렴성 확인

배경 (검증 완료된 사실):
  - 라벨 전파는 client Ball Tree ordering + adj(eps-이웃) 그래프 위에서
    max 연산 반복으로 라벨을 전파. FHE 연산을 평문 max로 1:1 모사 가능.
  - mask=1.0 이면 round 늘려도 라벨 damping 無 (작업 A가 이를 보장).
  - 필요 pass(=2*round) 최댓값은 검증 데이터에서 16 (circles).
    2*log₂N 이 이를 커버 (N=200→16, N=400→18).

사용법:
  python verify_convergence.py
  또는 evaluate_dataset(X, y, name) 직접 호출.

주의:
  - 이 도구는 '이상적 sign + mask=1.0' 을 가정 (작업 A 적용 후 상태).
    실제 FHE는 sign 근사/잔여 noise가 있으나, 작업 A로 mask=1.0이면
    이 평문 결과가 FHE 결과의 좋은 상한 근사가 됨.
  - ordering이 클러스터를 흩어놓는 문제(Q5)는 별개. 이 도구는 주어진
    ordering 하에서 'pass를 늘리면 수렴하는가'만 측정.
"""

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────
# Client Ball Tree DFS in-order (Client_main.build_ball_tree_order 복제)
# ─────────────────────────────────────────────────────────────────
def build_ball_tree_order(pts: np.ndarray):
    pts_arr = np.array(pts, dtype=np.float64)
    N = len(pts_arr)
    order = np.empty(N, dtype=int)

    def _b(idx, start):
        if len(idx) == 0:
            return
        if len(idx) == 1:
            order[start] = idx[0]
            return
        c = pts_arr[idx]
        cen = c.mean(axis=0)
        p1 = c[int(np.argmax(np.einsum('ij,ij->i', c - cen, c - cen)))]
        p2 = c[int(np.argmax(np.einsum('ij,ij->i', c - p1, c - p1)))]
        ax = p2 - p1
        n = float(np.dot(ax, ax)) ** 0.5
        po = (np.arange(len(idx)) if n < 1e-12
              else np.argsort((c - p1) @ (ax / n), kind='stable'))
        m = len(idx) // 2
        _b(idx[po[:m]], start)
        order[start + m] = idx[po[m]]
        _b(idx[po[m + 1:]], start + m + 1)

    _b(np.arange(N), 0)
    inv = np.empty(N, dtype=int)
    inv[order] = np.arange(N)
    return order, inv


def compute_kmax(pts: np.ndarray, eps: float, N: int) -> int:
    """Client_main.compute_kmax_from_ball_structure 복제."""
    pts_arr = np.array(pts, dtype=np.float64)
    k_max = 0

    def _a(idx):
        nonlocal k_max
        if len(idx) <= 1:
            return ((pts_arr[idx[0]].copy(), 0.0) if len(idx) == 1
                    else (np.zeros(pts_arr.shape[1]), 0.0))
        c = pts_arr[idx]
        cen = c.mean(axis=0)
        rad = float(np.max(np.linalg.norm(c - cen, axis=1)))
        p1 = c[int(np.argmax(np.einsum('ij,ij->i', c - cen, c - cen)))]
        p2 = c[int(np.argmax(np.einsum('ij,ij->i', c - p1, c - p1)))]
        ax = p2 - p1
        n = float(np.dot(ax, ax)) ** 0.5
        po = (np.arange(len(idx)) if n < 1e-12
              else np.argsort((c - p1) @ (ax / n), kind='stable'))
        m = len(idx) // 2
        nL, nR = m, len(idx) - m - 1
        cL, rL = _a(idx[po[:m]])
        cR, rR = _a(idx[po[m + 1:]])
        if nL > 0 and nR > 0:
            gap = float(np.linalg.norm(cL - cR)) - rL - rR
            if gap < eps:
                cr = nL + nR
                cand = min(cr, N - cr)
                if cand > k_max:
                    k_max = cand
        return cen, rad

    _a(np.arange(N))
    return min(max(k_max, 1), N // 2)


# ─────────────────────────────────────────────────────────────────
# 평문 라벨 전파 모사 (FHE fhe_kd_dense_propagation 1:1 대응)
# ─────────────────────────────────────────────────────────────────
def _build_adj(pts, eps, N):
    idx = np.arange(N)
    adj = []
    for k in range(1, N // 2 + 1):
        j = (idx + k) % N
        adj.append((np.linalg.norm(pts - pts[j], axis=1) <= eps).astype(float))
    return adj


def _shift(arr, k, N):
    return np.array([arr[(i + k) % N] for i in range(N)])


def _fhe_max(u, v, scale):
    d = u - v
    return ((u + v) + d * np.sign(d / scale)) / 2.0   # 이상적 sign


def _propagate_core(adj, lab, mask, k, N, scale):
    ak = adj[k - 1]
    nb = ak * _shift(mask, k, N) * _shift(lab, k, N)
    lab = _fhe_max(lab, nb, scale)
    lab = lab * mask
    if 2 * k < N:
        aNk = _shift(ak, N - k, N)
        nb2 = aNk * _shift(mask, N - k, N) * _shift(lab, N - k, N)
        lab = _fhe_max(lab, nb2, scale)
        lab = lab * mask
    return lab


def simulate_kd_dense(adj, N, k_max, n_rounds, mask_val=1.0):
    """fhe_kd_dense_propagation 평문 모사. mask_val=1.0 = 작업 A 적용 상태."""
    scale = float(N)
    mask = np.full(N, mask_val)
    lab = mask * np.arange(1, N + 1, dtype=float)
    passes = [list(range(1, k_max + 1)), list(range(k_max, 0, -1))]
    for _ in range(n_rounds):
        for k_order in passes:
            for k in k_order:
                lab = _propagate_core(adj, lab, mask, k, N, scale)
    return lab


# ─────────────────────────────────────────────────────────────────
# 데이터셋 평가
# ─────────────────────────────────────────────────────────────────
def evaluate_dataset(X, y, name="dataset", target=0.9, cap_rounds=40,
                     mask_val=1.0, verbose=True):
    """
    한 데이터셋에 대해 필요 round를 측정하고 log₂N 공식 적합성 판정.

    Returns: dict(name, N, kmax, needed_rounds, log2N_rounds, formula_ok, plain_ari)
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score

    X = np.array(X, dtype=np.float64)
    Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    N = len(Xn)

    # 최적 eps 탐색 (평문 DBSCAN 기준)
    best = (-1.0, None)
    for ne in np.linspace(0.02, 0.6, 100):
        lab = DBSCAN(eps=ne, min_samples=4).fit(Xn).labels_
        ari = adjusted_rand_score(y, lab)
        if ari > best[0]:
            best = (ari, ne)
    plain_ari, eps = best

    order, inv = build_ball_tree_order(Xn)
    Xh, yh = Xn[order], y[order]
    kmax = compute_kmax(Xn, eps, N)
    adj = _build_adj(Xh, eps, N)

    log2N = math.ceil(math.log2(N))

    # 필요 round 측정
    needed = None
    for r in range(1, cap_rounds + 1):
        lab = simulate_kd_dense(adj, N, kmax, r, mask_val=mask_val)
        rl = np.where(np.round(lab).astype(int) <= 0, -1, np.round(lab).astype(int))
        if adjusted_rand_score(yh, rl) >= target:
            needed = r
            break

    formula_ok = (needed is not None) and (log2N >= needed)

    if verbose:
        ns = str(needed) if needed else f">{cap_rounds}"
        status = "✓ 충분" if formula_ok else ("✗ 부족" if needed else "✗ 미수렴")
        print(f"[{name:>14}] N={N:>4} dim={Xn.shape[1]:>2} 평문ARI={plain_ari*100:>5.1f} "
              f"| k_max={kmax:>4} | 필요round={ns:>4} | log₂N={log2N} | 공식 {status}")

    return dict(name=name, N=N, kmax=kmax, needed_rounds=needed,
                log2N_rounds=log2N, formula_ok=formula_ok, plain_ari=plain_ari)


def main():
    """내장 데이터셋으로 작업 B 공식(log₂N round) 적합성 일괄 검증."""
    from sklearn.datasets import (load_iris, load_breast_cancer,
                                   make_moons, make_circles, make_blobs)

    print("=" * 92)
    print("[작업 C] 평문 수렴 검증 — n_rounds = ⌈log₂N⌉ 공식이 ARI≥0.9를 커버하는가?")
    print("  (mask=1.0 가정 = 작업 A 적용 후 상태)")
    print("=" * 92)

    results = []
    iris = load_iris()
    results.append(evaluate_dataset(iris.data, iris.target, "iris"))
    bc = load_breast_cancer()
    results.append(evaluate_dataset(bc.data, bc.target, "breast_cancer"))
    Xm, ym = make_moons(n_samples=200, noise=0.05, random_state=0)
    results.append(evaluate_dataset(Xm, ym, "two_moons"))
    Xc, yc = make_circles(n_samples=200, noise=0.04, factor=0.5, random_state=0)
    results.append(evaluate_dataset(Xc, yc, "circles"))
    Xb, yb = make_blobs(n_samples=210, centers=3, cluster_std=0.6, random_state=0)
    results.append(evaluate_dataset(Xb, yb, "blobs3"))
    Xm2, ym2 = make_moons(n_samples=400, noise=0.05, random_state=1)
    results.append(evaluate_dataset(Xm2, ym2, "moons_big"))

    print("-" * 92)
    ok = sum(1 for r in results if r['formula_ok'])
    solvable = sum(1 for r in results if r['needed_rounds'] is not None)
    print(f"공식 적합: {ok}/{len(results)}  |  수렴가능(평문 의미있는 ARI): {solvable}/{len(results)}")
    print("주의: 평문ARI가 낮은 데이터(iris/breast_cancer)는 DBSCAN 자체가 잘 안 풀리는 경우.")
    print("      이 경우 라벨전파 수렴과 무관하게 ARI 낮음 (데이터/eps 문제, 알고리즘 문제 아님).")


if __name__ == "__main__":
    main()
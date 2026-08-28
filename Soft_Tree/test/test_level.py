# soft_tree/test/test_level.py
#
# 레벨 진단 — forward / 1 train_step 이 레벨을 얼마나 소비하는지 실측.
#   목적: bootstrap(refresh) 시기(threshold)를 정하기 위한 수치 확보.
#   실행:  python -m Soft_Tree.test.test_level

from __future__ import annotations
import numpy as np
import desilofhe
from Soft_Tree.desilo.gate_eval import fit_sigmoid_chebyshev, gate_forward
from Soft_Tree.desilo.util.keypack import setup_fhe_engine, refresh
from Soft_Tree.desilo.tree_core import EncSoftTree, forward, backward


def lv(ct):
    try:    return ct.level
    except: return "N/A"


def run(N=32, F=4, depth=3, zscale=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, F))
    W = [rng.normal(0, 0.3, F) for _ in range(depth)]
    b = np.zeros(depth); v = rng.normal(0, 0.5, 2 ** depth)
    cc, dcc, band = fit_sigmoid_chebyshev(deg=3, band=8.0)

    engine, sk, kp = setup_fhe_engine(verbose=False)
    sc = engine.slot_count
    def bcast(x): return engine.encrypt([float(x)] * sc, kp.public_key)
    def col(vs):  return engine.encrypt(list(vs) + [0.0]*(sc-len(vs)), kp.public_key)

    fresh = bcast(0.1)
    print(f"[LEVEL] engine.max_level = {getattr(engine,'max_level','N/A')}")
    print(f"[LEVEL] fresh ciphertext level (L_max) = {lv(fresh)}")

    Wct = [[bcast(W[i][f]) for f in range(F)] for i in range(depth)]
    bct = [bcast(b[i]) for i in range(depth)]
    vct = [bcast(v[l]) for l in range(2**depth)]
    Xct = [col(X[:, f]) for f in range(F)]
    tree = EncSoftTree(Wct, bct, vct, cc, dcc, band, zscale)

    print(f"\n--- forward 소비 측정 ---")
    print(f"  param W[0][0] level (진입 전) = {lv(tree.W[0][0])}")
    # 수동 affine 1레벨 확인
    acc = engine.multiply(tree.W[0][0], Xct[0], kp.relinearization_key)
    print(f"  W·X (ct*ct) 후 level = {lv(acc)}")
    z = engine.add(acc, tree.b[0])
    z = engine.multiply(z, engine.encode([zscale]*sc))
    print(f"  affine 완료 z level = {lv(z)}")
    g = gate_forward(engine, z, cc, band, kp)
    print(f"  게이트 σ(deg3) 후 level = {lv(g)}   (게이트 소비 = {lv(z)}-{lv(g)})")

    f_ct, cache = forward(engine, tree, Xct, kp, return_cache=True)
    print(f"  전체 forward 후 f level = {lv(f_ct)}   (forward 총 소비 = {lv(fresh)}-{lv(f_ct)})")

    print(f"\n--- backward 후 gradient level ---")
    yct = col((rng.uniform(0,1,N)>0.5).astype(float))
    dL = engine.multiply(engine.subtract(f_ct, yct), engine.encode([1.0]*N+[0.0]*(sc-N)))
    gW, gb, gv = backward(engine, tree, Xct, cache, dL, kp)
    print(f"  grad_W[0][0] level = {lv(gW[0][0])}")
    print(f"  grad_b[0]   level = {lv(gb[0])}")
    print(f"  grad_v[0]   level = {lv(gv[0])}")

    print(f"\n--- refresh(bootstrap) 회복 레벨 측정 ---")
    depleted = gW[0][0]                         # 저레벨 ciphertext (grad_W, level~8)
    print(f"  refresh 전 level = {lv(depleted)}")
    refreshed = refresh(engine, depleted, kp)
    print(f"  refresh 후 level = {lv(refreshed)}   ★ 이 값이 forward+backward 소비(~18)보다 커야 학습 가능")
    print(f"  (full bootstrap 회복 레벨. 이 값 - 18 = 학습 1스텝 후 남는 레벨)")

    print(f"\n[결론]")
    print(f"  · forward 소비 ~10, forward+backward 소비 ~18")
    print(f"  · refresh 후 레벨이 18+ 이면 학습 가능. refresh_below 를 그에 맞춰 설정.")
    print(f"  · 만약 refresh 후 레벨이 여전히 낮으면(예:10) → bootstrap 목표레벨 상향 필요.")


if __name__ == "__main__":
    run()
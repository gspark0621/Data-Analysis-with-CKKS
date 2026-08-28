# soft_tree/test/test_packed_forward.py  (leaf-packing 버전)
from __future__ import annotations
import numpy as np
import desilofhe
from Soft_Tree.desilo.gate_eval import fit_sigmoid_chebyshev
from Soft_Tree.desilo.util.keypack import setup_fhe_engine
from Soft_Tree.desilo.packed_core import (EncSoftTreePacked, forward_packed, block_size,
                                          make_masks, encode_X_block)


def plaintext_forward(X, W, b, v, cc, band, zscale):
    N, F = X.shape; d = len(W)
    cs = lambda z: np.polynomial.chebyshev.chebval(np.clip(z/band,-1,1), cc)
    prob = np.ones((N,1))
    for i in range(d):
        g = cs((X@W[i]+b[i])*zscale).reshape(N,1); prob = np.concatenate([prob*(1-g), prob*g],1)
    return prob @ v


def run(N=8, F=3, depth=2, zscale=0.5, seed=0, tol=1e-2):
    rng = np.random.default_rng(seed)
    X = rng.normal(0,1,(N,F)); W = [rng.normal(0,0.3,F) for _ in range(depth)]
    b = np.zeros(depth); v = rng.normal(0,0.5,2**depth)
    cc, dcc, band = fit_sigmoid_chebyshev(deg=3, band=8.0)
    f_ref = plaintext_forward(X, W, b, v, cc, band, zscale)
    print(f"[평문]   forward = {np.round(f_ref,5)}")

    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    sc = engine.slot_count; B = block_size(F, depth); L = 2**depth
    assert N*B <= sc
    X_all = encode_X_block(engine, kp, X, B)
    smask, cmask, lmasks = make_masks(engine, N, F, B, depth)
    Wct = []
    for i in range(depth):
        wf = [0.0]*sc
        for f in range(F): wf[f] = float(W[i][f])
        Wct.append(engine.encrypt(wf, kp.public_key))
    bct = [engine.encrypt([0.0]*sc, kp.public_key) for _ in range(depth)]
    vf = [0.0]*sc
    for l in range(L): vf[l] = float(v[l])          # v canonical [0..L)
    vct = engine.encrypt(vf, kp.public_key)
    tree = EncSoftTreePacked(Wct, bct, vct, cc, dcc, band, zscale, F=F, N=N, B=B, depth=depth)

    f_ct = forward_packed(engine, tree, X_all, smask, kp)
    dec = np.real(engine.decrypt(f_ct, sk))
    f_fhe = np.array([dec[n*B] for n in range(N)])
    print(f"[FHE-pk] forward = {np.round(f_fhe,5)}")
    err = np.abs(f_fhe - f_ref).max()
    print(f"\n최대 오차 = {err:.3e}  → {'PASS ✓' if err<tol else 'FAIL ✗'}")
    print(f"  (B={B}, leaf-packed: v/prob 각 1 ct)")


if __name__ == "__main__":
    run()
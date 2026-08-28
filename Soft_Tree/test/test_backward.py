# soft_tree/test/test_packed_backward.py  (leaf-packing 버전)
from __future__ import annotations
import numpy as np
import desilofhe
from Soft_Tree.desilo.gate_eval import fit_sigmoid_chebyshev
from Soft_Tree.desilo.util.keypack import setup_fhe_engine
from Soft_Tree.desilo.packed_core import (EncSoftTreePacked, forward_packed, backward_packed,
                                          block_size, make_masks, encode_X_block, encode_y_block)


def pt_fwd_bwd(X, W, b, v, cc, dcc, band, zscale, y):
    N, F = X.shape; d = len(W)
    cs  = lambda z: np.polynomial.chebyshev.chebval(np.clip(z/band,-1,1), cc)
    dcs = lambda z: np.polynomial.chebyshev.chebval(np.clip(z/band,-1,1), dcc)/band
    prob=[np.ones(N)];pbs,G,Z=[],[],[]
    for i in range(d):
        z=(X@W[i]+b[i])*zscale;g=cs(z);G.append(g);pbs.append(list(prob));Z.append(z)
        prob=[p*(1-g) for p in prob]+[p*g for p in prob]
    f=sum(prob[l]*v[l] for l in range(len(prob)));dL=f-y
    gv=[float((dL*prob[l]).sum()) for l in range(len(prob))];gp=[dL*v[l] for l in range(len(prob))]
    gW=[[0.]*F for _ in range(d)];gb=[0.]*d
    for i in reversed(range(d)):
        num=2**i;left,right=gp[:num],gp[num:];g=G[i]
        grad_g=sum((right[k]-left[k])*pbs[i][k] for k in range(num));gz=grad_g*dcs(Z[i])*zscale
        for f_ in range(F): gW[i][f_]=float((gz*X[:,f_]).sum())
        gb[i]=float(gz.sum());gp=[left[k]*(1-g)+right[k]*g for k in range(num)]
    return gW,gb,gv


def run(N=8, F=3, depth=2, zscale=0.5, seed=0, tol=1e-2):
    rng=np.random.default_rng(seed)
    X=rng.normal(0,1,(N,F));W=[rng.normal(0,0.3,F) for _ in range(depth)]
    b=np.zeros(depth);v=rng.normal(0,0.5,2**depth);y=(rng.uniform(0,1,N)>0.5).astype(float)
    cc,dcc,band=fit_sigmoid_chebyshev(deg=3,band=8.0)
    gW_ref,gb_ref,gv_ref=pt_fwd_bwd(X,W,b,v,cc,dcc,band,zscale,y)
    print(f"[평문] grad_b={np.round(gb_ref,4)} grad_v={np.round(gv_ref,4)}")

    engine,sk,kp=setup_fhe_engine(mode="gpu",verbose=False)
    sc=engine.slot_count;B=block_size(F,depth);L=2**depth
    X_all=encode_X_block(engine,kp,X,B);y_ct=encode_y_block(engine,kp,y,B)
    smask,cmask,lmasks=make_masks(engine,N,F,B,depth)
    Wct=[]
    for i in range(depth):
        wf=[0.]*sc
        for f in range(F): wf[f]=float(W[i][f])
        Wct.append(engine.encrypt(wf,kp.public_key))
    bct=[engine.encrypt([0.]*sc,kp.public_key) for _ in range(depth)]
    vf=[0.]*sc
    for l in range(L): vf[l]=float(v[l])
    vct=engine.encrypt(vf,kp.public_key)
    tree=EncSoftTreePacked(Wct,bct,vct,cc,dcc,band,zscale,F=F,N=N,B=B,depth=depth)

    f_ct,cache=forward_packed(engine,tree,X_all,smask,kp,return_cache=True)
    dL=engine.multiply(engine.subtract(f_ct,y_ct),smask)
    gW_ct,gb_ct,gv_ct=backward_packed(engine,tree,X_all,smask,cmask,lmasks,cache,dL,kp)

    a=lambda ct:np.real(engine.decrypt(ct,sk))
    decW=lambda ct:[float(a(ct)[f]) for f in range(F)]
    decV=lambda ct:[float(a(ct)[l]) for l in range(L)]
    gv_fhe=decV(gv_ct)
    eW=max(abs(decW(gW_ct[i])[f]-gW_ref[i][f]) for i in range(depth) for f in range(F))
    eb=max(abs(float(a(gb_ct[i])[0])-gb_ref[i]) for i in range(depth))
    ev=max(abs(gv_fhe[l]-gv_ref[l]) for l in range(L))
    err=max(eW,eb,ev)
    print(f"[FHE ] grad_b={np.round([float(a(gb_ct[i])[0]) for i in range(depth)],4)} grad_v={np.round(gv_fhe,4)}")
    print(f"\n최대 오차: grad_W={eW:.3e} grad_b={eb:.3e} grad_v={ev:.3e}")
    print(f"→ 전체 {err:.3e}  {'PASS ✓' if err<tol else 'FAIL ✗'}")


if __name__ == "__main__":
    run()
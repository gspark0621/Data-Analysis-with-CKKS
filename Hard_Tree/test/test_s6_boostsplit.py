# Hard_Tree/test/test_s6_boostsplit.py
"""
S6 — 부스팅 분할 기준 검증.

  ★ 분류 MGI 를 부스팅에 쓰면 안 된다 (평문 실측 −0.025~−0.127, 깊을수록 악화).
    gradient 기반 gain 이 필요한데 정확한 식은 후보마다 나눗셈이 붙는다.

        G_L²/H_L + G_R²/H_R − G_p²/H_p = (G_L·H_p − G_p·H_L)² / (H_L·H_R·H_p)

    분자만 쓰면 나눗셈이 사라진다 (MGI 가 Gini 정규화를 버린 것과 같은 구조).
    평문 실측: 정확한 gain 대비 −0.016 ~ +0.004.

  이 테스트는 FHE 구현이 그 평문 대리 기준과 **같은 트리**를 만드는지 본다.

실행:  python -m Hard_Tree.test.test_s6_boostsplit
"""
import numpy as np, sys, time
from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import train_core as TC

CASES=[(80,4,6,3),(100,6,8,3),(200,12,10,4),(256,8,8,4)]

def ref_boost_tree(Xb, g, D):
    """평문 참조 — 나눗셈 없는 대리 기준 (G_L·H_p − G_p·H_L)²."""
    n,d=Xb.shape; nb=int(Xb.max())+1
    sel=[np.ones(n,bool)]; out=[]
    for _ in range(D):
        best,bs=(0,0),-np.inf
        for j in range(d):
            for k in range(nb-1):
                b=Xb[:,j]<=k; tot=0.0
                for s_ in sel:
                    HL=float((s_&b).sum()); Hp=float(s_.sum())
                    GL=g[s_&b].sum(); Gp=g[s_].sum()
                    tot += (GL*Hp-Gp*HL)**2
                if tot>bs: bs,best=tot,(j,k)
        out.append(best)
        b=Xb[:,best[0]]<=best[1]
        sel=sum([[s_&b,s_&~b] for s_ in sel],[])
    return out

engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
print("="*78); print("부스팅 분할 기준 (나눗셈 없는 대리) — FHE vs 평문"); print("="*78)
print(f"  {'n':>5}{'d':>4}{'n_max':>7}{'D':>3}{'타일':>6} | {'트리':>8}{'시간(s)':>10}")
print("  "+"-"*56)
for (n,d,n_max,D) in CASES:
    rng=np.random.default_rng(n+d)
    Xb=rng.integers(0,n_max,(n,d)); g=rng.normal(0,1,n)
    td=TC.train_dims(n,d,n_max,2,engine.slot_count)
    t0=time.time()
    tm=TC.make_train_masks(engine,td)
    tiles=TC.encode_onehot(engine,kp,Xb,td)
    pre=TC.prefix_cumsum(engine,tiles,kp,td,tm)
    sel2=TC.encode_boost_sel(engine,kp,g,td)          # ★ [잔차, 1]
    masks=[TC.encode_sample_all(engine,kp,np.ones(n),td)]
    Lq=[TC.root_stats(engine,pre,masks[0],sel2,kp,td,tm)]
    Nq=[TC.broadcast_all(engine,Lq[0],kp,td,tm)]
    S,n_cat,per=td["S"],td["n_cat"],td["per_ct"]
    got=[]
    for lv in range(D):
        score=TC.level_score(engine,Lq,Nq,kp,td,criterion="boost")   # ★ argmax
        sd=[np.real(engine.decrypt(score[t],sk)) for t in range(td["nb_ct"])]
        best,bs=None,-np.inf
        for j in range(d):
            for k in range(n_max-1):
                blk=j*n_cat+k; t,off=blk//per,blk%per
                if sd[t][off*S]>bs: bs,best=sd[t][off*S],(j,k)
        got.append(best)
        bv=TC.extract_branch(engine,pre,best[0],best[1],kp,td)
        masks,Lq,Nq=TC.next_level(engine,pre,masks,Lq,Nq,bv,sel2,kp,td,tm,
                                  with_stats=(lv<D-1))
    ref=ref_boost_tree(Xb,g,D)
    ok = got==ref
    print(f"  {n:>5}{d:>4}{n_max:>7}{D:>3}{td['nct']:>6} | "
          f"{'일치' if ok else '불일치':>8}{time.time()-t0:>10.2f}  [{'OK' if ok else 'FAIL'}]")
    if not ok:
        print(f"        FHE  {got}"); print(f"        평문 {ref}")
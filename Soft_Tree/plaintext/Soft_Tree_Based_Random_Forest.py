"""
================================================================================
 FHE-RF: Oblivious Soft Decision Tree Forest — 최종 평문 참조 구현
================================================================================
 이 파일은 Desilo 패키지(soft_tree_desilo/)와 1:1 대응하는 평문 정답 구현입니다.
 모든 최종 결정사항 반영:
   · 모델      : oblivious oblique soft tree (레벨당 게이트 공유, w·x 사선 분기)
   · 학습      : momentum GD (argmin 없음 = 핵심 novelty), 제곱손실(σ 링크 제거)
   · backward  : block-순서, 나눗셈 없음 (유한차분 검증)
   · 게이트    : degree-3 다항 σ 근사 (depth 절약)
   · β=1 고정  : zscale=0.5 고정 (√F 규칙 폐기 → 고차원 회복)
   · 앙상블    : FHE-RF = bootstrap 가중(공개 난수) + K개 SIMD 병렬 트리
   · 예측      : 전체경로 soft 가중합 → 클라이언트 argmax = hard 라벨 (Akavia식)

 ── 파이프라인 단계 ──
   ① 전처리(클라이언트 평문)  → preprocess
   ② 암호화 전송              → (평문에선 생략)
   ③ 서버 학습(CKKS-CORE)     → ObliviousSoftTree.{forward,backward,momentum_step}
                                 + FHESoftRF.fit (bootstrap bagging)
   ④ 암호화 모델 반환          → (평문에선 생략)
   ⑤ 복호(클라이언트)          → (평문에선 생략)
   ⑥ argmax → hard 라벨        → FHESoftRF.predict_label
================================================================================
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import QuantileTransformer


# ============================================================================
# ① 전처리 (클라이언트 평문 — Desilo: keypack.py 이전, encoding 만 맞춤)
# ============================================================================
def preprocess(df, target_col, drop_cols=(), pos_label=None, max_onehot=10, cap=6000, seed=0):
    rng = np.random.default_rng(seed)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df.pop(target_col)
    y = (y == pos_label).astype(int).values if pos_label is not None else y.astype(int).values
    num, cat = [], []
    for c in df.columns:
        col = df[c]
        if col.dtype.kind in "biufc" and col.nunique() > max_onehot:
            v = pd.to_numeric(col, errors="coerce").fillna(col.median()).values.astype(float)
            v = QuantileTransformer(output_distribution="normal",
                                    n_quantiles=min(1000, len(v)), random_state=0).fit_transform(v.reshape(-1,1)).ravel()
            num.append(np.clip(v, -4, 4).reshape(-1,1))
        else:
            cat.append(pd.get_dummies(col.astype(str), prefix=c).values.astype(float))
    X = np.hstack(num + cat)
    if len(y) > cap:
        pos, neg = np.where(y==1)[0], np.where(y==0)[0]; k = cap//2
        idx = np.concatenate([rng.choice(pos, min(k,len(pos)), replace=False),
                              rng.choice(neg, min(cap-k,len(neg)), replace=False)])
        rng.shuffle(idx); X, y = X[idx], y[idx]
    return X, y


def split3(X, y, seed=0, fr=(0.6, 0.2)):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    a, b = int(fr[0]*len(y)), int(sum(fr)*len(y))
    return X[idx[:a]],y[idx[:a]], X[idx[a:b]],y[idx[a:b]], X[idx[b:]],y[idx[b:]]


# ============================================================================
# 게이트 다항 (degree-3 σ 근사)  — Desilo: gate_eval.py
# ============================================================================
class Gate:
    def __init__(self, deg=3, band=8.0):
        z = np.linspace(-band, band, 4000)
        self.coef = np.polyfit(z, 1/(1+np.exp(-z)), deg)   # σ 근사
        self.dcoef = np.polyder(self.coef)
    def f(self, z):  return np.polyval(self.coef, z)       # g = σ(z)
    def df(self, z): return np.polyval(self.dcoef, z)      # σ'(z)  (검증엔 g(1-g) 도 가능)


# ============================================================================
# ③ oblivious soft tree — 서버 CKKS-CORE (forward/backward/momentum)
#    Desilo: tree_core.py
# ============================================================================
class ObliviousSoftTree:
    def __init__(self, F, depth, gate, rng, zscale=0.5, l2=1e-5):
        self.F=F; self.depth=depth; self.gate=gate; self.zscale=zscale; self.l2=l2
        self.W=[rng.normal(0,0.3,(1,F)) for _ in range(depth)]   # 레벨당 게이트(oblivious)
        self.b=[np.zeros(1) for _ in range(depth)]
        self.v=rng.normal(0,0.5,2**depth)
        self.mW=[np.zeros_like(w) for w in self.W]               # momentum velocity
        self.mb=[np.zeros_like(bb) for bb in self.b]; self.mv=np.zeros_like(self.v)

    # ---- ### CKKS-CORE : forward (모든 경로 계산, pbs 캐시) ----
    def forward(self, X, cache=None):
        N=X.shape[0]; prob=np.ones((N,1)); Zs=[]; pbs=[]
        for i in range(self.depth):
            Wk=np.repeat(self.W[i],2**i,0); bk=np.repeat(self.b[i],2**i)
            Z=(X@Wk.T+bk)*self.zscale                     # affine [ct*ct]내적 + [pt*ct]
            g=self.gate.f(Z)                              # 게이트 [poly3]
            pbs.append(prob); Zs.append(Z)
            prob=np.concatenate([prob*(1-g),prob*g],1)    # 경로곱 [ct*ct], block 순서
        f=prob@self.v                                     # 리프 가중 [ct*ct]+[add]
        if cache is not None: cache.update(X=X,Zs=Zs,prob=prob,pbs=pbs)
        return f

    # ---- ### CKKS-CORE : backward (block-순서, 나눗셈 없음 — 유한차분 검증) ----
    def backward(self, cache, dL_df):
        X=cache["X"]; gv=cache["prob"].T@dL_df; grad_prob=np.outer(dL_df,self.v)
        gW=[np.zeros_like(w) for w in self.W]; gb=[np.zeros_like(b) for b in self.b]
        for i in reversed(range(self.depth)):
            num=2**i
            gl,gr=grad_prob[:,:num],grad_prob[:,num:]     # block 분할 = 마스킹
            Z=cache["Zs"][i]; pb=cache["pbs"][i]; g=self.gate.f(Z)
            grad_prob=gl*(1-g)+gr*g                        # 다음 레벨로
            grad_Z=(pb*(gr-gl))*self.gate.df(Z)*self.zscale   # 나눗셈 없음
            gW[i]=(grad_Z.sum(1,keepdims=True).T@X); gb[i]=np.array([grad_Z.sum()])
        return gW,gb,gv

    # ---- ### CKKS-CORE : momentum 갱신 (덧셈+평문곱, 나눗셈/sqrt 없음) ----
    def momentum_step(self, gW,gb,gv, lr, mu=0.9):
        for k in range(self.depth):
            gW[k]=gW[k]+self.l2*self.W[k]
            self.mW[k]=mu*self.mW[k]-lr*gW[k]; self.W[k]+=self.mW[k]
            self.mb[k]=mu*self.mb[k]-lr*gb[k]; self.b[k]+=self.mb[k]
        self.mv=mu*self.mv-lr*gv; self.v+=self.mv

    # ---- 학습 루프 (제곱손실, 샘플 가중 = bootstrap) ----
    def fit(self, X, y, epochs, lr, mu=0.9, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else sample_weight
        for _ in range(epochs):
            c={}; f=self.forward(X,c)
            if not np.all(np.isfinite(f)): return False
            dL=((f-y)*w)/len(y)                            # 제곱손실 + bootstrap 가중
            gW,gb,gv=self.backward(c,dL); self.momentum_step(gW,gb,gv,lr,mu)
        return True

    def score(self, X): return self.forward(X)


# ============================================================================
# FHE-RF : bootstrap 가중 + K개 SIMD 병렬 트리 (Desilo: Server_main.py)
# ============================================================================
class FHESoftRF:
    def __init__(self, depth=4, K=20, epochs=400, lr=0.3, mu=0.9,
                 zscale=0.5, gate_deg=3, bootstrap=True, seed0=0):
        self.depth=depth; self.K=K; self.epochs=epochs; self.lr=lr; self.mu=mu
        self.zscale=zscale; self.bootstrap=bootstrap; self.seed0=seed0
        self.gate=Gate(gate_deg); self.trees=[]; self.F0=0.0

    def fit(self, X, y):
        base=np.clip(y.mean(),1e-3,1-1e-3); self.F0=np.log(base/(1-base))
        N=len(y)
        for k in range(self.K):
            rng=np.random.default_rng(self.seed0+k)
            if self.bootstrap:                             # 공개 난수 복원추출 가중치
                idx=rng.integers(0,N,N); w=np.bincount(idx,minlength=N).astype(float)
            else:
                w=None
            t=ObliviousSoftTree(X.shape[1],self.depth,self.gate,rng,zscale=self.zscale)
            if not t.fit(X,y,self.epochs,self.lr,self.mu,sample_weight=w): return False
            self.trees.append(t)
        return True

    def decision(self, X):                                 # 전체경로 soft 가중합
        return self.F0 + np.mean([t.score(X) for t in self.trees],axis=0)
    def predict_proba(self, X): return 1/(1+np.exp(-self.decision(X)))
    def predict_label(self, X, thr=0.5):                   # ⑥ 클라이언트 argmax/임계 = hard
        return (self.predict_proba(X) >= thr).astype(int)


def roc_auc(y, s):
    y=np.asarray(y); o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    p=y.sum(); n=len(y)-p
    return float("nan") if p==0 or n==0 else (r[y==1].sum()-p*(p+1)/2)/(p*n)


# ============================================================================
# 데모
# ============================================================================
if __name__ == "__main__":
    UP="/mnt/user-data/uploads/"
    data={
     "Bank":(pd.read_csv(UP+"bank-full.csv",sep=";"),"y",(),"yes"),
     "Credit":(pd.read_excel(UP+"default_of_credit_card_clients.xls",header=1),"default payment next month",["ID"],None),
     "Loan":(pd.read_csv(UP+"Loan_default.csv"),"Default",["LoanID"],None),
    }
    print("FHE-RF (bootstrap, K=15, depth=4, zscale=0.5, deg3, 제곱손실, momentum)\n")
    print(f"{'dataset':8s} {'test-AUC':>9s} {'hard-라벨 acc':>13s}")
    for name,(df,tgt,drop,pos) in data.items():
        X,y=preprocess(df,tgt,drop,pos); Xtr,ytr,Xva,yva,Xte,yte=split3(X,y)
        m=FHESoftRF(depth=4,K=15,epochs=300,lr=0.3); m.fit(Xtr,ytr)
        print(f"{name:8s} {roc_auc(yte,m.decision(Xte)):9.4f} {(m.predict_label(Xte)==yte).mean():13.4f}")
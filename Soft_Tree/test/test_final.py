# soft_tree/test/test_final.py
#
# 통합 실험 러너 — packed FHE-RF 학습·검증 + 연산 계측.
#   실행:  python -m Soft_Tree.test.test_final
#   CLI :  python -m Soft_Tree.test.test_final <경로> <depth> <K> <epochs>
#
# ═══════════ 이번 개정 사항 ═══════════
#  (1) 평문 참조에서 np.clip 제거 — FHE 는 클리핑을 못 하므로 같은 함수를 계산해야 한다.
#      (기존에는 평문이 clip 을 써서 서로 다른 모델을 학습 → 상관계수 0.9839)
#  (2) 게이트를 composite minimax f_1 로 교체. 평문/FHE 가 gate_eval 의 동일 계수를 공유.
#  (3) 전처리: QuantileTransformer+clip(±4) → z-score + clip(±6), 분할 후 train 에만 fit.
#  (4) column subsampling (mtry) 도입. 마스크는 gate_eval.feature_mask 로 평문/FHE 일원화.
#  (5) 연산 계측(OPCOUNT) 리포트 추가.
#  (6) 예측 시 n_te 를 명시적으로 전달 (forward_packed 의 복제 범위 버그 수정).

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from Soft_Tree.desilo.gate_eval import (fit_gate_composite, gate_plain, gate_deriv_plain,
                                        feature_mask)
from Soft_Tree.desilo.util.keypack import setup_fhe_engine
from Soft_Tree.desilo.packed_core import (block_size, encode_X_block, encode_y_block,
                                          PTPack, train_forest_packed,
                                          predict_forest_packed,
                                          opcount_reset, opcount_report,
                                          HBDT_UNIT_MS)

# ═══════════════ [사용자 설정 — 여기만 수정] ═══════════════
DATA_PATH      = "sklearn:iris"  # sklearn:iris, sklearn:cancer, sklearn:wine, sklearn:digits, 또는 CSV 경로
TARGET_COL     = -1        # 타깃 열 인덱스 (기본 마지막 열)
HEADER         = None      # 헤더 있으면 0, 없으면 None
SEP            = ","
DROP_COLS      = []        # 제외할 열 인덱스 (예: ID 열 [0])
POSITIVE_CLASS = None      # None=자동 (2클래스는 그대로, 다클래스는 class0 vs rest)

TRAIN_FRAC     = 0.6
MAX_TRAIN      = 64        # N*B <= 32768 이어야 함
MAX_TEST       = 64

DEPTH, K_TREES, EPOCHS, LR = 4, 20, 100, 0.3

# ── 게이트 / 정규화 ──
#   beta_eff = 3 * ZSCALE / GATE_B  가 유일한 knob. 권고 ~0.56.
#   (ZSCALE=1.5, GATE_B=8) 과 (2.25, 12) 와 (9.0, 48) 은 완전히 동일하다.
ZSCALE         = 1.5
GATE_B         = 8.0
CLIP_SIGMA     = 6.0       # z-score 후 이상치 보험 (관측 범위를 거의 건드리지 않음)
MTRY           = None      # column subsampling. None=전체. 권고: max(1, F//4) 또는 int(sqrt(F))
                           #   F=30 이면 8 권장.  None 으로 두면 아래에서 자동 설정 가능.
AUTO_MTRY      = True      # True 면 MTRY=None 일 때 max(2, F//4) 로 자동 설정

SEED           = 0
COUNT_OPS      = True

# ── refresh 임계값 (v3) ──
#   bootstrap 회복 레벨 = 10, 비용 ~4.35 s (입력 레벨과 무관하게 고정).
#   따라서 "늦게" refresh 할수록 회복량이 커서 효율적이다.
#   GRAD_REFRESH_BELOW: gradient 를 높은 레벨로 되돌려 momentum 사슬(vel→param)을 보호.
#     None 이면 사용 안 함.  실험적으로 8~10 을 권장.
REFRESH_BELOW      = 8
MID_REFRESH_BELOW  = 4
GRAD_REFRESH_BELOW = None      # 첫 트리 첫 epoch 의 연산 횟수 계측
RUN_FHE        = True      # False 면 평문 참조만 실행 (빠른 확인용)
# ═══════════════════════════════════════════════════════════


def _roc_auc(y, s):
    y = np.asarray(y); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    p = y.sum(); n = len(y) - p
    return float("nan") if p == 0 or n == 0 else (r[y == 1].sum() - p*(p+1)/2)/(p*n)


def _acc(y, s, thr=0.5):
    return float(np.mean((np.asarray(s) > thr).astype(int) == np.asarray(y)))


def load_any(path, target_col=-1, header=None, sep=",", drop_cols=None, positive_class=None):
    if path.startswith("sklearn:"):
        from sklearn import datasets as D
        name = path.split(":")[1]
        d = {"cancer": D.load_breast_cancer, "iris": D.load_iris,
             "wine": D.load_wine, "digits": D.load_digits}[name]()
        X, yr = d.data.astype(float), np.asarray(d.target)
    else:
        df = pd.read_csv(path, header=header, sep=sep, skip_blank_lines=True).dropna(how="all")
        cols = list(df.columns); tcol = cols[target_col]
        yr = np.asarray(df[tcol])
        drop = set([cols[i] for i in (drop_cols or [])] + [tcol])
        X = df[[c for c in cols if c not in drop]].values.astype(float)
    classes = sorted(set(yr.tolist()))
    if positive_class is not None:
        y = (yr == positive_class).astype(int)
    elif len(classes) == 2:
        y = (yr == classes[1]).astype(int)
    else:
        y = (yr == classes[0]).astype(int)
    return X, y, classes


def preprocess(Xtr_raw, Xte_raw, clip_sigma=6.0):
    """z-score (train 에만 fit) + clip(±clip_sigma).  ★분할 이후에 호출할 것★"""
    mu = Xtr_raw.mean(0)
    sd = Xtr_raw.std(0)
    sd[sd < 1e-12] = 1.0
    f = lambda Z: np.clip((Z - mu) / sd, -clip_sigma, clip_sigma)
    return f(Xtr_raw), f(Xte_raw)


# ══════════════════════════════════════════════════════════════════
# 평문 참조 — FHE 와 수학적으로 완전히 동일해야 한다.
#   ★ np.clip 없음 (FHE 는 클리핑 불가)
#   ★ 게이트는 gate_eval 의 동일 계수
#   ★ feature 마스크는 gate_eval.feature_mask 로 동일하게 생성
#   ★ 난수 스트림 소비 순서도 init_packed_tree 와 동일 (벡터 호출)
# ══════════════════════════════════════════════════════════════════
def plaintext_softforest(Xtr, ytr, Xte, depth, K, epochs, lr, cc, dcc, band,
                         zscale=1.5, mu=0.9, mtry=None, verbose=True):
    N, F = Xtr.shape
    cs = lambda z: gate_plain(z, cc, band)
    ds = lambda z: gate_deriv_plain(z, dcc, band)      # 1/B 가 이미 흡수됨
    lr_eff = lr / N
    scores = np.zeros(len(Xte))
    zmax_all = 0.0; over_cnt = 0; over_tot = 0
    for k in range(K):
        rng = np.random.default_rng(k)
        W = [rng.normal(0, 0.3, F) for _ in range(depth)]
        M = [feature_mask(F, k, i, mtry) for i in range(depth)]
        W = [W[i] * M[i] for i in range(depth)]        # ★ 초기값부터 마스킹 (FHE 와 동일)
        b = np.zeros(depth)
        v = rng.normal(0, 0.5, 2 ** depth)
        vW = [np.zeros(F) for _ in range(depth)]; vb = np.zeros(depth); vv = np.zeros(2 ** depth)
        sw = np.ones(N)
        if K > 1:
            r2 = np.random.default_rng(10_000 + k)
            sw = np.bincount(r2.integers(0, N, N), minlength=N).astype(float)
        for _ in range(epochs):
            prob = [np.ones(N)]; pbs = []; G = []; Z = []
            for i in range(depth):
                z = (Xtr @ W[i] + b[i]) * zscale
                g = cs(z); G.append(g); pbs.append(list(prob)); Z.append(z)
                prob = [p * (1 - g) for p in prob] + [p * g for p in prob]
            P = np.array(prob); f = P.T @ v
            if not np.all(np.isfinite(f)):
                print(f"  ⚠️ [평문] tree {k}: 발산 감지 — 학습 중단"); break
            dL = (f - ytr) * sw
            gv = P @ dL
            gp = [dL * v[l] for l in range(len(prob))]
            for i in reversed(range(depth)):
                num = 2 ** i; L, R = gp[:num], gp[num:]; g = G[i]
                grad_g = sum((R[t] - L[t]) * pbs[i][t] for t in range(num))
                gz = grad_g * ds(Z[i]) * zscale
                gWi = (Xtr.T @ gz) * M[i]              # ★ gradient 도 마스킹
                vW[i] = mu * vW[i] - lr_eff * gWi; W[i] = W[i] + vW[i]
                vb[i] = mu * vb[i] - lr_eff * float(gz.sum()); b[i] += vb[i]
                gp = [L[t] * (1 - g) + R[t] * g for t in range(num)]
            vv = mu * vv - lr_eff * gv; v = v + vv
        # 진단: |z|/B 분포
        for i in range(depth):
            z = (Xtr @ W[i] + b[i]) * zscale
            r = np.abs(z) / band
            zmax_all = max(zmax_all, float(r.max())); over_cnt += int((r > 2).sum()); over_tot += r.size
        # 예측
        q = np.ones((len(Xte), 1))
        for i in range(depth):
            g = cs((Xte @ W[i] + b[i]) * zscale).reshape(-1, 1)
            q = np.concatenate([q * (1 - g), q * g], 1)
        scores += q @ v
    if verbose:
        print(f"  [평문 진단] |z|/B max = {zmax_all:.2f},  안전범위(<=2) 초과 비율 = "
              f"{over_cnt/max(over_tot,1):.3%}   (초과 1% 미만이면 정상)")
    return scores / K


def main():
    path   = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    depth  = int(sys.argv[2]) if len(sys.argv) > 2 else DEPTH
    K      = int(sys.argv[3]) if len(sys.argv) > 3 else K_TREES
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else EPOCHS

    X, y, classes = load_any(path, TARGET_COL, HEADER, SEP, DROP_COLS, POSITIVE_CLASS)
    print(f"[데이터] {path}")
    print(f"  X={X.shape}, 클래스={classes}, 양성={int(y.sum())}/{len(y)}")

    # ★ 분할 먼저 → 전처리는 train 에만 fit (leakage 제거)
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(X))
    n_tr = min(MAX_TRAIN, int(len(X) * TRAIN_FRAC))
    n_te = min(MAX_TEST, len(X) - n_tr)
    tr, te = idx[:n_tr], idx[n_tr:n_tr + n_te]
    Xtr, Xte = preprocess(X[tr], X[te], CLIP_SIGMA)
    ytr, yte = y[tr], y[te]

    F = X.shape[1]; B = block_size(F, depth)
    mtry = MTRY
    if mtry is None and AUTO_MTRY:
        mtry = max(2, F // 4)
    print(f"[설정] Ntr={n_tr}, Nte={n_te}, F={F}, B={B}, N*B={n_tr*B}/32768, "
          f"depth={depth}, K={K}, epochs={epochs}")
    print(f"       zscale={ZSCALE}, gate_B={GATE_B}, beta={3*ZSCALE/GATE_B:.3f}, "
          f"mtry={mtry}, clip=±{CLIP_SIGMA}")
    if n_tr * B > 32768:
        print("  ⚠️ N*B > 32768 — MAX_TRAIN 을 줄이세요."); return
    if n_te * B > 32768:
        print("  ⚠️ N_te*B > 32768 — MAX_TEST 를 줄이세요."); return

    # ── 평문 참조 ──
    cc, dcc, band = fit_gate_composite(B=GATE_B, n=1)
    print(f"\n[게이트] composite minimax f_1,  cheb={np.round(cc,6).tolist()},  B={band}")
    s_pt = plaintext_softforest(Xtr, ytr, Xte, depth, K, epochs, LR, cc, dcc, band,
                                zscale=ZSCALE, mtry=mtry)
    print(f"[평문 참조]  AUC = {_roc_auc(yte, s_pt):.4f}   ACC = {_acc(yte, s_pt):.4f}")

    if not RUN_FHE:
        print("\n(RUN_FHE=False — FHE 실행 생략)"); return

    # ── FHE ──
    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=False)
    Xtr_all = encode_X_block(engine, kp, Xtr, B)
    ytr_ct  = encode_y_block(engine, kp, ytr, B)
    pt_tr = PTPack(engine, n_tr, F, B, depth, ZSCALE, GATE_B)

    opcount_reset()
    trees = train_forest_packed(engine, kp, sk, Xtr_all, ytr_ct, pt_tr,
                                F=F, N=n_tr, B=B, depth=depth, K=K, epochs=epochs, lr=LR,
                                zscale=ZSCALE, gate_B=GATE_B, mtry=mtry,
                                refresh_below=REFRESH_BELOW,
                                mid_refresh_below=MID_REFRESH_BELOW,
                                grad_refresh_below=GRAD_REFRESH_BELOW,
                                count_ops=COUNT_OPS)
    opcount_report(f"학습 전체 (K={K}, depth={depth}, epochs={epochs})")
    opcount_report(f"학습 전체 — HBDT 단가 환산", HBDT_UNIT_MS, "HBDT Table 4")

    Xte_all = encode_X_block(engine, kp, Xte, B)
    pt_te = PTPack(engine, n_te, F, B, depth, ZSCALE, GATE_B)
    opcount_reset()
    score_ct = predict_forest_packed(engine, trees, Xte_all, pt_te, kp, n_te=n_te)
    opcount_report("예측")

    dec = np.real(engine.decrypt(score_ct, sk))
    s_fhe = np.array([dec[n * B] for n in range(n_te)])
    corr = np.corrcoef(s_fhe, s_pt)[0, 1]
    print(f"\n[FHE packed] AUC = {_roc_auc(yte, s_fhe):.4f}   ACC = {_acc(yte, s_fhe):.4f}")
    print(f"[대조] 예측 상관계수(FHE vs 평문) = {corr:.6f}")
    print(f"[대조] 최대 절대오차 = {np.max(np.abs(s_fhe - s_pt)):.3e},  "
          f"평균 |평문| = {np.mean(np.abs(s_pt)):.3e}")
    if corr < 0.999:
        print("  ⚠️ 상관계수 < 0.999 — 게이트가 동일 함수인데도 어긋난다면 packing/마스킹 문제.")
        print("     체크: (1) feature_mask 시드 일치  (2) init 난수 스트림 일치  "
              "(3) n_te 전달  (4) vcmask")
    else:
        print("  ✅ 평문-FHE 일치 (게이트가 근사가 아니라 동일 다항식이므로 1.0 에 근접해야 정상)")


if __name__ == "__main__":
    main()
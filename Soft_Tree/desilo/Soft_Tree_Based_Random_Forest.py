# soft_tree/Server_main.py
#
# FHE oblivious soft tree — 서버 메인 파이프라인 (학습).
#   ① 클라이언트: 전처리(quantile) + 암호화 + keypack (Client 측, 여기선 인터페이스)
#   ② 서버: K개 트리 다중초기화 병렬 momentum 학습 (본 파일)
#   ③ 반환: 암호화된 모델 → 클라이언트 복호 → argmax hard 라벨
#
# ── 인터페이스 (DBSCAN send_to_server_fhe 와 유사) ──────────────────────────
#   train_soft_forest_fhe(engine, keypack, secret_key, X_cols, y_ct, ...)
#
# ★ 실행 전제: keypack.py / gate_eval.py / tree_core.py 완비 + backward affine 구현.
#   현재 forward(추론)는 완결. train_step 은 backward affine(packing) 채우면 완결.

from __future__ import annotations
import gc
from time import time
from typing import List
import numpy as np
try:
    from desilofhe import Engine, Ciphertext
except Exception:
    Engine = Ciphertext = object   # 폴백
from Soft_Tree.desilo.util.keypack import KeyPack, refresh
from Soft_Tree.desilo.gate_eval import fit_sigmoid_chebyshev
from Soft_Tree.desilo.tree_core import EncSoftTree, forward, train_step


def _init_enc_tree(engine, keypack, F, depth, cheb_coeffs, dcheb_coeffs, band, zscale, seed):
    """공개 난수로 초기 파라미터 생성 후 암호화(다중초기화의 한 seed)."""
    rng = np.random.default_rng(seed)
    sc  = engine.slot_count
    def bcast(val):   # 스칼라 → 전 슬롯 broadcast 암호문
        return engine.encrypt([float(val)] * sc, keypack.public_key)
    W = [[bcast(rng.normal(0, 0.3)) for _ in range(F)] for _ in range(depth)]
    b = [bcast(0.0) for _ in range(depth)]
    v = [bcast(rng.normal(0, 0.5)) for _ in range(2 ** depth)]
    return EncSoftTree(W, b, v, cheb_coeffs, dcheb_coeffs, band, zscale)


def train_soft_forest_fhe(
    engine: Engine, keypack: KeyPack, secret_key,
    X_cols: List[Ciphertext], y_ct: Ciphertext,
    F: int, N: int, depth: int = 4, K: int = 20,  # ★ N=샘플 수, K↑(bootstrap bagging)
    epochs: int = 400, lr: float = 0.3, mu: float = 0.9,
    zscale: float = 0.5, gate_deg: int = 3, gate_band: float = 8.0,
    bootstrap: bool = True,                        # ★ True=FHE-RF(진짜 bagging), False=multi-init
    refresh_below: int = 8,                       # step 시작 파라미터 refresh 임계
    mid_refresh_below: int = 6,                     # ★ 중간 refresh 임계 (small 회복레벨보다 낮게!)
):
    """
    서버 학습 메인 (FHE-RF).

    Parameters
    ----------
    X_cols : list[Ciphertext]   feature 별 암호문(F개, 각 N샘플 슬롯).
    y_ct   : Ciphertext          레이블(N샘플 슬롯).
    K      : int                 트리 수(SIMD 병렬, depth-free).
    bootstrap : bool             각 트리를 복원추출 표본으로 학습(비상관 → 진짜 RF).
                                 샘플 가중치는 공개 난수(평문) → pt*ct 곱 1회.

    Returns
    -------
    trees : list[EncSoftTree]    학습된 암호화 모델(클라이언트가 복호).
    """
    cheb_coeffs, dcheb_coeffs, band = fit_sigmoid_chebyshev(deg=gate_deg, band=gate_band)
    print(f"[Server] FHE-RF 학습: K={K}, depth={depth}, epochs={epochs}, "
          f"zscale={zscale}, bootstrap={bootstrap}, gate(deg={gate_deg})")
    sc = engine.slot_count

    def _bootstrap_weight_pt(k):
        """
        트리 k 의 샘플 가중치(공개 평문). 항상 슬롯 [N..끝]=0 → 쓰레기 마스킹 겸용.
          bootstrap=True : 복원추출 등장 횟수 (진짜 bagging)
          bootstrap=False: 전부 1 (마스크만; 가중 없음)
        ★ dL_df 에 곱해져서 슬롯 [N..끝](broadcast bias 쓰레기)을 0 으로 만든다.
        """
        if bootstrap:
            rng = np.random.default_rng(10_000 + k)
            idx = rng.integers(0, N, N)
            counts = np.bincount(idx, minlength=N).astype(float)
            return engine.encode(list(counts) + [0.0] * (sc - N))
        else:
            return engine.encode([1.0] * N + [0.0] * (sc - N))  # 마스크(가중 없음)

    trees = []
    for k in range(K):
        t0 = time()
        tree = _init_enc_tree(engine, keypack, F, depth, cheb_coeffs, dcheb_coeffs, band, zscale, seed=k)
        sw_pt = _bootstrap_weight_pt(k)                    # 공개 난수 bootstrap 가중치
        vel_W = vel_b = vel_v = None
        lr_eff = lr / float(N)                             # ★ 1/N 을 lr 로 흡수
        for ep in range(epochs):
            tree, vel_W, vel_b, vel_v = train_step(
                engine, tree, X_cols, y_ct, keypack, lr=lr_eff, mu=mu,
                vel_W=vel_W, vel_b=vel_b, vel_v=vel_v, sample_weight_pt=sw_pt,
                refresh_below=refresh_below, mid_refresh_below=mid_refresh_below,
                verbose_level=(k == 0 and ep < 2))   # 첫 트리 첫 2스텝만 레벨 로깅
            if (ep + 1) % 50 == 0:
                print(f"  [tree {k}] epoch {ep+1}/{epochs}  ({time()-t0:.1f}s)")
        trees.append(tree)
        del vel_W, vel_b, vel_v; gc.collect()   # ★ 트리 간 메모리 회수
        print(f"[Server] tree {k+1}/{K} 학습 완료 ({time()-t0:.1f}s)")
    return trees


def _refresh_tree_params(engine, tree, keypack):
    """학습 후 저레벨 상태인 모델 파라미터를 refresh → 예측 forward 가 완주하도록.
       예측은 backward 가 없어 forward(소비~10)만 하므로 1회 refresh 로 충분."""
    for i in range(tree.depth):
        for f_ in range(tree.F):
            tree.W[i][f_] = refresh(engine, tree.W[i][f_], keypack)
        tree.b[i] = refresh(engine, tree.b[i], keypack)
    for l in range(len(tree.v)):
        tree.v[l] = refresh(engine, tree.v[l], keypack)
    return tree


def predict_score_fhe(engine: Engine, trees: List[EncSoftTree],
                      X_cols: List[Ciphertext], keypack: KeyPack,
                      mid_refresh_below: int = 6) -> Ciphertext:
    """K개 트리 soft 가중합 → 로짓 스코어 암호문. (⑥ 클라이언트가 복호 후 argmax)"""
    # ★ 예측 forward: 파라미터를 affine 에서 그때그때 refresh(refresh_params_below)
    #   → 140개를 한꺼번에 refresh 해 동시 생존시키던 방식의 OOM 회피.
    #   + mid-refresh(refresh_below) 로 게이트 출력 refresh → prob 유지.
    f = forward(engine, trees[0], X_cols, keypack,
                refresh_below=mid_refresh_below, refresh_params_below=8)
    for t in trees[1:]:
        f = engine.add(f, forward(engine, t, X_cols, keypack,
                                  refresh_below=mid_refresh_below, refresh_params_below=8))
    invK = engine.encode([1.0 / len(trees)] * engine.slot_count)
    return engine.multiply(f, invK)                     # 평균
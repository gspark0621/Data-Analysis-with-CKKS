# Hard_Tree/desilo/train_core.py
"""
symmetric tree 학습 루프 — CKKS 코어.

 [레이아웃]  feature-block major
     one-hot 데이터   slot[ (j·n_cat + b)·n_pad + l ] = 1  iff  bin(x_lj) == b
       j ∈ [0,d)   feature
       b ∈ [0,n_cat)  bin        n_cat  = 2^⌈log2 n_max⌉
       l ∈ [0,n_pad)  sample     n_pad  = 2^⌈log2 n⌉
     블록 하나 = n_pad 슬롯.  암호문 하나에 M/n_pad 개 블록.
     전체 d·n_cat 블록 → nct = ⌈d·n_cat·n_pad / M⌉ 개 암호문(타일).

 [핵심 이득]
   · prefix (X_j ≤ k) 는 b 축 누적합 → **회전만**, 비교 없음
   · 파티션·클래스는 샘플 축 벡터를 블록 전체로 복제해 곱한다 (pt/ct 곱 1회)
   · 결과 L[j][k] 는 슬롯 (j·n_cat+k)·n_pad 에 모인다

 [절감]
   T0  subsample        공개 평문 마스크.  대규모에서 최대 10×
   T2  left-only MGI    오른쪽 자식 통계 불필요.  2×
   T5  방식 C           오른쪽 파티션은 뺄셈.  2×
   symmetric            T3(argmin)·T4(모델갱신) 이 노드당 → 레벨당

 [회전 규약]  engine.rotate(ct,key,s) = 오른쪽 = np.roll(+s).  packed_core 와 동일.
"""
from __future__ import annotations
import gc
import numpy as np

try:
    from desilofhe import Engine, Ciphertext
    _CT_CLS = Ciphertext
except Exception:
    Engine = Ciphertext = object
    _CT_CLS = None

from Hard_Tree.desilo.hard_core import (p2, _rot, _lvl,
                                        build_stride_masks, apply_stride)

_MEM_FREE = True


# ══════════════════════════════════════════════════════════════════
# 크기
# ══════════════════════════════════════════════════════════════════
def train_dims(n, d, n_max, c, slot_count=32768):
    """
    두 단계 타일링으로 **모든 n 을 수용**한다.

      ① 샘플 축 청크    S = min(2^⌈log2 n⌉, M),  n_chunk = n_pad / S
         n 이 슬롯 수를 넘으면 샘플을 나눈다.  카운트는 청크 합산이라 덧셈뿐.
      ② 블록 타일       per_ct = M / S  블록을 한 암호문에

      per_ct ≥ n_cat  →  n_cat 배수로 내림 (feature 가 타일 안에 통째로)
      per_ct <  n_cat →  2의 거듭제곱으로 내림 (n_cat 을 나눔 → feature 가
                          연속한 여러 타일에 걸침.  b 축 누적합은 carry 로 처리)

    ★ 이전 버전은 per_ct 를 n_cat 배수로 강제해 per_ct·n_pad > M 이 되는
      경우(n 이 크면 발생)에 조용히 깨졌다.
    """
    n_pad = p2(n)
    n_cat = p2(n_max)
    S = min(n_pad, slot_count)                    # 청크당 샘플 슬롯
    n_chunk = max(1, n_pad // S)
    raw = max(1, slot_count // S)                 # 암호문당 블록 수
    if raw >= n_cat:
        per_ct = (raw // n_cat) * n_cat           # n_cat 배수
        split_feat = False
    else:
        per_ct = 1 << (raw.bit_length() - 1)      # 2의 거듭제곱 (n_cat 을 나눔)
        split_feat = True
    blocks = d * n_cat
    nb_ct = max(1, -(-blocks // per_ct))
    assert per_ct * S <= slot_count, "레이아웃 오류"
    return dict(n=n, d=d, n_max=n_max, c=c, n_pad=n_pad, n_cat=n_cat, S=S,
                n_chunk=n_chunk, blocks=blocks, per_ct=per_ct,
                nb_ct=nb_ct, nct=nb_ct * n_chunk, split_feat=split_feat,
                M=slot_count)


def _tile(td, bt, ch):
    """(블록타일 bt, 샘플청크 ch) → 타일 리스트 인덱스."""
    return ch * td["nb_ct"] + bt


def _blk_range(td, bt):
    """블록타일 bt 가 담는 블록 인덱스 [lo, hi)."""
    lo = bt * td["per_ct"]
    return lo, min(td["blocks"], lo + td["per_ct"])


# ══════════════════════════════════════════════════════════════════
# 인코딩 (클라이언트)
# ══════════════════════════════════════════════════════════════════
def encode_onehot(engine, kp, Xb, td):
    """one-hot 데이터 → 타일 리스트.  인덱스는 _tile(td, bt, ch)."""
    M, S, n_cat, d = td["M"], td["S"], td["n_cat"], td["d"]
    n = len(Xb)
    tiles = [None] * td["nct"]
    for ch in range(td["n_chunk"]):
        s0 = ch * S
        for bt in range(td["nb_ct"]):
            lo, hi = _blk_range(td, bt)
            v = [0.0] * M
            for blk in range(lo, hi):
                j, b = divmod(blk, n_cat)
                if j >= d or b >= td["n_max"]:
                    continue
                off = (blk - lo) * S
                for l in range(s0, min(n, s0 + S)):
                    if Xb[l, j] == b:
                        v[off + (l - s0)] = 1.0
            tiles[_tile(td, bt, ch)] = engine.encrypt(v, kp.public_key)
    return tiles


def encode_sample_vec(engine, kp, vec, td, chunk=0, encrypted=True):
    """샘플 축 벡터 → 한 청크 분량, 블록 전체에 복제."""
    M, S = td["M"], td["S"]
    s0 = chunk * S
    v = [0.0] * M
    for blk in range(td["per_ct"]):
        for l in range(s0, min(len(vec), s0 + S)):
            v[blk * S + (l - s0)] = float(vec[l])
    return engine.encrypt(v, kp.public_key) if encrypted else engine.encode(v)


def encode_sample_all(engine, kp, vec, td, encrypted=True):
    """청크마다 하나씩 → 리스트."""
    return [encode_sample_vec(engine, kp, vec, td, ch, encrypted)
            for ch in range(td["n_chunk"])]


def sample_mask_pt(engine, vec, td):
    """공개 평문 샘플 마스크 (subsample 등).  청크별 리스트."""
    return encode_sample_all(engine, None, vec, td, encrypted=False)


# ══════════════════════════════════════════════════════════════════
# T1 코어 연산
# ══════════════════════════════════════════════════════════════════
def make_train_masks(engine, td):
    """반복 사용하는 평문 마스크를 한 번만 만든다 (engine.encode 는 비싸다)."""
    M, S, n_cat, n_max, per_ct = (td["M"], td["S"], td["n_cat"],
                                  td["n_max"], td["per_ct"])
    m = {}
    if not td["split_feat"]:
        # per_ct 가 n_cat 배수 → 타일 안에서 b 축 누적합.  타일마다 패턴 동일
        cum = []
        step = 1
        while step < n_cat:
            mv = [0.0] * M
            for b in range(per_ct):
                if (b % n_cat) >= step:
                    mv[b * S:(b + 1) * S] = [1.0] * S
            cum.append((engine.encode(mv), step * S))
            step *= 2
        m["cum"] = cum
        fm = [0.0] * M
        lastm = [0.0] * M
        for b in range(per_ct):
            if (b % n_cat) < n_max:
                fm[b * S] = 1.0
            if (b % n_cat) == n_max - 1:
                lastm[b * S] = 1.0
        m["first"], m["last"] = engine.encode(fm), engine.encode(lastm)
    else:
        # per_ct 가 n_cat 을 나눔 → feature 가 여러 타일에 걸침.  타일 내부는
        # per_ct 블록만 누적하고, 타일 간은 carry(덧셈)로 잇는다.
        cum = []
        step = 1
        while step < per_ct:
            mv = [0.0] * M
            for b in range(per_ct):
                if b >= step:
                    mv[b * S:(b + 1) * S] = [1.0] * S
            cum.append((engine.encode(mv), step * S))
            step *= 2
        m["cum"] = cum
        fm = [0.0] * M
        for b in range(per_ct):
            fm[b * S] = 1.0
        m["first"] = engine.encode(fm)
        lastblk = [0.0] * M                      # 타일의 마지막 블록만
        lastblk[(per_ct - 1) * S:per_ct * S] = [1.0] * S
        m["lastblk"] = engine.encode(lastblk)
    return m


def _bcast_block(engine, ct, kp, td):
    """블록 0 → 타일의 per_ct 블록 전체로 복제."""
    S, per_ct = td["S"], td["per_ct"]
    out, step = ct, S
    while step < per_ct * S:
        out = engine.add(out, _rot(engine, out, kp, step)); step *= 2
    return out


def prefix_cumsum(engine, tiles, kp, td, tm):
    """
    b 축 누적합 →  [ bin(x_lj) ≤ k ].  ★ 비교 없이 회전/덧셈만.

      split_feat=False : 타일 안에서 log2(n_cat) 회전
      split_feat=True  : 타일 안 log2(per_ct) 회전 + 타일 간 carry(덧셈)
                         per_ct=1 이면 **회전 0회, 덧셈만**
    """
    S, n_cat, per_ct = td["S"], td["n_cat"], td["per_ct"]
    out = [None] * td["nct"]
    tpf = max(1, n_cat // per_ct)                # feature 당 블록타일 수
    for ch in range(td["n_chunk"]):
        carry = None
        for bt in range(td["nb_ct"]):
            ct = tiles[_tile(td, bt, ch)]
            acc = ct
            for mv, sh in tm["cum"]:             # 타일 내부 누적
                acc = engine.add(acc, engine.multiply(_rot(engine, acc, kp, sh), mv))
            if td["split_feat"]:
                if bt % tpf == 0:
                    carry = None                 # 새 feature 시작
                if carry is not None:
                    acc = engine.add(acc, carry)
                last = engine.multiply(acc, tm["lastblk"])
                last = _rot(engine, last, kp, -(per_ct - 1) * S)
                carry = _bcast_block(engine, last, kp, td)
            out[_tile(td, bt, ch)] = acc
    return out


def counts_for(engine, pre, sel_list, kp, td, tm=None):
    """
    L[j][k] = Σ_l sel(l)·pre_{j,k}(l).   샘플 청크를 합산한다.
      sel_list : 청크별 선택 벡터 (파티션 마스크 ⊙ 클래스 ⊙ subsample)
    반환: 블록타일별 리스트.  값은 슬롯 (blk−lo)·S 에.
    """
    S = td["S"]
    rlk = kp.relinearization_key
    is_list = isinstance(sel_list, (list, tuple))
    sel0 = sel_list[0] if is_list else sel_list
    ct_sel = _is_ct(sel0)                       # 루프 밖에서 1회만 판별
    out = []
    for bt in range(td["nb_ct"]):
        acc = None
        for ch in range(td["n_chunk"]):
            sel = sel_list[ch] if is_list else sel_list
            ct = pre[_tile(td, bt, ch)]
            prod = (engine.multiply(ct, sel, rlk) if ct_sel
                    else engine.multiply(ct, sel))
            step = 1
            while step < S:                      # 블록 내 샘플 축 합
                prod = engine.add(prod, _rot(engine, prod, kp, -step)); step *= 2
            acc = prod if acc is None else engine.add(acc, prod)   # 청크 합산
        if tm is not None and "first" in tm:
            acc = engine.multiply(acc, tm["first"])
        out.append(acc)
    return out


def _is_ct(x):
    """Ciphertext 판별.  ★ Plaintext 에도 .level 이 있어 hasattr 로는 안 된다.
       ★ import 는 모듈 레벨에서 한 번만 — 핫루프 안에 두면 눈에 띄게 느려진다."""
    if _CT_CLS is not None:
        return isinstance(x, _CT_CLS)
    return type(x).__name__.endswith("Ciphertext") or hasattr(x, "level")


def mul_sample(engine, a, b, kp):
    """샘플 축 벡터끼리 곱.  리스트(청크별)면 원소별로."""
    if isinstance(a, (list, tuple)):
        return [mul_sample(engine, x, (b[i] if isinstance(b, (list, tuple)) else b),
                           kp) for i, x in enumerate(a)]
    return (engine.multiply(a, b, kp.relinearization_key) if _is_ct(b)
            else engine.multiply(a, b))


def sub_sample(engine, a, b):
    """샘플 축 벡터 뺄셈 (방식 C).  리스트 대응."""
    if isinstance(a, (list, tuple)):
        return [engine.subtract(x, y) for x, y in zip(a, b)]
    return engine.subtract(a, b)


def broadcast_total(engine, Lct_list, kp, td, tm):
    """
    각 feature 의 k=n_max−1 값(파티션 전체 카운트 N)을 그 feature 의 모든 k 로 복제.

      split_feat=False : 정렬 1회 + log2(n_cat) 복제  (타일 내부)
      split_feat=True  : feature 의 마지막 타일을 그 feature 의 모든 타일에 배정
                         → **회전 0회, 대입만**
    """
    S, n_cat, per_ct = td["S"], td["n_cat"], td["per_ct"]
    if not td["split_feat"]:
        out = []
        for L in Lct_list:
            v = engine.multiply(L, tm["last"])
            v = _rot(engine, v, kp, -(td["n_max"] - 1) * S)
            step = S
            while step < n_cat * S:
                v = engine.add(v, _rot(engine, v, kp, step)); step *= 2
            out.append(v)
        return out
    # feature 가 여러 타일에 걸침 — 마지막 유효 k 가 있는 타일을 찾아 복제
    tpf = n_cat // per_ct
    last_bt_off = (td["n_max"] - 1) // per_ct            # feature 내 타일 오프셋
    last_pos = (td["n_max"] - 1) % per_ct
    mv = [0.0] * td["M"]
    mv[last_pos * S] = 1.0
    lastm = engine.encode(mv)
    out = [None] * td["nb_ct"]
    for f0 in range(0, td["nb_ct"], tpf):
        src = engine.multiply(Lct_list[f0 + last_bt_off], lastm)
        if last_pos:
            src = _rot(engine, src, kp, -last_pos * S)
        src = _bcast_block(engine, src, kp, td)
        for o in range(tpf):
            if f0 + o < td["nb_ct"]:
                out[f0 + o] = src
    return out


# ══════════════════════════════════════════════════════════════════
# T2  left-only MGI
# ══════════════════════════════════════════════════════════════════
def mgi_left_only(engine, Lq, Nq, kp, td):
    """
    한 파티션의 기여:
        2[ L² − N·L + Σ_q N_q·L_q − Σ_q L_q² ]      (+ const, argmin 에 무관)
      ★ 인수분해하면
        2[ L·(L−N) + Σ_q L_q·(N_q − L_q) ]
      ct·ct 곱이 **2c+2 → c+1** 로 반이 된다.
      (ct·ct 곱 9.664ms vs 뺄셈 0.399ms — 뺄셈으로 바꾸는 게 압도적으로 이득)

      Lq : 후보별 왼쪽 자식 클래스 카운트   (타일 리스트) × c
      Nq : 파티션 전체 클래스 카운트        (타일 리스트) × c

    ★ 오른쪽 자식 통계를 계산하지 않는다.  실측 오차 0 의 항등식.
    """
    rlk = kp.relinearization_key
    c = td["c"]
    out = []
    for t in range(td["nb_ct"]):
        L = Lq[0][t]
        N = Nq[0][t]
        for q in range(1, c):
            L = engine.add(L, Lq[q][t])
            N = engine.add(N, Nq[q][t])
        term = engine.multiply(L, engine.subtract(L, N), rlk)     # L·(L−N)
        for q in range(c):
            term = engine.add(term, engine.multiply(                # L_q·(N_q−L_q)
                Lq[q][t], engine.subtract(Nq[q][t], Lq[q][t]), rlk))
        out.append(term)          # 상수 2 는 argmin 에 무관하므로 생략
    return out


def boost_gain(engine, GL, HL, Gp, Hp, kp, td):
    """
    부스팅 분할 이득 (나눗셈 없는 대리 기준).  ★ 클수록 좋다 (argmax).

      정확한 gain:
          G_L²/H_L + G_R²/H_R − G_p²/H_p
        = (G_L·H_p − G_p·H_L)² / (H_L·H_R·H_p)      ← 대수적 항등식

      분모는 후보마다 달라 나눗셈이 필요하지만, **분자만** 쓰면
      나눗셈 없이 같은 역할을 한다.  MGI 가 Gini 의 정규화를 버린 것과 같은 구조.
      평문 실측: 정확한 gain 대비 −0.016 ~ +0.004 (D=3~6).

      GL, HL : 왼쪽 자식의 gradient 합 / 크기   (후보별, 타일 리스트)
      Gp, Hp : 부모의 gradient 합 / 크기        (broadcast_total 결과)

    ct·ct 곱 3회/타일 (G_L·H_p, G_p·H_L, 제곱).  분류 MGI(c+1)와 동급.
    """
    rlk = kp.relinearization_key
    out = []
    for t in range(td["nb_ct"]):
        diff = engine.subtract(engine.multiply(GL[t], Hp[t], rlk),
                               engine.multiply(Gp[t], HL[t], rlk))
        out.append(engine.multiply(diff, diff, rlk))
    return out


def add_tiles(engine, a, b):
    return [engine.add(x, y) for x, y in zip(a, b)] if a is not None else b


# ══════════════════════════════════════════════════════════════════
# T1 + T5  레벨 진행 (방식 C)
# ══════════════════════════════════════════════════════════════════
def root_stats(engine, pre, base_mask, ycls, kp, td, tm):
    """루트(파티션 1개)의 클래스 카운트.  반환 [ [Lq_tiles] × c ]."""
    return [counts_for(engine, pre, mul_sample(engine, base_mask, ycls[q], kp),
                       kp, td, tm) for q in range(len(ycls))]


def next_level(engine, pre, masks, Lq_list, Nq_list, b_left, ycls, kp, td, tm,
               with_stats=True):
    """
    레벨 t → t+1.  ★ 방식 C — 왼쪽 자식만 계산하고 오른쪽은 뺄셈.

        L[2p][j][k][q] + L[2p+1][j][k][q] = L[p][j][k][q]     (대수적 항등식)
        N[2p+1] = N[p] − N[2p]        broadcast_total 이 선형이라 그대로 성립
        m_right = m_p − m_left        (곱 없음)

    ★ with_stats=False : 마스크만 갱신.  **마지막 레벨에서 필수** —
      레벨 D 의 통계는 점수 계산에 쓰이지 않으므로 계산하면 순손해다.

    반환: (new_masks, new_Lq_list, new_Nq_list)
    """
    c, nb = len(ycls), td["nb_ct"]
    nm, nL, nN = [], [], []
    for i, mp in enumerate(masks):
        ml = mul_sample(engine, mp, b_left, kp)          # 왼쪽 마스크
        mr = sub_sample(engine, mp, ml)                  # ★ 뺄셈
        nm += [ml, mr]
        if not with_stats:
            continue
        Lp, Np = Lq_list[i], Nq_list[i]
        Ll = [counts_for(engine, pre, mul_sample(engine, ml, ycls[q], kp),
                         kp, td, tm) for q in range(c)]  # 왼쪽만 계산
        Nl = [broadcast_total(engine, Ll[q], kp, td, tm) for q in range(c)]
        Lr = [[engine.subtract(Lp[q][t], Ll[q][t]) for t in range(nb)]
              for q in range(c)]                         # ★ 뺄셈
        Nr = [[engine.subtract(Np[q][t], Nl[q][t]) for t in range(nb)]
              for q in range(c)]                         # ★ 뺄셈
        nL += [Ll, Lr]; nN += [Nl, Nr]
        if _MEM_FREE:
            Lq_list[i] = None; Nq_list[i] = None         # 부모 통계 해제
    if _MEM_FREE:
        gc.collect()
    return nm, nL, nN


def broadcast_all(engine, Lq, kp, td, tm):
    """Lq → Nq (파티션 전체 값).  루트에서 한 번만."""
    return [broadcast_total(engine, Lq[q], kp, td, tm) for q in range(len(Lq))]


def partition_cube_sum(engine, Nq_list, kp, td):
    """
    Σ_p (H′_p)³ .  H′_p = Nq[p][1] (파티션 크기, n 으로 정규화됨).
    파티션당 ct·ct 곱 2회.
    """
    rlk = kp.relinearization_key
    out = None
    for Nq in Nq_list:
        H = Nq[1]                                   # 타일 리스트
        cube = [engine.multiply(engine.multiply(h, h, rlk), h, rlk) for h in H]
        out = cube if out is None else [engine.add(a, b) for a, b in zip(out, cube)]
    return out


NORM_ITERS_MAX = 6      # (6−1)+⌈log2 6⌉+2 = 10 = L_SMALL


def norm_iters(level):
    """
    레벨별 Goldschmidt 반복 수.  ★ level 은 공개 파라미터라 분기해도 된다.

      t=0 : 파티션이 루트 하나뿐 → H′ = n/n = 1 → Σ_p H′_p³ = 1
            분모가 **상수 4** 다.  나눗셈이 아예 필요 없다 (깊이 0).
      t≥1 : Σ_p H′_p³ ∈ [1/4^t, 1]  →  동적 범위 4^t  →  2t+5 회면 수렴.

    (이전 코드는 max(8, …) 로 t=0 에서도 8회를 돌아 10 레벨을 낭비했다)
    """
    if level == 0:
        return 0
    it = 2 * level + 5
    # ★ refresh 회복(L_SMALL=10) 안에 들어가야 한다.
    #     깊이 = (it−1) + ⌈log2 it⌉ + 2 ≤ 10   →   it ≤ 6
    #   깊은 레벨에서는 반복이 모자라 1/x 를 **과소추정**한다.
    #   Goldschmidt 는 아래에서 수렴하므로 점수가 작아질 뿐 상한 1 은 지켜진다
    #   (안전).  대신 dead zone 이 늘어 α 가 더 필요하다 — pick_alpha 가 흡수.
    return min(it, NORM_ITERS_MAX)


def normalize_score(engine, score, Nq_list, kp, td, level, iters=None):
    """
    score ← score / (4·Σ_p H′_p³)    →   [0, 1] 보장.

    ★ 휴리스틱 상수 없이 계산으로 상한을 잡는다.
    ★ level 별 분기 — 루트는 분모가 공개 상수라 나눗셈을 건너뛴다.

    깊이:  t=0 → 1 (pt 곱만),   t≥1 → (2t+5) + log2(2t+5) + 2
    """
    import math
    from Hard_Tree.desilo.util.keypack import refresh as _rf
    if iters is None:
        iters = norm_iters(level)
    if iters == 0:                                  # ★ 루트: 분모가 공개 상수 4
        q = engine.encode([0.25] * td["M"])
        return [engine.multiply(x, q) for x in score]

    # ★ Goldschmidt 깊이 = (iters−1) + ⌈log2 iters⌉,  거기에 den(1) + score곱(1).
    #   깊은 레벨에서 이게 통째로 들어가 점수 레벨이 0 이 되던 버그.
    need = (iters - 1) + int(math.ceil(math.log2(max(2, iters)))) + 2
    cube = partition_cube_sum(engine, Nq_list, kp, td)
    if 0 <= _lvl(cube[0]) < need:
        cube = [_rf(engine, c, kp) for c in cube]
    four = engine.encode([4.0] * td["M"])
    out = []
    for t in range(td["nb_ct"]):
        den = engine.multiply(cube[t], four)
        inv = inv_goldschmidt(engine, den, kp, hi=4.0, iters=iters)
        sc_t = score[t]
        if 0 <= _lvl(sc_t) < 2 or 0 <= _lvl(inv) < 1:
            sc_t = _rf(engine, sc_t, kp) if _lvl(sc_t) < 2 else sc_t
            if _lvl(inv) < 1:
                inv = _rf(engine, inv, kp)
        out.append(engine.multiply(sc_t, inv, kp.relinearization_key))
    return out


def level_score(engine, Lq_list, Nq_list, kp, td, criterion="mgi",
                level=0, normalize=False):
    """
    레벨 전체 파티션의 점수 합.

      criterion="mgi"   분류.  left-only MGI.  **낮을수록** 좋음 (argmin)
      criterion="boost" 부스팅. (G′_L·H′_p − G′_p·H′_L)².  **높을수록** (argmax)
                        Lq = [G′_L, H′_L], Nq = [G′_p, H′_p]
                        normalize=True 면 normalize_score 로 [0,1] 보장
                        (Σ_p H′_p³ 로 나눔 — 휴리스틱 없음, Goldschmidt 1회)

    ★ 부스팅에 분류 MGI 를 쓰면 안 된다 — 평문 실측 −0.025~−0.127 이고
      깊을수록 악화한다.
    """
    score = None
    for Lq, Nq in zip(Lq_list, Nq_list):
        part = (mgi_left_only(engine, Lq, Nq, kp, td) if criterion == "mgi"
                else boost_gain(engine, Lq[0], Lq[1], Nq[0], Nq[1], kp, td))
        score = add_tiles(engine, score, part)
    if criterion == "boost" and normalize:
        score = normalize_score(engine, score, Nq_list, kp, td, level)
    return score


def encode_boost_sel(engine, kp, resid, td, unit_norm=True):
    """
    부스팅용 선택 벡터 쌍.  분류의 클래스 지시자 자리에 들어간다.
      [0] g/‖g‖  (단위노름)  → |G′_L| ≤ √H_L  (Cauchy-Schwarz)
      [1] 1/n                → H′ = H/n
    ★ 다중 클래스일수록 이득: 분류는 c 개, 부스팅은 항상 2 개.
    ★ 사전 정규화(공개 상수 1/n)로 이후 점수가 유계가 된다.  score_scale 참고.
    """
    n = td["n"]
    g = np.asarray(resid, float)
    if unit_norm:
        nrm = float(np.sqrt((g * g).sum()))
        g = g / nrm if nrm > 0 else g
    return [encode_sample_all(engine, kp, g, td),
            encode_sample_all(engine, kp, np.full(n, 1.0 / n), td)]


# ── 점수 정규화 ──
#   ★ 이론 상한만 쓰면 100~5,500배 헐거워 α 를 7~13비트 낭비한다.
#     원인:  |G_L| ≤ H_L·gmax  로 잡지만 실제는 ≈ √H_L·σ.  제곱하면 손실 2배.
#   ★ 실측(wine/cancer/synth, 분위수 8구간)은 데이터셋 간에 자릿수가 비슷해
#     공개 보정 상수를 쓸 수 있다.  안전하게 실측 최댓값의 4배로 잡는다.
# ══════════════════════════════════════════════════════════════════
# T5  파티션 마스크 갱신 (방식 C) — 개별 사용
# ══════════════════════════════════════════════════════════════════
def split_masks(engine, mask_p, b_left, kp):
    """
    왼쪽만 곱으로 만들고 오른쪽은 뺄셈.
      m_left  = m_p ⊙ b_left           (b_left = [x_j* ≤ k*])
      m_right = m_p − m_left           ★ 방식 C — 곱 없음
    """
    ml = mul_sample(engine, mask_p, b_left, kp)
    return ml, sub_sample(engine, mask_p, ml)


def extract_branch(engine, pre, j, k, kp, td):
    """
    선택된 (j,k) 의 분기 지시자 b_left(l) 를 청크별 샘플 축 벡터로 꺼낸다.
    ★ 평문 (j,k) 를 받는 임시판.  T3 확정 후 암호문 one-hot 내적으로 교체.
    """
    M, S, n_cat, per_ct = td["M"], td["S"], td["n_cat"], td["per_ct"]
    blk = j * n_cat + k
    bt, off = blk // per_ct, blk % per_ct
    keep = [0.0] * M
    keep[:S] = [1.0] * S
    keep_pt = engine.encode(keep)
    out = []
    for ch in range(td["n_chunk"]):
        v = pre[_tile(td, bt, ch)]
        if off:
            v = _rot(engine, v, kp, -off * S)        # 블록 0 으로
        v = engine.multiply(v, keep_pt)
        step = S                                      # 타일 전 블록에 복제
        while step < per_ct * S:
            v = engine.add(v, _rot(engine, v, kp, step)); step *= 2
        out.append(v)
    return out


# ══════════════════════════════════════════════════════════════════
# T4  분기 → 모델 좌표계  (symmetric 전용)
# ══════════════════════════════════════════════════════════════════
#   symmetric 은 레벨당 분기가 하나뿐이므로 모델이 D·dp 슬롯이면 된다.
#     Model[ t·dp + f ] = 1  iff  f == j_t
#     Theta[ t·dp ]     = (k_t + 1) / n_max
#   비대칭(2^D 노드)의 hard_core.extract_model 과 달리 청크가 필요 없다.

def make_split_masks(engine, td, dp, depth, seq_bits=None):
    """argmin 출력(학습 좌표계) → 모델 좌표계 변환용 마스크. 한 번만 만든다."""
    M, S, n_cat = td["M"], td["S"], td["n_cat"]
    d, n_max, per_ct = td["d"], td["n_max"], td["per_ct"]
    feat_per_ct = max(1, per_ct // n_cat)               # 타일당 feature 수

    # θ 가중치:  w[(j·n_cat+k)·n_pad] = (k+1)/n_max
    # θ 가중치는 타일마다 다르다 (split_feat 이면 타일이 k 구간을 나눠 가짐)
    wl = []
    for bt in range(td["nb_ct"]):
        lo, _ = _blk_range(td, bt)
        w = [0.0] * M
        for b in range(per_ct):
            blk = lo + b
            j, k = divmod(blk, n_cat)
            if j < d and k < n_max:
                w[b * S] = (k + 1.0) / n_max
        wl.append(engine.encode(w))

    # k 축 합 후 각 feature 그룹의 0번 블록만 남기는 마스크
    g0l = []
    for bt in range(td["nb_ct"]):
        lo, _ = _blk_range(td, bt)
        g0 = [0.0] * M
        for b in range(per_ct):
            if (lo + b) % n_cat == 0:
                g0[b * S] = 1.0
        g0l.append(engine.encode(g0))

    # feature 압축:  j_loc·(n_cat·n_pad) → j_loc   (stride 압축)
    if seq_bits is None:
        seq_bits = max(1, feat_per_ct - 1).bit_length()
    comp = (build_stride_masks(engine, min(feat_per_ct, d), n_cat * S, 1, seq_bits)
            if feat_per_ct > 1 else None)

    z0 = [0.0] * M
    z0[0] = 1.0
    return dict(w=wl, g0=g0l, comp=comp, z0=engine.encode(z0),
                blk0=engine.encode(z0),          # 타일의 0번 블록 첫 슬롯
                feat_per_ct=feat_per_ct, dp=dp, tpf=max(1, n_cat // per_ct))


def extract_split(engine, sel_tiles, sm, kp, td, level):
    """
    argmin 출력 → (Model 기여, Theta 기여).  레벨 level 자리에 놓는다.

      입력  slot[(j·n_cat+k)·S] one-hot (타일 리스트).  분류는 argmin,
            부스팅은 argmax 의 지시자.
      출력  Model  slot[level·dp + f],  Theta  slot[level·dp]

    비용: θ 는 pt곱 1 + 전체합,  onehot_j 는 k축합 log2(n_cat) + 압축(대각)
    """
    M, S, n_cat, per_ct = td["M"], td["S"], td["n_cat"], td["per_ct"]
    dp, fpc, tpf = sm["dp"], sm["feat_per_ct"], sm["tpf"]

    # ── θ ──  가중치 곱 → 블록 합 → 슬롯 0 → level·dp 로
    #   ★ argmin one-hot 은 블록 첫 슬롯에만 값이 있다.  블록 내부(S 슬롯)를
    #     돌 필요가 없으므로 log2(M) 대신 **log2(per_ct)** 회전이면 된다.
    #     per_ct=64, M=32768 이면 15회 → 6회.
    th = None
    for bt, ct in enumerate(sel_tiles):
        v = engine.multiply(ct, sm["w"][bt])
        step = S
        while step < per_ct * S:
            v = engine.add(v, _rot(engine, v, kp, -step)); step *= 2
        th = v if th is None else engine.add(th, v)
    th = engine.multiply(th, sm["z0"])
    if level:
        th = _rot(engine, th, kp, level * dp)

    # ── onehot_j ──
    model = None
    if not td["split_feat"]:
        # feature 가 타일 안에 통째로: k 축 합 → 그룹 0번 → stride 압축
        for bt, ct in enumerate(sel_tiles):
            v = ct
            step = S
            while step < n_cat * S:
                v = engine.add(v, _rot(engine, v, kp, -step)); step *= 2
            v = engine.multiply(v, sm["g0"][bt])
            v = apply_stride(engine, v, kp, sm["comp"])   # slot[j_loc]
            sh = level * dp + bt * fpc
            if sh:
                v = _rot(engine, v, kp, sh)
            model = v if model is None else engine.add(model, v)
    else:
        # ★ feature 가 tpf 개 타일에 걸침.  타일 안 블록을 슬롯 0 으로 모으고,
        #   그 feature 의 타일들을 더한 뒤 slot[level·dp + j] 로 보낸다.
        #   (g0 로 그룹 0번만 남기면 k≠0 인 타일이 통째로 지워진다 — 이전 버그)
        for f0 in range(0, len(sel_tiles), tpf):
            acc = None
            for o in range(tpf):
                bt = f0 + o
                if bt >= len(sel_tiles):
                    break
                v = sel_tiles[bt]
                step = S
                while step < per_ct * S:                  # 타일 안 블록 합
                    v = engine.add(v, _rot(engine, v, kp, -step)); step *= 2
                v = engine.multiply(v, sm["blk0"])        # 슬롯 0 만
                acc = v if acc is None else engine.add(acc, v)
            j = f0 // tpf
            sh = level * dp + j
            if sh:
                acc = _rot(engine, acc, kp, sh)
            model = acc if model is None else engine.add(model, acc)
    return model, th


# ══════════════════════════════════════════════════════════════════
# T6  리프값  (부스팅: −lr·G/(H+λ))
# ══════════════════════════════════════════════════════════════════
#   나눗셈 없는 근사(기대 리프 크기로 나누기)는 깊이가 깊을수록 나빠진다
#   (평문 실측 D=3 −0.02, D=4 −0.03).  우리 주장이 깊이이므로 쓸 수 없다.
#   ★ λ = 기대 리프 크기(n/2^D, 공개) 로 잡으면 H+λ 의 동적 범위가
#     (2^D+1):1 로 좁아져 Goldschmidt 가 ~10회면 수렴한다.
#     ct·ct 곱 2k−2 ≈ 18회(≈180ms) — 트리 하나당 0.4% 수준.

def gold_iters(depth, bits=20):
    """
    필요한 Goldschmidt 반복 수.
      H+λ 의 동적 범위 = (n+λ)/λ = 2^D + 1  →  m = D
      상대오차 2^-bits 를 얻으려면  iters ≳ m + log2(bits)
    실측: D=4→10, D=6→11, D=10→15 에서 1e-7 이하.
    """
    import math
    return max(8, depth + int(math.ceil(math.log2(max(2, bits)))))


def inv_goldschmidt(engine, x, kp, hi, iters=10):
    """
    1/x  (x ∈ (0, hi]).  스케일드 Goldschmidt.

        z = x/hi ∈ (0,1],   e = 1−z,   1/z = Π_{i} (1 + e^{2^i})

    깊이 ≈ (iters−1) + log2(iters),   ct·ct 곱 = 2·iters − 2
    ★ x 가 0 에 가까우면 수렴이 느리다 → 반드시 λ 를 더해 하한을 확보할 것.
    """
    sc = engine.slot_count
    rlk = kp.relinearization_key
    one = engine.encode([1.0] * sc)
    inv_hi = engine.encode([1.0 / hi] * sc)

    z = engine.multiply(x, inv_hi)
    e = engine.subtract(one, z)
    terms = [engine.add(one, e)]
    for _ in range(iters - 1):
        e = engine.multiply(e, e, rlk)                 # e^{2^i}
        terms.append(engine.add(one, e))
    while len(terms) > 1:                              # 이진 트리 곱 (깊이 절감)
        nxt = [engine.multiply(terms[i], terms[i + 1], rlk)
               for i in range(0, len(terms) - 1, 2)]
        if len(terms) % 2:
            nxt.append(terms[-1])
        terms = nxt
    return engine.multiply(terms[0], inv_hi)           # 1/x = (1/hi)·(1/z)


def leaf_stats(engine, masks, gvec, kp, td, dp):
    """
    리프별 합 G[ℓ] 과 크기 H[ℓ] 을 리프 슬롯 ℓ·dp 에 모은다.

      masks : 레벨 D 의 파티션 마스크 (2^D 개, 각각 청크 리스트)
      gvec  : **음의 gradient**(= 제곱손실이면 잔차 y − F) 샘플 축 벡터

    ★ 부호 규약을 여기서 못박는다.  leaf_values 가 +lr·G/(H+λ) 를 돌려주므로
      gvec 에는 **잔차(y − F)** 를 넣는다.  gradient(∂L/∂F = F − y)를 넣으면
      부스팅이 반대 방향으로 간다.

    비용: 리프당 (곱 1 + log2(S) 회전) × 청크 + 배치 회전 1
    """
    S = td["S"]
    rlk = kp.relinearization_key
    z0 = [0.0] * td["M"]; z0[0] = 1.0
    z0 = engine.encode(z0)
    G = H = None
    for l, mp in enumerate(masks):
        gs = hs = None
        for ch in range(td["n_chunk"]):
            m = mp[ch]
            p = engine.multiply(m, gvec[ch], rlk)
            q = m
            step = 1
            while step < S:                            # 블록 하나만 합하면 된다
                p = engine.add(p, _rot(engine, p, kp, -step))
                q = engine.add(q, _rot(engine, q, kp, -step))
                step *= 2
            gs = p if gs is None else engine.add(gs, p)
            hs = q if hs is None else engine.add(hs, q)
        gs = engine.multiply(gs, z0)
        hs = engine.multiply(hs, z0)
        if l:
            gs = _rot(engine, gs, kp, l * dp)
            hs = _rot(engine, hs, kp, l * dp)
        G = gs if G is None else engine.add(G, gs)
        H = hs if H is None else engine.add(H, hs)
    return G, H


def leaf_values(engine, G, H, kp, n, depth, lr=0.1, lam=None, iters=None):
    """
    leaf[ℓ] = +lr · G[ℓ] / (H[ℓ] + λ)

    ★ 부호 규약 — G 는 **잔차 합**(= 음의 gradient 합)이다.
        제곱손실  L = ½(y−F)²  →  ∂L/∂F = F − y = −(잔차)
        부스팅    F_t = F_{t−1} + lr·h_t,  h_t ≈ 음의 gradient = 잔차
      따라서 잔차를 넣고 **+lr** 을 곱하는 것이 맞다.
      (gradient 를 넣는다면 −lr 이어야 한다 — 혼동하기 쉬운 지점)

    기호
        lr    학습률          공개 스칼라
        G     리프 내 잔차 합  암호문
        H     리프 크기        암호문
        λ     정규화          공개.  기본 n/2^D (기대 리프 크기)
        n     전체 샘플 수     공개
        depth 트리 깊이 D      공개

      λ 기본값 = 기대 리프 크기 n/2^D.  H+λ ∈ [λ, n+λ] → 범위 (2^D+1):1
      → 반복 수는 gold_iters(depth) 로 자동 (D=4→10, D=6→11, D=10→15).
    ★ 깊이 ≈ iters + log2(iters) 를 소모하므로, 깊은 트리에서는 이 앞에
      refresh 가 필요할 수 있다.
    """
    sc = engine.slot_count
    L = 1 << depth
    if lam is None:
        lam = max(1.0, n / L)
    if iters is None:
        iters = gold_iters(depth)
    denom = engine.add(H, engine.encode([lam] * sc))
    inv = inv_goldschmidt(engine, denom, kp, hi=float(n) + lam, iters=iters)
    v = engine.multiply(G, inv, kp.relinearization_key)
    return engine.multiply(v, engine.encode([float(lr)] * sc))   # ★ +lr


# ══════════════════════════════════════════════════════════════════
# T3  argmax / argmin  — league (전쌍 비교)
# ══════════════════════════════════════════════════════════════════
#   ind_i = Π_{j≠i} [ s_i ≥ s_j ]        (argmin 이면 부호 반전)
#
#   ★ sign 을 SIMD 로 **1회**만 부른다.
#     토너먼트(HBDT FindMinPos) ⌈log₄ ns⌉ 회, Masalha 계층적 ⌈log₂log₂ ns⌉ 회.
#   ★ α(top) 만 있으면 된다 — 근접 동점은 전부 패자끼리라 결과에 안 나타난다.
#         i=최댓값 : 모든 인자가 깨끗한 1  → ind=1
#         i≠최댓값 : [s_i ≥ s_max] 가 깨끗한 0 → ind=0 (흐린 인자는 소멸)
#   ★ 제약:  ns² ≤ M.   mtry=√d 로 후보를 줄여 맞춘다
#              (cancer ns 210 → 35, ns² 1,225)
#
#   ★ **α = 8 이 작동점** (실제 FHE 4/4 깨끗한 one-hot).
#     측정된 필요 α 는 2~7 이고, α=8 의 첫 구간 깊이가 refresh 회복(10) 안에
#     들어가는 마지막 값이다.  α≥10 은 레벨 19 가 필요해 불가.
ARGMAX_ALPHA = 8
# ★ Appendix E 는 부트 없는 합성을 가정한다.  성분 사이에 sign_bootstrap 을
#   넣으면 더 낮은 차수로 같은 α 에 도달한다 → DP 로 다시 찾는다.
#   α=10 에서 깊이 17→13, 곱 29→25.  탐색은 캐시되므로 1회만 든다.

def cand_index(td, mtry_feats=None):
    """
    유효 후보 (j,k) 목록과 선형 좌표 정보.

    ★ 선형 인덱스를 **i = jj·ntp + k** 로 패딩한다 (ntp = 2^⌈log2(n_max−1)⌉).
      그러면 argmax 결과를 타일로 되돌리지 않고 **선형 좌표에서 바로**
      θ / onehot_j 를 뽑을 수 있다 (extract_split_linear).
      k 축 합이 블록합(2의 거듭제곱)으로 성립하기 때문.

    반환: (cands, ns, idx, ntp)
      cands[t] = (j, k),  idx[t] = 선형 슬롯,  ns = 유효 후보 수
    """
    d, n_max = td["d"], td["n_max"]
    feats = list(range(d)) if mtry_feats is None else list(mtry_feats)
    ntp = 1 << (max(1, n_max - 2)).bit_length()          # ≥ n_max−1
    cands, idx = [], []
    for jj, j in enumerate(feats):
        for k in range(n_max - 1):
            cands.append((j, k)); idx.append(jj * ntp + k)
    return cands, len(cands), idx, ntp


# 차분행렬 A 구성의 stride 확장에서 순차 처리할 비트 수.
#   0 → BSGS (대각 span 개, 깊이 1) / 큰 값 → 순차 (회전 log2(span), 깊이 그만큼)
#   span=64 기준
#     seq=0  회전 64,  깊이 1
#     seq=1  회전 33,  깊이 2   ★ 레벨이 빠듯하므로 기본
#     seq=4  회전  8,  깊이 5
#   레벨 1 이 회전 수십 개보다 비싸다 (argmax 전체가 20 레벨을 먹는다).
#   ★ 0 으로 둔다 — refresh 회복(10) 안에 α=6 을 넣으려면 A/B 깊이가 2 여야 한다.
SPREAD_SEQ_BITS_AM = 0

# 차분행렬 A/B 구성이 먹는 깊이:  stride(seq+1) + keep 마스크(1)
AB_DEPTH = SPREAD_SEQ_BITS_AM + 2


def binning_report(slot_count=32768, n_max_list=(4, 6, 8, 9, 10, 16, 18, 32)):
    """
    n_max 가 비용 두 축에 어떻게 들어가는지.

      T1 비용   n_cat = 2^⌈log2 n_max⌉        타일 수 nct ∝ d·n_cat
      argmax    ntp   = 2^⌈log2(n_max−1)⌉    span = n_feat·ntp,  nsp² ≤ M

    두 축이 **다른 값**을 쓴다는 게 핵심이다 (n_cat 은 n_max, ntp 는 n_max−1).
      n_max=8  → n_cat 8,  ntp 8   두 축 최소  ★ 안전한 기본값
      n_max=9  → n_cat 16, ntp 8   T1 만 2배
      n_max=10 → n_cat 16, ntp 16  feature 수용량 절반
      n_max=4  → n_cat 4,  ntp 4   두 축 더 작다 (정확도 영향 −0.008 이내)
    """
    nsp_max = 1 << ((int(slot_count ** 0.5) - 1).bit_length() - 1)
    rows = []
    for nm in n_max_list:
        ncat, ntp = p2(nm), p2(nm - 1)
        rows.append((nm, ncat, ntp, max(1, nsp_max // ntp)))
    return rows


def recommend_n_max(d, slot_count=32768, want_feats=None):
    """
    d 개 feature 중 want_feats 개를 argmax 후보로 담으려면 n_max 가 얼마여야 하는가.

    ★ 실측상 정확도는 n_max 4~32 에서 평평하다 (exp_nmax.py) → 비용으로 정한다.
    ★ 분위수 구간화를 쓸 것 (등폭 대비 n_max=4 에서 +0.08).
    """
    nsp_max = 1 << ((int(slot_count ** 0.5) - 1).bit_length() - 1)
    need = want_feats if want_feats else d
    for nm in (8, 7, 6, 5, 4):
        if nsp_max // p2(nm - 1) >= need:
            return nm
    return 4


def check_binning(td, n_feat=None, slot_count=None, verbose=True):
    """
    현재 (d, n_max) 로 league argmax 가 성립하는지 점검하고 대안을 제시한다.
    반환: (ok, 메시지)
    """
    M = slot_count or td["M"]
    d, n_max = td["d"], td["n_max"]
    nf = n_feat or d
    ntp = p2(n_max - 1)
    span = nf * ntp
    nsp = p2(span)
    if nsp * nsp <= M:
        return True, f"OK (ntp={ntp}, span={span}, nsp={nsp})"
    nsp_max = 1 << ((int(M ** 0.5) - 1).bit_length() - 1)
    fit = max(1, nsp_max // ntp)
    alt = recommend_n_max(d, M, want_feats=nf)
    msg = (f"nsp²={nsp*nsp} > M={M}.  "
           f"현재 n_max={n_max}(ntp={ntp}) 로는 feature {fit}개까지.  "
           f"n_max={alt} 로 구간화하면 {nsp_max // p2(alt-1)}개까지 담긴다 "
           f"(정확도 영향 무시할 수준 — exp_nmax.py).  "
           f"또는 auto_mtry 로 feature 를 {fit}개로 줄일 것.")
    return False, msg


def auto_mtry(engine, td, seed=0):
    """
    nsp² ≤ M 을 만족하는 최대 feature 수를 골라 무작위 부분추출.
      span = n_feat·ntp,  nsp = 2^⌈log2 span⌉,  제약 nsp² ≤ M
    ★ mtry 는 정확도(−0.004~−0.011)와 T1 비용(65%)에도 걸리는 축이라,
      제약을 만족하는 최대값을 쓰되 √d 이하로도 줄일 수 있다.
    """
    M, d, n_max = td["M"], td["d"], td["n_max"]
    ntp = 1 << (max(1, n_max - 2)).bit_length()
    nsp_max = 1 << ((int(M ** 0.5) - 1).bit_length() - 1)
    n_feat = max(1, min(d, nsp_max // ntp))
    if n_feat >= d:
        return None                                   # 전부 써도 들어간다
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(d, n_feat, replace=False).tolist())


def make_argmax_masks(engine, td, mtry_feats=None):
    """
    league argmax 용 마스크.  후보 집합이 바뀌면(mtry) 다시 만든다.

      compact : 타일 좌표 slot[(j·n_cat+k)·S] → 연속 슬롯 [0, ns)
      spread  : slot[i] → slot[i·ns]          (차분 행렬의 행 축)
      expand  : slot[i] → 타일 좌표            (extract_split 입력으로)
    """
    M, S, n_cat, per_ct = td["M"], td["S"], td["n_cat"], td["per_ct"]
    cands, ns, idx, ntp = cand_index(td, mtry_feats)
    n_feat = len(mtry_feats) if mtry_feats else td["d"]
    span = n_feat * ntp                                   # 선형 좌표 길이
    # ★ 블록 곱(rotate-and-multiply)은 블록 크기가 2의 거듭제곱일 때만 성립.
    #   ns 가 아니면 회전이 옆 블록을 침범한다 → nsp 로 패딩하고 남는 슬롯은 1 로 둔다.
    nsp = 1 << (max(1, span - 1)).bit_length()
    if nsp * nsp > M:
        ok, msg = check_binning(td, n_feat, M)
        raise AssertionError(msg)

    # ── compact: 타일별 대각 (shift 별로 묶음) ──
    comp = []
    for bt in range(td["nb_ct"]):
        lo, hi = _blk_range(td, bt)
        byshift = {}
        for t_, (j, k) in enumerate(cands):
            blk = j * n_cat + k
            if not (lo <= blk < hi):
                continue
            src = (blk - lo) * S
            byshift.setdefault(idx[t_] - src, []).append(idx[t_])
        dg = []
        for sh, tgts in byshift.items():
            mv = [0.0] * M
            for t_ in tgts:
                mv[t_] = 1.0
            dg.append((engine.encode(mv), sh))
        comp.append(dg)

    # ── spread: slot[i] → slot[i·nsp] ──
    #   ★ 대각을 span 개 쓰면 회전이 그만큼 든다.  build_stride_masks 의
    #     순차 doubling 을 쓰면 log2(span) 단계로 줄어든다 (깊이는 그만큼 증가).
    #     seq_bits 로 회전/깊이 균형을 잡는다.
    spread = build_stride_masks(engine, span, 1, nsp, SPREAD_SEQ_BITS_AM)

    # ── expand: slot[i] → 타일 좌표 ──
    exp = []
    for bt in range(td["nb_ct"]):
        lo, hi = _blk_range(td, bt)
        byshift = {}
        for t_, (j, k) in enumerate(cands):
            blk = j * n_cat + k
            if not (lo <= blk < hi):
                continue
            dst = (blk - lo) * S
            byshift.setdefault(dst - idx[t_], []).append(dst)
        dg = []
        for sh, tgts in byshift.items():
            mv = [0.0] * M
            for t_ in tgts:
                mv[t_] = 1.0
            dg.append((engine.encode(mv), sh))
        exp.append(dg)

    # ── back: slot[i·nsp] → slot[i]  (argmax 결과를 선형 좌표로) ──
    #   ★ 이전에는 onehot_to_tiles 가 호출마다 encode 했다 — 구간 비용의 39%.
    bs = {}
    for i in idx:
        bs.setdefault(i - i * nsp, []).append(i)
    back = []
    for sh, tgts in bs.items():
        mv = [0.0] * M
        for t_ in tgts:
            mv[t_] = 1.0
        back.append((engine.encode(mv), sh))

    # 유효 슬롯 (i,j 둘 다 < ns) 만 1.  나머지(패딩)는 곱에서 1 이 되도록 따로 준다.
    keep = [0.0] * M
    pad = [0.0] * M
    dmask = [0.0] * M
    first = [0.0] * M
    iset = set(idx)
    for i in range(nsp):
        for j in range(nsp):
            if i in iset and j in iset:
                keep[i * nsp + j] = 1.0
                if i == j:
                    dmask[i * nsp + j] = 1.0
            else:
                pad[i * nsp + j] = 1.0      # 패딩 → b=1 로 채워 곱에 무영향
        if i in iset:
            first[i * nsp] = 1.0
    # ★ 평문끼리는 add/subtract 가 안 된다 → 여기서 미리 합쳐 하나로 인코딩
    dpad = [dmask[i] + pad[i] for i in range(M)]        # 대각 1, 패딩 1
    return dict(ns=ns, nsp=nsp, span=span, ntp=ntp, idx=idx, n_feat=n_feat,
                feats=list(mtry_feats) if mtry_feats else list(range(td["d"])),
                cands=cands, comp=comp, spread=spread, exp=exp, back=back,
                diag=engine.encode(dmask), keep=engine.encode(keep),
                pad=engine.encode(pad), diag_pad=engine.encode(dpad),
                first=engine.encode(first))


def recommended_tiebreak(ns, alpha, top_gap=2.0 ** -6):
    """
    공개 타이브레이커 크기.  ★ 기본은 0 (끔) — 아래 긴장 때문.

      dead zone 을 넘으려면   ρ ≥ ns·2^-α        (안 그러면 무효)
      순위를 안 흔들려면      ρ ≪ 최고 간격 ≈ 2^-6
      둘 다 만족하려면        α > log2(ns) + 6   (ns=35 이면 α > 11)

    즉 **α ≥ 12 가 아니면 타이브레이커가 오히려 해롭다** (실측 확인).
    α 가 충분치 않으면 끄고, 정확한 동점은 multi-hot 을 허용한다.
      · 정확한 동점은 중복/퇴화 후보에서만 생기고 드물다
      · 같은 파티션을 만드는 후보끼리면 혼합돼도 무해하다
    """
    need = ns * 2.0 ** (-alpha)
    return need if need < top_gap else 0.0


def _sec(engine, name):
    """
    계측용 구간.  CountingEngine 이면 section 을, 아니면 아무것도 안 한다.
    운영 코드에 부담을 주지 않으면서 병목을 나눠 볼 수 있다.
    """
    from contextlib import nullcontext
    f = getattr(engine, "section", None)
    return f(name) if callable(f) else nullcontext()


def _apply_diag(engine, ct, kp, diags):
    """(마스크, shift) 목록을 원본에서 출발해 누적.  깊이 1."""
    out = None
    for mv, sh in diags:
        term = engine.multiply(_rot(engine, ct, kp, sh), mv)
        out = term if out is None else engine.add(out, term)
    return out


def compact_scores(engine, score_tiles, am, kp, td, tiebreak=0.0):
    """
    타일 좌표의 점수 → 연속 슬롯 [0, ns).  깊이 1.

    tiebreak ρ > 0 이면  score_i += ρ·i/ns  (공개 평문 덧셈, 깊이 0).
    ★ **정확히 같은 점수**를 가르기 위한 것이다.  α 를 아무리 올려도
      완전 동점은 못 가르므로 지시자가 두 곳에서 1 이 되어 one-hot 이 깨진다.
    ★ ρ 는 **인접 인덱스 간 간격**이 dead zone 을 넘어야 효과가 있다:
          ρ/ns ≥ 2^-α    →    ρ ≥ ns·2^-α        (recommended_tiebreak 참고)
      더 작으면 타이브레이커가 dead zone 에 묻혀 무효다 (실측 확인).
    ★ 대가: 점수에 최대 ρ 만큼의 왜곡.  평문 실측에서 ρ=0.0625 는 무해했다.
    """
    out = None
    for bt, ct in enumerate(score_tiles):
        if not am["comp"][bt]:
            continue
        v = _apply_diag(engine, ct, kp, am["comp"][bt])
        out = v if out is None else engine.add(out, v)
    if tiebreak:
        ns = am["ns"]
        tb = [0.0] * td["M"]
        for i in range(ns):
            tb[i] = tiebreak * i / ns
        out = engine.add(out, engine.encode(tb))
    return out


def argmax_league(engine, s, am, kp, sign_spec, maximize=True,
                  secret_key=None, clean_every=2, clean_out=6):
    """
    전쌍 비교로 one-hot 지시자를 만든다.

    ⚠️ **현재 상태 (2026-08-30)**
      · 레벨 0 은 통과하나 깊은 레벨에서 트리가 평문과 어긋난다.
      · 원인: SCORE_TIGHTEN 이 데이터셋·레벨마다 맞아야 하는데 고정 표를 쓴다.
        점수가 너무 크면 minimax 발산, 작으면 dead zone.
      · 해결 전까지는 test_s5_endtoend 의 ARGMAX_MODE="plain" 을 쓸 것.
      · 다음 작업: α 측정 실험으로 (α, SCORE_TIGHTEN) 을 데이터셋별로 확정.

        D[i·ns+j] = s_i − s_j          (maximize=False 면 부호 반전)
        b = (sign(D)+1)/2              ★ sign SIMD 1회
        b[i·ns+i] := 1                 대각은 자기 자신
        ind_i = Π_j b[i·ns+j]          블록 내 곱, 깊이 log2(ns)

    깊이 ≈ 2 + sdep + 1 + log2(ns).

    ★ 입력 레벨이 sdep+2 미만이면 먼저 refresh 한다.
      ⚠️ **학습에는 full bootstrap_key 가 필요하다.**
        small 키 refresh 는 레벨 10 까지만 회복하는데 α=8 의 sdep 이 15 라
        α ≤ 4 밖에 못 쓴다.  keypack.setup_fhe_engine(create_full_bootstrap_key=True).
    """
    from Hard_Tree.desilo.hard_core import _sign01
    from Hard_Tree.desilo.util.keypack import refresh as _refresh
    from Hard_Tree.desilo.minimax_sign import sign_depth
    rlk = kp.relinearization_key
    ns = am["ns"]

    # ★ sign 은 성분마다 표준 bootstrap 으로 레벨이 회복되고, 마지막에
    #   sign_bootstrap 이 붙는다.  따라서 입력에 필요한 건
    #   **A/B 구성(AB_DEPTH) + sign_depth(가장 깊은 성분 + 부트 입력분)** 뿐이고,
    #   sign 이후(대각보정·블록곱·cleaning)는 그 부트가 회복해 준 레벨에서 돈다.
    #   ★ 필요한 건 **첫 부트까지의 구간** 깊이다.  그 구간이 부트로 끝나면
    #     sign_bootstrap 입력 레벨(stage_count, 최소 3)도 남아 있어야 한다.
    need_in = AB_DEPTH + sign_depth(sign_spec)
    lv = _lvl(s)
    if 0 <= lv < need_in:
        import gc; gc.collect()                  # ★ bootstrap 은 메모리를 크게 쓴다
        with _sec(engine, "am:입력 refresh"):
            s = _refresh(engine, s, kp)          # small key 로 충분 (회복 10)
        lv = _lvl(s)
        if lv < need_in:
            # ★ refresh 로도 부족하면 α 를 낮춘다.  첫 부트가 걸리는 α 는
            #   첫 구간이 짧아 refresh 회복(10) 안에 들어간다.
            a2, sp2 = pick_alpha(lv)
            if sp2 is None:
                raise RuntimeError(
                    f"레벨 {lv} 에 들어가는 α 가 없다 (need {need_in}, AB={AB_DEPTH}). "
                    f"점수 레벨을 높이거나 mtry 로 ns 를 줄이세요.")
            sign_spec = sp2

    nsp = am["nsp"]
    with _sec(engine, "am:차분행렬 A/B"):
        A = apply_stride(engine, s, kp, am["spread"])    # s_i → slot i·nsp
        step = 1
        while step < nsp:                                  # 블록 안에 복제
            A = engine.add(A, _rot(engine, A, kp, step)); step *= 2
        B = s
        step = nsp
        while step < nsp * nsp:                            # 벡터를 nsp 번 타일
            B = engine.add(B, _rot(engine, B, kp, step)); step *= 2
        B = engine.multiply(B, am["keep"])
        A = engine.multiply(A, am["keep"])
        D = engine.subtract(A, B) if maximize else engine.subtract(B, A)
        del A, B                                       # ★ ns² 슬롯짜리 — 즉시 해제

    # ★ minimax_sign01 이 성분 사이 표준 bootstrap + 마지막 sign_bootstrap 을
    #   내장한다.  여기서 따로 부트를 걸 필요가 없다.
    b = _sign01(engine, D, kp, secret_key, spec=sign_spec)

    #   대각은 1, 패딩도 1 (곱에 무영향), 나머지는 비교 결과
    #   ★ 평문끼리 add/subtract 는 불가 → ct 쪽에서 빼고, diag+pad 는 미리 합쳐둔 것
    b = engine.add(engine.subtract(engine.multiply(b, am["keep"]),
                                   engine.multiply(b, am["diag"])),
                   am["diag_pad"])

    # 블록 내 곱 (깊이 log2 ns).
    #   ★ ind_max = Π_j b_j 라 b 가 1 에서 조금만 벗어나도 ns 제곱으로 증폭된다
    #     (b=0.9, ns=35 이면 0.9^35 = 0.025).  bit_cleaning 으로 {0,1} 로
    #     되돌려 증폭을 끊는다 (h(x)=3x²−2x³, 이차 수렴, 레벨 2).
    #   ★ clean_every 로 간격 조절.  매 단계(1)면 log2(nsp)×2 레벨을 먹어
    #     argmax 전체 예산을 압박한다.  2 면 절반, 0 이면 끄기.
    from Hard_Tree.desilo.util.keypack import bit_cleaning
    step, k = 1, 0
    while step < nsp:
        with _sec(engine, "am:블록곱"):
            if 0 <= _lvl(b) < 2:
                with _sec(engine, "am:곱루프 refresh"):
                    b = _refresh(engine, b, kp)
            b = engine.multiply(b, _rot(engine, b, kp, -step), rlk)
        k += 1
        if clean_every and step * 2 < nsp and k % clean_every == 0:
            if 0 <= _lvl(b) < 3:
                with _sec(engine, "am:곱루프 refresh"):
                    b = _refresh(engine, b, kp)
            with _sec(engine, "am:bit_cleaning"):
                b = bit_cleaning(engine, b, kp, n_iters=1)
        step *= 2
    ind = engine.multiply(b, am["first"])                  # slot i·nsp 에 ind_i
    del b                                                  # ★ nsp² 슬롯 — 즉시 해제
    # ★ 최종 정리 — 지시자가 {0,1} 에서 벗어나면 Model 이 두 후보의 **혼합**이 되고,
    #   추론에서 축정렬이 깨져 binning 마진 보장(α=4~8 의 근거)이 무너진다.
    #   (실측: 트리는 맞는데 추론이 8/20 으로 떨어졌다)
    #   ★ 근접 동점이면 0.58/0.42 로 갈라진다.  h(x)=3x²−2x³ 는 0.5 근처에서
    #     천천히 수렴한다: 0.5757 → .613 → .666 → .740 → .832 → .925 → .984
    #     실측상 **6회**에서 4/4 통과 (5회는 19/20).  반복마다 레벨 2 를 먹으므로
    #     중간에 refresh 가 들어간다 — 비용은 test_s8_argmax_prof 로 확인할 것.
    for _ in range(clean_out):
        if 0 <= _lvl(ind) < 3:                     # cleaning 1회 = 레벨 2
            with _sec(engine, "am:출력 refresh"):
                ind = _refresh(engine, ind, kp)
        with _sec(engine, "am:출력 cleaning"):
            ind = bit_cleaning(engine, ind, kp, n_iters=1)
    import gc; gc.collect()                            # ★ GPU 메모리 회수
    return ind


def onehot_to_tiles(engine, ind_blocked, am, kp, td, min_level=6):
    """
    argmax_league 결과(slot i·ns)를 타일 좌표로 되돌린다.
      ① slot i·ns → slot i   ② 타일 좌표로 확장

    ★ 지시자는 league 곱(깊이 log2 ns)을 거쳐 레벨이 낮다.  extract_split 과
      그 뒤 모델 누적이 이어지므로 여기서 한 번 refresh 한다.
      (지시자는 {0,1} 이라 sign_bootstrap 은 못 쓴다 — 일반 refresh)
    """
    from Hard_Tree.desilo.util.keypack import refresh as _refresh
    M = td["M"]
    if 0 <= _lvl(ind_blocked) < min_level:
        ind_blocked = _refresh(engine, ind_blocked, kp)
    lin = _apply_diag(engine, ind_blocked, kp, am["back"])   # ① i·nsp → i
    zero = am.get("zero")
    if zero is None:
        zero = engine.encode([0.0] * M); am["zero"] = zero
    return [(_apply_diag(engine, lin, kp, am["exp"][bt]) if am["exp"][bt]  # ② 타일
             else engine.multiply(lin, zero))
            for bt in range(td["nb_ct"])]


# ══════════════════════════════════════════════════════════════════
# T4-선형  argmax 결과를 **선형 좌표에서 바로** 모델 좌표계로
# ══════════════════════════════════════════════════════════════════
#   기존:  선형 one-hot → expand(대각 ~ns 개) → 타일 좌표 → extract_split
#   개선:  선형 one-hot → **바로** Model/Theta          ← expand 생략
#
#   선형 인덱스가 i = jj·ntp + k 로 패딩돼 있어 k 축 합이 블록합으로 성립한다.
#     θ        평문곱 1 + 전체합 log2(span)
#     onehot_j k축 블록합 log2(ntp) → slot[jj·ntp] → stride 압축 → slot[j]

# extract_split_linear 의 stride 압축 깊이.  깊이 seq+1, 회전 seq+2^(log2 d − seq).
#   레벨 예산이 빠듯하므로 1 (깊이 2).  회전이 늘어도 레벨이 더 비싸다.
SPLIT_SEQ_BITS_LIN = 1


def make_split_masks_linear(engine, am, td, dp, depth, seq_bits=None):
    """extract_split_linear 용 마스크.  후보 집합(mtry)이 바뀌면 다시 만든다."""
    M, ntp, span = td["M"], am["ntp"], am["span"]
    n_max, feats = td["n_max"], am["feats"]

    w = [0.0] * M                                   # θ 가중치 (k+1)/n_max
    g0 = [0.0] * M                                  # feature 그룹 0번만
    for t_, (j, k) in enumerate(am["cands"]):
        w[am["idx"][t_]] = (k + 1.0) / n_max
    for jj in range(len(feats)):
        g0[jj * ntp] = 1.0

    z0 = [0.0] * M; z0[0] = 1.0
    if seq_bits is None:
        seq_bits = SPLIT_SEQ_BITS_LIN        # 레벨이 빠듯하므로 얕게
    comp = build_stride_masks(engine, len(feats), ntp, 1, seq_bits)

    # jj → 실제 feature j 로 되돌리는 마스크 (mtry 로 부분추출한 경우)
    sc_map = [0.0] * M
    for jj, j in enumerate(feats):
        sc_map[j] = 1.0
    return dict(w=engine.encode(w), g0=engine.encode(g0), z0=engine.encode(z0),
                comp=comp, dp=dp, feats=feats, ntp=ntp, span=span,
                fmap=engine.encode(sc_map))


def extract_split_linear(engine, ind_lin, slm, am, kp, td, level, min_level=6):
    """
    선형 one-hot(slot[i], i = jj·ntp + k) → (Model 기여, Theta 기여).

    ★ 타일 좌표로 되돌리지 않는다 (expand 생략 — 계측상 argmax 비용의 35%).
    ★ 결과가 Model 로 누적되어 추론까지 살아남아야 하므로, 입력 레벨이 낮으면
      먼저 refresh 한다 (지시자는 {0,1} 이라 sign_bootstrap 은 못 쓴다).
    """
    from Hard_Tree.desilo.util.keypack import refresh as _rf
    M, ntp, span = td["M"], slm["ntp"], slm["span"]
    dp = slm["dp"]
    if 0 <= _lvl(ind_lin) < min_level:
        ind_lin = _rf(engine, ind_lin, kp)

    # ── θ ──  가중치 곱 → 전체합 → 슬롯 0 → level·dp
    th = engine.multiply(ind_lin, slm["w"])
    step = 1
    while step < span:
        th = engine.add(th, _rot(engine, th, kp, -step)); step *= 2
    th = engine.multiply(th, slm["z0"])
    if level:
        th = _rot(engine, th, kp, level * dp)

    # ── onehot_j ──  k축 블록합 → 그룹 0번 → stride 압축 → level·dp + j
    v = ind_lin
    step = 1
    while step < ntp:
        v = engine.add(v, _rot(engine, v, kp, -step)); step *= 2
    v = engine.multiply(v, slm["g0"])
    v = apply_stride(engine, v, kp, slm["comp"])       # slot[jj]
    if len(slm["feats"]) < td["d"]:
        # mtry 로 뽑은 경우 jj → j 로 흩뿌린다 (대각 mtry 개)
        dg = []
        for jj, j in enumerate(slm["feats"]):
            mv = [0.0] * M
            mv[j] = 1.0
            dg.append((engine.encode(mv), j - jj))
        v = _apply_diag(engine, v, kp, dg)
    if level:
        v = _rot(engine, v, kp, level * dp)
    return v, th


# ══════════════════════════════════════════════════════════════════
# T4-b  부스팅 라운드 — 암호문에서 잔차 갱신
# ══════════════════════════════════════════════════════════════════
#   부스팅이 성립하려면 라운드마다
#       pred_i = Leaf[ℓ(i)]        리프값을 **샘플 공간**으로 흩뿌리고
#       g_i   ← g_i − lr·pred_i    잔차를 갱신
#   해야 한다.  둘 다 암호문에서 이뤄져야 클라이언트 왕복이 없다.
#
#   ★ 어려운 점: Leaf 는 **모델 공간**(슬롯 ℓ·dp), masks 는 **샘플 공간**(슬롯 i·S).
#     암호화된 인덱스로 gather 하는 건 비싸다.
#   ★ 해법: 리프마다 Leaf_ℓ 을 전 슬롯에 broadcast 한 뒤 masks[ℓ] 로 거른다.
#       Leaf_ℓ_bcast = full_sum(Leaf ⊙ onehot_{ℓ·dp})     회전 log2(M)
#       pred        = Σ_ℓ Leaf_ℓ_bcast ⊙ masks[ℓ]        ct·ct 곱 L 회
#     L=2^D 개 리프에 대해 L·log2(M) 회전 + L 곱.  D=4, M=32768 이면
#     16×15=240 회전(≈1.5 s) + 16 곱.  라운드당 한 번이라 감당된다.


def _full_sum(engine, ct, kp, M):
    """전 슬롯 합을 모든 슬롯에 복제 (rotate-and-add, log2(M) 회)."""
    out = ct
    step = 1
    while step < M:
        out = engine.add(out, _rot(engine, out, kp, -step))
        step *= 2
    return out


# 라운드 시작 시 잔차가 가져야 할 최소 레벨.
#   학습 1 라운드가 소모하는 양(prefix 3 + 통계 3 + 정규화 + 리프 Goldschmidt)
#   을 감당하려면 refresh 회복(10) 근처가 필요하다.
BOOST_MIN_LEVEL = 10


def make_leaf_onehots(engine, td, dp, depth):
    """리프 ℓ 의 모델 슬롯(ℓ·dp)만 1 인 평문 마스크 L 개."""
    M = td["M"]
    out = []
    for l in range(1 << depth):
        v = [0.0] * M
        v[l * dp] = 1.0
        out.append(engine.encode(v))
    return out


def scatter_leaves(engine, Leaf, masks, leaf_oh, kp, td):
    """
    리프값을 샘플 공간으로 흩뿌린다.   pred = Σ_ℓ Leaf_ℓ ⊙ masks[ℓ]

      Leaf     모델 공간 (슬롯 ℓ·dp)
      masks    레벨 D 파티션 마스크 (2^D 개, 각각 청크 리스트)
      leaf_oh  make_leaf_onehots 결과

    반환: 샘플 공간 청크 리스트 (gvec 과 같은 모양)
    """
    from Hard_Tree.desilo.util.keypack import refresh as _rf
    import gc
    rlk = kp.relinearization_key
    # ★ Leaf 는 Goldschmidt 를 거쳐 레벨이 낮다.  masks 도 레벨 D 만큼 내려와 있다.
    if 0 <= _lvl(Leaf) < 2:
        Leaf = _rf(engine, Leaf, kp)
    pred = None
    for l, mp in enumerate(masks):
        v = engine.multiply(Leaf, leaf_oh[l])          # ℓ 슬롯만 남김
        v = _full_sum(engine, v, kp, td["M"])          # 전 슬롯 broadcast
        term = []
        for m in mp:
            mm = _rf(engine, m, kp) if 0 <= _lvl(m) < 2 else m
            term.append(engine.multiply(v, mm, rlk))
            del mm
        pred = term if pred is None else [engine.add(a, b) for a, b in zip(pred, term)]
        del v, term
        if (l & 3) == 3:
            gc.collect()                               # ★ OOM 완화
    gc.collect()
    return pred


def free_round(*objs):
    """부스팅 라운드 끝에서 중간 암호문을 끊는다.  GPU 메모리 회수용."""
    import gc
    for o in objs:
        del o
    gc.collect()


def update_residual(engine, gvec, pred, kp, td, lr=1.0, min_level=None):
    """
    g ← g − lr·pred.   lr 은 공개 상수라 평문 곱 1 회.

    ★ 라운드 경계에서 refresh 한다.  잔차는 라운드마다 학습 파이프라인을
      통째로 통과하므로(리프값 Goldschmidt + scatter 의 full_sum·곱),
      갱신된 g 를 그대로 다음 라운드에 넣으면 몇 라운드 만에 레벨이 0 이 된다.
      잔차는 **임의 실수**라 sign_bootstrap 은 못 쓰고 표준 bootstrap 을 쓴다.
    """
    from Hard_Tree.desilo.util.keypack import refresh as _rf
    lr_pt = engine.encode([float(lr)] * td["M"])
    # pred 는 scatter 를 거쳐 레벨이 낮다 — 평문곱 1 회분이 남아야 한다
    pred = [(_rf(engine, p, kp) if 0 <= _lvl(p) < 1 else p) for p in pred]
    out = [engine.subtract(g, engine.multiply(p, lr_pt))
           for g, p in zip(gvec, pred)]
    need = min_level if min_level is not None else BOOST_MIN_LEVEL
    if 0 <= _lvl(out[0]) < need:
        out = [_rf(engine, x, kp) for x in out]
    return out


def encode_boost_sel_ct(engine, gvec, kp, td):
    """
    **암호문 잔차**로 선택 벡터 쌍을 만든다 (부스팅 2 라운드부터 필요).

      [0] g        (암호문 그대로)
      [1] 1/n      (공개 평문)

    ★ 평문판 encode_boost_sel 은 g/‖g‖ 로 단위노름을 맞추지만, 암호문에서는
      ‖g‖ 를 모른다.  대신 **라운드마다 ‖g‖ 가 줄어들 뿐 커지지 않으므로**
      1 라운드의 ‖g₀‖ 로 정규화해 둔 스케일이 그대로 상한으로 유효하다.
      (Cauchy-Schwarz 상한은 ‖g‖ 에 단조 → 뒤 라운드는 더 여유로워진다)
    ★ 그 결과 점수가 라운드마다 작아져 dead zone 이 늘 수 있다.
      pick_alpha 가 흡수하지만, 라운드가 아주 많으면 재정규화가 필요하다.
    """
    n = td["n"]
    ones = encode_sample_all(engine, kp, np.full(n, 1.0 / n), td, encrypted=False)
    return [gvec, ones]


def decode_sample_all(engine, vec, sk, td, n):
    """
    샘플 축 암호문(청크 리스트) → 길이 n 의 numpy 배열.
    ★ encode_sample_vec 의 역: 청크 ch 는 샘플 [ch·S, (ch+1)·S) 를
      블록 0 의 슬롯 [0, S) 에 담는다 (그리고 블록 전체에 복제).
    """
    S = td["S"]
    out = np.zeros(n)
    for ch in range(td["n_chunk"]):
        dec = np.real(engine.decrypt(vec[ch], sk))
        lo = ch * S
        for i in range(min(S, n - lo)):
            out[lo + i] = dec[i]
    return out
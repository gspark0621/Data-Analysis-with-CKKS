# soft_tree/desilo/packed_core.py   (v3 — 레벨 소비 축소 + 태그별 refresh 계측)
"""
Block-packing (SIMD) 소프트트리 — feature 축 + leaf 축 이중 packing.

 [인코딩] 샘플 n → 슬롯 블록 [n*B : n*B+B],  B = 2^ceil(log2 max(F, 2^depth))
   X_all : 1 ct, slot[nB+f]=X[n][f]      W[i] : 1 ct canonical [0..F)
   b[i]  : broadcast 스칼라               v    : 1 ct canonical [0..2^depth)
   prob  : 1 ct, slot[nB+l]=P(leaf l)

 ═══════════ v3 개정: 레벨 소비 17 → 10 ═══════════
  bootstrap 회복 레벨이 10 이므로, 스텝당 소비를 10 이하로 낮추면
  체인당 refresh 가 2회 → 1회가 되어 약 2배 빨라진다.  정확도 손실은 없다.

  ① combo_pt = smask · zscale · (1/B) 를 평문 하나로 접기            (-2)
       현재:  z*smask → +b*smask → *zscale → (gate 안에서) *1/B   = 평문곱 4회
       변경:  u = z*combo → +b*combo                              = 평문곱 2회
       ★ backward 불변: 양쪽 모두 캐시되는 값이 u = (Wx+b)*zscale/B 로 동일.
  ② 게이트 마스킹을 평문 뺄셈으로                                    (-1)
       combo 에 smask 가 있어 비샘플 슬롯의 u=0 → g(0)=0.5 (상수).
       곱셈 대신 상수 평문을 빼면 레벨 소모 0.
  ③ dsig 마스킹을 평문 뺄셈으로                                      (-1)
       같은 이유. dg/dz(0) = 0.75/B (상수).
  ④ grad_g 마스킹 제거                                              (-1)
       grad_z = grad_g * dsig 이고 dsig 가 이미 마스킹되어 있으므로 중복.
  ⑤ grad_z * zscale 제거 → lr 에 흡수                               (-1)
       vel = mu*vel - (lr*zscale)*grad 로 등가.
  ⑥ grad_W * lr 을 colmask 에 접기                                  (-1)
       colmask_scaled = colmask * lr * zscale.  이후 vel = mu*vel - grad_W.

 ═══════════ 계측 ═══════════
  OPCOUNT        : 연산 횟수 (HBDT Table 4 / Desilo 실측 단가로 환산)
  REFRESH_BY_TAG : 어느 암호문이 refresh 를 잡아먹는지 태그별 집계 + 평균 입력 레벨
                   (bootstrap 비용은 입력 레벨과 무관하게 일정하므로,
                    입력 레벨이 높은 refresh 는 낭비 = 임계값을 낮춰야 한다는 신호)

 [Desilo 실측 단가] refresh 4346.8 / mult_ct 16.03 / mult_pt 3.52 / rot 12.27 / add 0.673 ms
 [bootstrap] small key + 일반 bootstrap 확정.  회복 레벨 10, 비용 ~4.35 s (입력 레벨 무관).
             OOM 발생 시 lossy_bootstrap(stage=3, 회복 13)으로 전환 검토.
"""
from __future__ import annotations
import gc
import numpy as np
try:
    from desilofhe import Engine, Ciphertext
except Exception:
    Engine = Ciphertext = object
from Soft_Tree.desilo.util.keypack import KeyPack, refresh
from Soft_Tree.desilo.gate_eval import (gate_forward, gate_derivative_poly,
                                        fit_gate_composite, feature_mask,
                                        gate_offset_values)

_MEM_FREE = True


# ═══════════════════════════════════════════════════════════════════════
# 계측
# ═══════════════════════════════════════════════════════════════════════
OPCOUNT = {"refresh": 0, "mult_ct": 0, "mult_pt": 0, "rot": 0, "add": 0}
REFRESH_BY_TAG: dict[str, list[int]] = {}

HBDT_UNIT_MS   = {"refresh": 149.0,    "mult_ct": 0.72,  "mult_pt": 0.26,
                  "rot": 0.60,  "add": 0.017}
DESILO_UNIT_MS = {"refresh": 4346.808, "mult_ct": 16.032, "mult_pt": 3.520,
                  "rot": 12.268, "add": 0.673}


def opcount_reset():
    for k in OPCOUNT:
        OPCOUNT[k] = 0
    REFRESH_BY_TAG.clear()


def opcount_report(tag: str = "", unit=None, unit_name="Desilo 실측"):
    U = unit or DESILO_UNIT_MS
    tot = sum(OPCOUNT[k] * U.get(k, 0.0) for k in OPCOUNT)
    print(f"\n[OPCOUNT] {tag}   (단가: {unit_name})")
    for k in OPCOUNT:
        print(f"    {k:<9s} {OPCOUNT[k]:>8d}  x {U.get(k,0.0):>8.3f} ms = "
              f"{OPCOUNT[k]*U.get(k,0.0):>11.1f} ms  ({OPCOUNT[k]*U.get(k,0.0)/max(tot,1e-9):>5.1%})")
    print(f"    {'합계':<9s} {'':>8s}    {'':>8s}   {tot:>11.1f} ms  ({tot/1000.0:.3f} s)")
    if REFRESH_BY_TAG:
        print(f"\n  [refresh 태그별]  (입력 레벨이 높을수록 낭비 — bootstrap 비용은 레벨 무관 고정)")
        print(f"    {'태그':<12}{'횟수':>6}{'평균 입력lv':>12}{'회복량(=10-lv)':>16}{'비중':>8}")
        n_all = sum(len(v) for v in REFRESH_BY_TAG.values())
        for t, lv in sorted(REFRESH_BY_TAG.items(), key=lambda x: -len(x[1])):
            m = float(np.mean(lv)) if lv else 0.0
            print(f"    {t:<12}{len(lv):>6}{m:>12.1f}{10-m:>16.1f}{len(lv)/max(n_all,1):>8.1%}")
    return tot


# ═══════════════════════════════════════════════════════════════════════
# 계측 래퍼
# ═══════════════════════════════════════════════════════════════════════
def _rot(engine, ct, keypack, delta):
    OPCOUNT["rot"] += 1
    return engine.rotate(ct, keypack.rotation_key, delta)


def _mul(engine, a, b, rlk=None):
    if rlk is not None:
        OPCOUNT["mult_ct"] += 1
        return engine.multiply(a, b, rlk)
    OPCOUNT["mult_pt"] += 1
    return engine.multiply(a, b)


def _add(engine, a, b):
    OPCOUNT["add"] += 1
    return engine.add(a, b)


def _sub(engine, a, b):
    OPCOUNT["add"] += 1
    return engine.subtract(a, b)


def _ensure(engine, ct, keypack, need, tag="", verbose=False):
    try:
        lv = ct.level
        if lv < need:
            OPCOUNT["refresh"] += 1
            base = tag.split("[")[0] if tag else "?"
            REFRESH_BY_TAG.setdefault(base, []).append(int(lv))
            if verbose:
                print(f"    [BOOT] {tag}: {lv}<{need} → refresh")
            return refresh(engine, ct, keypack)
    except Exception:
        pass
    return ct


def block_size(F, depth):
    return 1 << (max(1, max(F, 2 ** depth) - 1)).bit_length()


# ── 회전 집계 ──
def _block_sum(engine, ct, keypack, upto):
    out = ct; s = 1
    while s < upto:
        out = _add(engine, out, _rot(engine, out, keypack, -s)); s *= 2
    return out


def _block_bcast(engine, ct, keypack, upto):
    out = ct; s = 1
    while s < upto:
        out = _add(engine, out, _rot(engine, out, keypack, s)); s *= 2
    return out


def _block_replicate(engine, ct, keypack, B, n_blocks):
    out = ct; s = B; lim = B * (1 << ((max(1, n_blocks - 1)).bit_length()))
    while s < lim:
        out = _add(engine, out, _rot(engine, out, keypack, s)); s *= 2
    return out


def _strided_sum(engine, ct, keypack, B, n_blocks):
    out = ct; s = B; lim = B * (1 << ((max(1, n_blocks - 1)).bit_length()))
    while s < lim:
        out = _add(engine, out, _rot(engine, out, keypack, -s)); s *= 2
    return out


def _full_sum(engine, ct, keypack):
    out = ct; s = 1; sc = engine.slot_count
    while s < sc:
        out = _add(engine, out, _rot(engine, out, keypack, -s)); s *= 2
    return out


class _ConstCache:
    def __init__(self): self.d = {}
    def get(self, engine, s):
        k = float(s)
        if k not in self.d:
            self.d[k] = engine.encode([k] * engine.slot_count)
        return self.d[k]


_CC = _ConstCache()
def _c(engine, s): return _CC.get(engine, s)


# ═══════════════════════════════════════════════════════════════════════
# 평문 묶음 — 접힌 상수들을 한 곳에서 관리
# ═══════════════════════════════════════════════════════════════════════
class PTPack:
    """
    smask   : slot[nB]=1
    combo   : slot[nB] = zscale/gate_B          ★ smask·zscale·(1/B) 통합  (①)
    g_off   : slot[nB]=0, 나머지 = g(0)=0.5     ★ 게이트 마스킹용 뺄셈 상수 (②)
    d_off   : slot[nB]=0, 나머지 = 0.75/gate_B  ★ dsig  마스킹용 뺄셈 상수 (③)
    lmasks  : [0..depth] 활성 leaf, [depth+1] = vcmask(block0 의 [0..2^d))
    cmask   : [0..F)=1  (tree.masks 가 없을 때 fallback)
    """
    def __init__(self, engine, N, F, B, depth, zscale, gate_B):
        sc = engine.slot_count
        self.N, self.F, self.B, self.depth = N, F, B, depth
        self.zscale, self.gate_B = float(zscale), float(gate_B)

        sm = [0.0] * sc
        cm = [0.0] * sc
        cb = [0.0] * sc
        g0, d0 = gate_offset_values(gate_B)          # (0.5, 0.75/B)
        go = [g0] * sc
        do = [d0] * sc
        for f in range(F):
            cm[f] = 1.0
        for n in range(N):
            sm[n * B] = 1.0
            cb[n * B] = self.zscale / self.gate_B
            go[n * B] = 0.0
            do[n * B] = 0.0
        self.smask = engine.encode(sm)
        self.cmask = engine.encode(cm)
        self.combo = engine.encode(cb)
        self.g_off = engine.encode(go)
        self.d_off = engine.encode(do)

        self.lr_v = 1.0          # ⑦ 에서 vcmask 에 접을 lr (set_lr 로 설정)
        self.lmasks = []
        for i in range(depth + 1):
            lm = [0.0] * sc; act = 2 ** i
            for n in range(N):
                for l in range(min(act, B)):
                    lm[n * B + l] = 1.0
            self.lmasks.append(engine.encode(lm))
        vc = [0.0] * sc
        for l in range(2 ** depth):
            vc[l] = 1.0
        self.lmasks.append(engine.encode(vc))
        self._vc_raw = vc
        self._engine = engine

    def set_lr_v(self, lr_v):
        """⑦ vcmask 에 lr 을 접는다 → grad_v 의 lr 곱셈(1레벨) 제거.
           ★ grad_v 는 zscale 을 거치지 않으므로 lr_zs 가 아니라 lr 을 써야 한다."""
        self.lr_v = float(lr_v)
        v = [x * float(lr_v) for x in self._vc_raw]
        self.lmasks[-1] = self._engine.encode(v)

    def scaled_colmask(self, engine, mask_vec, lr_zs):
        """⑥ colmask · lr · zscale 을 하나로 (grad_W 에 곱하면 momentum 에서 lr 곱이 사라짐)."""
        sc = engine.slot_count
        v = [0.0] * sc
        for f in range(self.F):
            v[f] = float(mask_vec[f]) * float(lr_zs)
        return engine.encode(v)


class EncSoftTreePacked:
    def __init__(self, W, b, v, cheb_coeffs, dcheb_coeffs, band, zscale,
                 F, N, B, depth, masks=None, gmasks=None):
        self.W = W; self.b = b; self.v = v
        self.cheb_coeffs = cheb_coeffs; self.dcheb_coeffs = dcheb_coeffs
        self.band = band; self.zscale = zscale
        self.F = F; self.N = N; self.B = B; self.depth = depth
        self.masks = masks     # forward/backward 의 순수 0/1 마스크 (평문)
        self.gmasks = gmasks   # ⑥ colmask·lr·zscale 접은 평문 (grad_W 용)


# ── forward ──
def forward_packed(engine, tree, X_all, pt: PTPack, keypack,
                   return_cache=False, refresh_below=None, verbose=False,
                   n_blocks=None):
    rlk = keypack.relinearization_key
    d, B = tree.depth, tree.B
    N = tree.N if n_blocks is None else int(n_blocks)

    G = []; Zc = []
    for i in range(d):
        Wblk = _block_replicate(engine, tree.W[i], keypack, B, N)
        prod = _mul(engine, Wblk, X_all, rlk)
        z = _block_sum(engine, prod, keypack, B)
        if _MEM_FREE: del prod, Wblk
        # ★① combo = smask · zscale · (1/B)  → 평문곱 4회를 2회로
        u = _mul(engine, z, pt.combo)
        u = _add(engine, u, _mul(engine, tree.b[i], pt.combo))
        if _MEM_FREE: del z
        # 게이트: u 는 이미 z/B 이므로 gate 내부의 1/B 곱은 하지 않는다
        g_i, u_nonNTT = gate_forward(engine, u, tree.cheb_coeffs, tree.band,
                                     keypack, return_zscaled=True, prescaled=True)
        # ★② 비샘플 슬롯의 g(0)=0.5 를 평문 뺄셈으로 제거 (레벨 0)
        g_i = _sub(engine, g_i, pt.g_off)
        if refresh_below is not None:
            g_i = _ensure(engine, g_i, keypack, refresh_below, tag=f"g[{i}]", verbose=verbose)
        G.append(g_i); Zc.append(u_nonNTT)

    prob = _mul(engine, engine.encrypt([1.0] * engine.slot_count, keypack.public_key), pt.smask)
    pbs = []
    for i in range(d):
        pbs.append(prob)
        g_blk = _block_bcast(engine, G[i], keypack, B)
        one_m = _sub(engine, _c(engine, 1.0), g_blk)
        left = _mul(engine, prob, one_m, rlk)
        right = _rot(engine, _mul(engine, prob, g_blk, rlk), keypack, 2 ** i)
        prob = _add(engine, left, right)
        if refresh_below is not None:
            prob = _ensure(engine, prob, keypack, refresh_below, tag=f"prob[{i}]", verbose=verbose)
        if _MEM_FREE: del g_blk, one_m, left, right

    v_blk = _block_replicate(engine, tree.v, keypack, B, N)
    f = _block_sum(engine, _mul(engine, prob, v_blk, rlk), keypack, B)
    if _MEM_FREE: del v_blk

    if return_cache:
        return f, {"G": G, "Zc": Zc, "prob": prob, "pbs": pbs}
    if _MEM_FREE:
        del G, Zc, prob, pbs; gc.collect()
    return f


# ── backward ──
def backward_packed(engine, tree, X_all, pt: PTPack, cache, dL_df, keypack,
                    refresh_below=None, verbose=False, n_blocks=None):
    """반환 grad_W 는 ⑥에 의해 이미 (lr·zscale) 이 곱해진 상태.
       grad_b / grad_v 는 스케일 없음 → momentum 에서 lr_zs 를 곱한다."""
    rlk = keypack.relinearization_key
    d, B = tree.depth, tree.B
    N = tree.N if n_blocks is None else int(n_blocks)
    G, Zc, prob, pbs = cache["G"], cache["Zc"], cache["prob"], cache["pbs"]

    if refresh_below is not None:
        dL_df = _ensure(engine, dL_df, keypack, refresh_below, tag="dL_df", verbose=verbose)

    dL_blk = _block_bcast(engine, dL_df, keypack, B)
    grad_v = _strided_sum(engine, _mul(engine, dL_blk, prob, rlk), keypack, B, N)
    grad_v = _mul(engine, grad_v, pt.lmasks[d + 1])
    if refresh_below is not None:
        grad_v = _ensure(engine, grad_v, keypack, refresh_below, tag="gv", verbose=verbose)

    v_blk = _block_replicate(engine, tree.v, keypack, B, N)
    grad_prob = _mul(engine, dL_blk, v_blk, rlk)
    if _MEM_FREE: del v_blk, dL_blk

    grad_W = [None] * d; grad_b = [None] * d
    for i in reversed(range(d)):
        num = 2 ** i
        g_blk = _block_bcast(engine, G[i], keypack, B)
        if refresh_below is not None:
            grad_prob = _ensure(engine, grad_prob, keypack, refresh_below,
                                tag=f"gp[{i}]", verbose=verbose)
        right_al = _rot(engine, grad_prob, keypack, -num)
        diff = _mul(engine, _sub(engine, right_al, grad_prob), pt.lmasks[i])
        if _MEM_FREE: del right_al
        grad_g = _block_sum(engine, _mul(engine, diff, pbs[i], rlk), keypack, B)
        # ★④ grad_g 마스킹 제거 — 아래 dsig 가 마스킹되어 있어 곱하면 자동으로 0
        if _MEM_FREE: del diff
        if refresh_below is not None:
            grad_g = _ensure(engine, grad_g, keypack, refresh_below, tag=f"gg[{i}]", verbose=verbose)
        dsig = gate_derivative_poly(engine, Zc[i], tree.dcheb_coeffs, tree.band, keypack)
        # ★③ 비샘플 슬롯의 dg/dz(0)=0.75/B 를 평문 뺄셈으로 제거 (레벨 0)
        dsig = _sub(engine, dsig, pt.d_off)
        grad_z = _mul(engine, grad_g, dsig, rlk)
        if _MEM_FREE: del grad_g, dsig
        if refresh_below is not None:
            grad_z = _ensure(engine, grad_z, keypack, refresh_below, tag=f"gz[{i}]", verbose=verbose)
        # ★⑤ grad_z * zscale 제거 — lr 에 흡수됨

        gzb = _block_bcast(engine, grad_z, keypack, B)
        gWi = _strided_sum(engine, _mul(engine, gzb, X_all, rlk), keypack, B, N)
        # ★⑥ colmask · lr · zscale 을 한 번에 (momentum 에서 lr 곱이 사라짐)
        gmask = (tree.gmasks[i] if getattr(tree, "gmasks", None) is not None
                 else (tree.masks[i] if getattr(tree, "masks", None) is not None else pt.cmask))
        grad_W[i] = _mul(engine, gWi, gmask)
        grad_b[i] = _full_sum(engine, grad_z, keypack)
        if _MEM_FREE: del gzb, gWi, grad_z
        if refresh_below is not None:
            grad_W[i] = _ensure(engine, grad_W[i], keypack, refresh_below, tag=f"gW[{i}]", verbose=verbose)
            grad_b[i] = _ensure(engine, grad_b[i], keypack, refresh_below, tag=f"gb[{i}]", verbose=verbose)

        new_gp = _add(engine,
                      _mul(engine, grad_prob, _sub(engine, _c(engine, 1.0), g_blk), rlk),
                      _mul(engine, _rot(engine, grad_prob, keypack, -num), g_blk, rlk))
        grad_prob = _mul(engine, new_gp, pt.lmasks[i])
        if _MEM_FREE: del g_blk, new_gp
        if _MEM_FREE: gc.collect()
    if _MEM_FREE: del grad_prob
    return grad_W, grad_b, grad_v


# ── train_step ──
def train_step_packed(engine, tree, X_all, y_ct, pt: PTPack, keypack, lr_zs, mu=0.9,
                      vel_W=None, vel_b=None, vel_v=None, sample_weight_pt=None,
                      refresh_below=8, mid_refresh_below=6, grad_refresh_below=None,
                      verbose_level=False, level_trace=False):
    """
    lr_zs : lr_effective * zscale  (⑤⑥ 때문에 zscale 이 여기로 흡수됨)
    grad_refresh_below : gradient 전용 refresh 임계값.  None 이면 mid_refresh_below 사용.
        gradient 레벨이 낮으면 momentum 사슬(vel → param)이 통째로 끌려 내려가
        다음 스텝에서 param/vel 을 전부 refresh 해야 한다.  이 값을 높이면
        gradient refresh(5개)로 param/vel refresh(10개)를 대체할 수 있다.
    """
    d = tree.depth
    for i in range(d):
        tree.W[i] = _ensure(engine, tree.W[i], keypack, refresh_below, tag=f"W[{i}]", verbose=verbose_level)
        tree.b[i] = _ensure(engine, tree.b[i], keypack, refresh_below, tag=f"b[{i}]", verbose=verbose_level)
    tree.v = _ensure(engine, tree.v, keypack, refresh_below, tag="v", verbose=verbose_level)
    if vel_W is not None:
        for i in range(d):
            vel_W[i] = _ensure(engine, vel_W[i], keypack, refresh_below, tag=f"vW[{i}]", verbose=verbose_level)
            vel_b[i] = _ensure(engine, vel_b[i], keypack, refresh_below, tag=f"vb[{i}]", verbose=verbose_level)
        vel_v = _ensure(engine, vel_v, keypack, refresh_below, tag="vv", verbose=verbose_level)

    lvl = (lambda t, c: print(f"    [LV] {t:<10} {getattr(c,'level','?')}")) if level_trace else (lambda t, c: None)
    lvl("W[0] 시작", tree.W[0])

    if vel_W is None:
        z0 = lambda: engine.encrypt([0.0] * engine.slot_count, keypack.public_key)
        vel_W = [z0() for _ in range(d)]; vel_b = [z0() for _ in range(d)]; vel_v = z0()

    f, cache = forward_packed(engine, tree, X_all, pt, keypack,
                              return_cache=True, refresh_below=mid_refresh_below,
                              verbose=verbose_level)
    lvl("f", f)
    dL_df = _sub(engine, f, y_ct)
    dL_df = _mul(engine, dL_df, sample_weight_pt if sample_weight_pt is not None else pt.smask)
    grad_W, grad_b, grad_v = backward_packed(engine, tree, X_all, pt, cache, dL_df, keypack,
                                             refresh_below=mid_refresh_below, verbose=verbose_level)
    lvl("grad_W[0]", grad_W[0])

    # ★ gradient 전용 임계값 — momentum 사슬 보호
    if grad_refresh_below is not None:
        for i in range(d):
            grad_W[i] = _ensure(engine, grad_W[i], keypack, grad_refresh_below, tag=f"GW[{i}]", verbose=verbose_level)
            grad_b[i] = _ensure(engine, grad_b[i], keypack, grad_refresh_below, tag=f"GB[{i}]", verbose=verbose_level)
        grad_v = _ensure(engine, grad_v, keypack, grad_refresh_below, tag="GV", verbose=verbose_level)

    mu_pt = _c(engine, mu); lr_pt = _c(engine, lr_zs)
    for i in range(d):
        # grad_W 는 ⑥에서 이미 lr·zscale 이 곱해져 있음
        vel_W[i] = _sub(engine, _mul(engine, vel_W[i], mu_pt), grad_W[i])
        tree.W[i] = _add(engine, tree.W[i], vel_W[i])
        vel_b[i] = _sub(engine, _mul(engine, vel_b[i], mu_pt), _mul(engine, grad_b[i], lr_pt))
        tree.b[i] = _add(engine, tree.b[i], vel_b[i])
        if _MEM_FREE: grad_W[i] = None; grad_b[i] = None
    # ★⑦ grad_v 는 vcmask 에 lr 이 접혀 있음 (zscale 은 관여하지 않음 — 등가성 검증 완료)
    vel_v = _sub(engine, _mul(engine, vel_v, mu_pt), grad_v)
    tree.v = _add(engine, tree.v, vel_v)
    lvl("W[0] 끝", tree.W[0])
    if _MEM_FREE:
        del grad_W, grad_b, grad_v, cache, dL_df, f; gc.collect()
    return tree, vel_W, vel_b, vel_v


# ── 인코딩 헬퍼 ──
def encode_X_block(engine, keypack, X, B):
    sc = engine.slot_count; N, F = X.shape
    xf = [0.0] * sc
    for n in range(N):
        for f in range(F): xf[n * B + f] = float(X[n, f])
    return engine.encrypt(xf, keypack.public_key)


def encode_y_block(engine, keypack, y, B):
    sc = engine.slot_count; ys = [0.0] * sc
    for n in range(len(y)): ys[n * B] = float(y[n])
    return engine.encrypt(ys, keypack.public_key)


def init_packed_tree(engine, keypack, F, N, B, depth, cheb, dcheb, band, zscale,
                     seed=0, mtry=None, pt: PTPack = None, lr_zs=1.0):
    """난수 소비 순서를 평문 참조와 일치(벡터 호출). W 는 초기값부터 마스킹."""
    sc = engine.slot_count; rng = np.random.default_rng(seed); L = 2 ** depth
    Wct = []; masks = []; gmasks = []
    W_plain = [rng.normal(0, 0.3, F) for _ in range(depth)]
    for i in range(depth):
        m = feature_mask(F, seed, i, mtry)
        w = W_plain[i] * m
        wf = [0.0] * sc
        for f in range(F): wf[f] = float(w[f])
        Wct.append(engine.encrypt(wf, keypack.public_key))
        mv = [0.0] * sc
        for f in range(F): mv[f] = float(m[f])
        masks.append(engine.encode(mv))
        gmasks.append(pt.scaled_colmask(engine, m, lr_zs) if pt is not None else engine.encode(mv))
    bct = [engine.encrypt([0.0] * sc, keypack.public_key) for _ in range(depth)]
    v_plain = rng.normal(0, 0.5, L)
    vf = [0.0] * sc
    for l in range(L): vf[l] = float(v_plain[l])
    vct = engine.encrypt(vf, keypack.public_key)
    return EncSoftTreePacked(Wct, bct, vct, cheb, dcheb, band, zscale,
                             F=F, N=N, B=B, depth=depth, masks=masks, gmasks=gmasks)


# ── RF 학습/예측 ──
def train_forest_packed(engine, keypack, secret_key, X_all, y_ct, pt: PTPack,
                        F, N, B, depth=4, K=20, epochs=100, lr=0.3, mu=0.9,
                        zscale=1.5, gate_B=8.0, mtry=None,
                        refresh_below=8, mid_refresh_below=6, grad_refresh_below=None,
                        count_ops=False):
    from time import time
    cheb, dcheb, band = fit_gate_composite(B=gate_B, n=1)
    lr_eff = lr / float(N)
    lr_zs = lr_eff * float(zscale)                   # ★⑤ W/b 경로: zscale 흡수
    pt.set_lr_v(lr_eff)                              # ★⑦ v 경로: zscale 무관, lr 만
    print(f"[Server-packed] K={K}, depth={depth}, epochs={epochs}, F={F}, B={B}, mtry={mtry}")
    print(f"                zscale={zscale}, gate_B={gate_B}, beta={3*zscale/gate_B:.3f}, "
          f"lr_eff={lr_eff:.3e}, lr_zs={lr_zs:.3e}")
    print(f"                refresh_below={refresh_below}, mid={mid_refresh_below}, "
          f"grad={grad_refresh_below}")
    trees = []
    for k in range(K):
        t0 = time()
        tree = init_packed_tree(engine, keypack, F, N, B, depth, cheb, dcheb, band,
                                zscale, seed=k, mtry=mtry, pt=pt, lr_zs=lr_zs)
        sw = [0.0] * engine.slot_count
        if K > 1:
            rng = np.random.default_rng(10_000 + k)
            cnt = np.bincount(rng.integers(0, N, N), minlength=N)
            for n in range(N): sw[n * B] = float(cnt[n])
        else:
            for n in range(N): sw[n * B] = 1.0
        sw_pt = engine.encode(sw)
        vW = vb = vv = None
        for ep in range(epochs):
            first = (k == 0 and ep == 0)
            if count_ops and first:
                opcount_reset()
            tree, vW, vb, vv = train_step_packed(
                engine, tree, X_all, y_ct, pt, keypack, lr_zs=lr_zs, mu=mu,
                vel_W=vW, vel_b=vb, vel_v=vv, sample_weight_pt=sw_pt,
                refresh_below=refresh_below, mid_refresh_below=mid_refresh_below,
                grad_refresh_below=grad_refresh_below,
                verbose_level=(k == 0 and ep < 2), level_trace=first)
            if count_ops and first:
                ms = opcount_report(f"학습 1 step (depth={depth})")
                print(f"    → 추정: {ms/1000*epochs:.1f} s/tree, {ms/1000*epochs*K:.1f} s/forest")
                opcount_report(f"학습 1 step (depth={depth})", HBDT_UNIT_MS, "HBDT Table 4")
        trees.append(tree)
        del vW, vb, vv; gc.collect()
        print(f"[Server-packed] tree {k+1}/{K} 완료 ({time()-t0:.1f}s)")
    return trees


def predict_forest_packed(engine, trees, X_all, pt_te: PTPack, keypack,
                          mid_refresh_below=6, n_te=None, pred_level=24):
    def _one(tree):
        W = [_ensure(engine, tree.W[i], keypack, pred_level, tag=f"pW[{i}]") for i in range(tree.depth)]
        b = [_ensure(engine, tree.b[i], keypack, pred_level, tag=f"pb[{i}]") for i in range(tree.depth)]
        v = _ensure(engine, tree.v, keypack, pred_level, tag="pv")
        tmp = EncSoftTreePacked(W, b, v, tree.cheb_coeffs, tree.dcheb_coeffs,
                                tree.band, tree.zscale, tree.F, tree.N, tree.B,
                                tree.depth, masks=tree.masks, gmasks=tree.gmasks)
        return forward_packed(engine, tmp, X_all, pt_te, keypack,
                              refresh_below=mid_refresh_below, n_blocks=n_te)
    f = _one(trees[0])
    for t in trees[1:]:
        f = _add(engine, f, _one(t))
    invK = engine.encode([1.0 / len(trees)] * engine.slot_count)
    return _mul(engine, f, invK)
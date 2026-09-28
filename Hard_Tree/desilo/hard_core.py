# soft_tree/desilo/hard_core.py
"""
비교 기반 hard decision tree — CKKS 코어.

 [모델 표현]  노드 i 마다 (onehot_d(j_i), θ_i).  HBDT 의 onehot ⊗ prefix (n_max·d 슬롯)
              대신 dp + 1 슬롯만 사용 → 깊이 벽이 D≤5 에서 D≤10 으로.

   Model  ct   slot[i·dp + f] = 1 iff f == j_i        (dp = 2^⌈log₂ d⌉)
   Theta  ct   slot[i·dp]     = θ_i = (k_i + 1)/n_max
   Leaf   ct   slot[ℓ·dp]     = 리프 ℓ 라벨

 [마진 보장]  클라이언트가 bin b 를 x = (b+0.5)/n_max 로 매핑하고
              θ 를 bin 경계에 두면  |x − θ| ≥ 1/(2·n_max).
              → sign 근사의 ε 가 크게 잡혀 낮은 차수로 충분.

 [회전 규약]  engine.rotate(ct, key, s) = 오른쪽 = np.roll(+s).
              slot i+s 를 i 로 가져오려면 rotate(-s).  packed_core 와 동일.

 [검증]  layout_verify.py 에서 200/200 (d=3..30, D=4..10).
         회전만 사용, sign 은 트리 전체에 1회, 깊이 = dep(sign) + 4.

 ★ S1 단계: SIGN_IMPL="plain" 으로 두면 sign 만 평문 대체.
   회전 규약이 실제 FHE 에서 맞는지 먼저 확인한 뒤 minimax 로 교체한다.
"""
from __future__ import annotations
import gc
import numpy as np
try:
    from desilofhe import Engine, Ciphertext
except Exception:
    Engine = Ciphertext = object
from Hard_Tree.desilo.util.keypack import KeyPack, refresh

_MEM_FREE = True
# ══════════════════════════════════════════════════════════════════
# 회전 헬퍼 (packed_core 규약 그대로)
# ══════════════════════════════════════════════════════════════════
# 임의 shift 회전은 2^k 대비 5.4배 비싸다 (실측 33.9 vs 6.2 ms).
#   ★ 회전은 덧셈적으로 합성된다:  rot(rot(v,a),b) = rot(v,a+b)
#     따라서 임의 shift 를 2^k 회전 여러 번으로 대체할 수 있다.
#     (전체 합의 log₂N 회전과는 다른 얘기 — 여기서는 더하지 않고 연달아 건다.)
#
#   ★ 이진 전개가 아니라 **부호 있는 전개(NAF)** 를 쓴다.
#     4095 = 0b111111111111 → 이진 popcount 12  (분해 이득 없음)
#            4096 − 1        → NAF weight 2     (5.4배 → 2회로)
#     NAF 는 0 아닌 자릿수를 최소화한다 (평균 bits/3).
ROT_DECOMP_MAX = 5


def _naf(n):
    """비인접 형식(NAF).  n = Σ z·2^k,  z ∈ {+1,−1}.  0 아닌 항이 최소."""
    out = []
    k = 0
    while n:
        if n & 1:
            z = 2 - (n & 3)              # +1 또는 −1
            out.append((k, z))
            n -= z
        n >>= 1
        k += 1
    return out


def _rot(engine, ct, kp, delta, decomp=True):
    """
    회전.  임의 shift 는 2^k 회전의 연쇄로 분해한다 (NAF).
      분해 횟수 × 6.2ms  <  33.9ms  이면 이득 → ROT_DECOMP_MAX = 5
    """
    if delta == 0:
        return ct
    rk = kp.rotation_key
    if decomp and (abs(delta) & (abs(delta) - 1)):        # 2의 거듭제곱이 아니면
        terms = _naf(abs(delta))
        if len(terms) <= ROT_DECOMP_MAX:
            sign = 1 if delta > 0 else -1
            out = ct
            for k, z in terms:
                out = engine.rotate(out, rk, sign * z * (1 << k))
            return out
    return engine.rotate(ct, rk, delta)


def p2(x: int) -> int:
    """x 이상의 최소 2의 거듭제곱."""
    return 1 << (max(1, x - 1)).bit_length()


def dims(d: int, n_max: int, depth: int) -> dict:
    """
    dp   : 노드 블록 크기 (모델/추론 좌표계)
    n_pad: bin 축 패딩   (학습 전이 버퍼)
    blk  : 노드 블록     (학습 전이 버퍼)
    ★ 블록 합(rotate-and-add)은 블록 크기가 2의 거듭제곱일 때만 성립 → 전부 패딩.
    """
    dp = p2(d)
    n_pad = p2(n_max)
    return dict(dp=dp, n_pad=n_pad, blk=p2(d * n_pad),
                NI=(1 << depth) - 1, L=1 << depth, depth=depth, d=d, n_max=n_max)


def model_slots(d: int, depth: int) -> int:
    return (1 << depth) * p2(d)


def max_depth(d: int, slot_count: int = 32768) -> int:
    """2^D · dp ≤ M 을 만족하는 최대 D."""
    dp, D = p2(d), 0
    while (1 << (D + 1)) * dp <= slot_count:
        D += 1
    return D


# ══════════════════════════════════════════════════════════════════
# 회전 헬퍼 (packed_core 규약 그대로)
# ══════════════════════════════════════════════════════════════════
def _block_sum(engine, ct, kp, upto):
    """블록 내 [0..upto) 합 → 블록 첫 슬롯.  upto 는 2의 거듭제곱."""
    out, s = ct, 1
    while s < upto:
        out = engine.add(out, _rot(engine, out, kp, -s)); s *= 2
    return out


def _replicate(engine, ct, kp, stride, count):
    """블록0 → count 개 블록으로 복제.  count 는 2의 거듭제곱."""
    out, s = ct, stride
    lim = stride * count
    while s < lim:
        out = engine.add(out, _rot(engine, out, kp, s)); s *= 2
    return out


def _c(engine, s, cache=None):
    """상수 평문. cache 를 주면 재인코딩 회피 (packed_core 개선점 E)."""
    if cache is not None and s in cache:
        return cache[s]
    pt = engine.encode([float(s)] * engine.slot_count)
    if cache is not None:
        cache[s] = pt
    return pt


def _ensure(engine, ct, kp, need, tag=""):
    try:
        if ct.level < need:
            return refresh(engine, ct, kp)
    except Exception:
        pass
    return ct


# ══════════════════════════════════════════════════════════════════
# 평문 마스크 (클라이언트/오프라인 생성)
# ══════════════════════════════════════════════════════════════════
def make_infer_masks(engine, dm: dict, seq_bits: int = None) -> dict:
    """추론에 필요한 공개 평문 마스크 일괄 생성."""
    sc = engine.slot_count
    dp, D, NI, L = dm["dp"], dm["depth"], dm["NI"], dm["L"]
    SEQ = SPREAD_SEQ_BITS if seq_bits is None else seq_bits
    if SEQ is None:
        SEQ = 0

    def enc(v):
        return engine.encode(v)

    # 블록 첫 슬롯
    fm = [0.0] * sc
    for i in range(NI):
        fm[i * dp] = 1.0

    # 리프 슬롯
    lm = [0.0] * sc
    for l in range(L):
        lm[l * dp] = 1.0

    # 레벨별: 레벨 마스크 / stride 확장 대각 / β 패턴
    lvl, spread, beta, beta2 = [], [], [], []
    for t in range(D):
        n_now, width = 1 << t, 1 << (D - t)
        base = n_now - 1

        v = [0.0] * sc
        for q in range(n_now):
            v[(base + q) * dp] = 1.0
        lvl.append(enc(v))

        # stride 확장:  slot[q·dp] → slot[q·width·dp]
        #   q = q_hi·2^m + q_lo   (q_hi 는 상위 R 비트)
        #   ① 순차: q·S0 → q_hi·2^m·S + q_lo·S0   shift 가 2^k 배수, 깊이 R
        #   ② BSGS: q_lo 만 확장                   대각 2^m 개,      깊이 1
        #   ★ 순서가 중요.  BSGS 를 먼저 하면 중간 위치가 충돌한다.
        R = min(t, SEQ)
        m = t - R
        S0, S = dp, width * dp

        # ① 순차 doubling (상위 비트부터)
        seq = []
        cur = {q: q * S0 for q in range(n_now)}
        for k in range(R - 1, -1, -1):
            sel = [q for q in range(n_now) if ((q >> m) >> k) & 1]
            if not sel:
                continue
            mv = [0.0] * sc
            for q in sel:
                mv[cur[q]] = 1.0
            shift = (1 << k) * (1 << m) * (S - S0)
            seq.append((enc(mv), shift))
            for q in sel:
                cur[q] += shift

        # ② BSGS — q_lo 만 stride 확장.  전부 현재 위치에서 출발 → 깊이 1
        diag = []
        for ql in range(1 << m):
            mv = [0.0] * sc
            for qh in range(1 << R):
                q = qh * (1 << m) + ql
                mv[cur[q] + ql * (S - S0)] = 1.0        # 목적지
            diag.append((enc(mv), ql * (S - S0)))
        spread.append((seq, diag))

        b = [0.0] * sc
        b2 = [0.0] * sc
        for l in range(L):
            v = float((l >> (D - t - 1)) & 1)
            b[l * dp] = v
            b2[l * dp] = 2.0 * v          # ★ 2β 를 미리 평문으로 → 곱 1회·레벨 1 절감
        beta.append(enc(b))
        beta2.append(enc(b2))

    return dict(fmask=enc(fm), lmask=enc(lm), lvl=lvl, spread=spread,
                beta=beta, beta2=beta2)


# ══════════════════════════════════════════════════════════════════
# 인코딩 (클라이언트)
# ══════════════════════════════════════════════════════════════════
def encode_input(engine, kp, x_bins: np.ndarray, n_max: int, dp: int):
    """
    bin 인덱스 b → x = (b + 0.5)/n_max,  slot[f] = x_f.
    ★ 이 매핑과 θ = (k+1)/n_max 조합이 마진 1/(2·n_max) 를 만든다.
    """
    sc = engine.slot_count
    v = [0.0] * sc
    for f, b in enumerate(x_bins):
        v[f] = (float(b) + 0.5) / n_max
    return engine.encrypt(v, kp.public_key)


class EncHardTreePacked:
    """Model / Theta / Leaf / Prune 를 담는 암호화 모델."""

    def __init__(self, Model, Theta, Leaf, dm, Prune=None):
        self.Model, self.Theta, self.Leaf, self.Prune = Model, Theta, Leaf, Prune
        self.dm = dm


# ══════════════════════════════════════════════════════════════════
# sign — S1 에서는 평문, S2 에서 minimax composite 로 교체
# ══════════════════════════════════════════════════════════════════
SIGN_IMPL = "plain"          # "plain" | "minimax"

# spread 의 상위 몇 비트를 순차 doubling 으로 처리할지.
#   0 → 순수 BSGS   회전 2^t,  깊이 1   (shift 가 임의 → 실측 42 ms)
#   R → 하이브리드  회전 2^(t-R)+R, 깊이 R+1
#       순차 단계의 shift 는 2^k 배수라 싸다 (실측 7.6 ms, 4.8× 차이)
# None 이면 깊이 예산에서 자동 선택 (auto_spread_bits 참고).
SPREAD_SEQ_BITS = None      # None = auto

# ── 실측 상수 (RTX A6000 / DesiloFHE) ──
# ── 실측 (RTX A6000 / DesiloFHE, test_rot_cost.py) ──
L_FRESH = 26          # fresh 암호문 레벨
ROT_P2_MS = 3.49      # 2^k shift 회전      (level 13 기준)
ROT_ANY_MS = 20.25    # 임의 shift 회전     (level 13 기준, 5.8× 비쌈)
#   회전 비용은 레벨에 비례하지만 2^k/임의 비율은 레벨 무관하게 ~5.8×


def spread_rots(d, D, R):
    """추론 1회의 회전 수를 (2^k shift, 임의 shift) 로 분리해 반환."""
    dp = p2(d)
    p2rot = D.bit_length() + (dp - 1).bit_length() + D * (D + 1) // 2   # 복제·블록합·width
    anyrot = 0
    for t in range(D):
        r = min(t, R)
        p2rot += r                       # 순차 단계 = 2^k 배수 shift
        anyrot += 1 << (t - r)           # BSGS 대각 = 임의 shift
    return p2rot, anyrot


# ── sign_bootstrap (DesiloFHE 1.9) ──  ★ 실측 반영
#   부호 값(±1) 전용 lossy bootstrap 변형.  잡음까지 줄여준다.
#   회복 레벨 = 16 − stage_count.
#
#   실측 (A6000/Desilo):
#     stage  입력≥  회복   시간(ms)   최대오차
#       3      3     13     2280.3    7.69e-07
#       4      4     12     1461.4    8.08e-07
#       5      5     11     1256.5    7.85e-07
#     refresh              2814.4              ← 모든 stage 보다 비싸다
#
#   ★ 정밀도는 stage 와 무관.  레벨 1개당 ~800ms 인데 그 1레벨로 아끼는
#     회전은 60ms 수준 → 대개 큰 stage(=낮은 회복 레벨)가 유리하다.
#     단 입력 레벨이 stage_count 이상이어야 하므로 planner 가 자동 선택한다.
SIGNBOOT_TABLE = {3: 2280.3, 4: 1461.4, 5: 1256.5}
SIGNBOOT_STAGES = (5, 4, 3)   # 시도 순서 (싼 것부터)
SIGNBOOT_STAGE = 5            # 기본값
SIGNBOOT_MS = SIGNBOOT_TABLE[SIGNBOOT_STAGE]


def signboot_level(stage_count=SIGNBOOT_STAGE):
    """sign_bootstrap 이후 도달 레벨."""
    return 16 - stage_count


def signboot_ms(stage_count=SIGNBOOT_STAGE):
    return SIGNBOOT_TABLE.get(stage_count, 2280.3)


BOOT_MS = 2814.4      # ★ refresh 1회 실측.  HBDT(149ms)/HEaaN-ID3(130ms) 와
                      #   19× 차이 — 라이브러리/키 구성이 다르다.
                      #   ★ sign_bootstrap(1257~2280ms) 이 항상 더 싸므로,
                      #     ±1 상태에서 걸 수 있으면 그쪽을 쓴다.
L_SMALL = 10          # small-key bootstrap 회복 레벨

# 파이프라인 구간별 깊이
PRE_SIGN_DEPTH = 1    # Xr×Model(ct·ct) 만.  fmask 는 lvl[t] 가 대신한다
POST_FIXED = 3        # ×lvl[t] + ×beta2 + α   (실측 보정, spread 제외)
# spread 깊이 = R + 1


def plan_inference(d, D, sign_depth, model_level, l_small=L_SMALL, verbose=False):
    """
    추론 실행 계획 — 재암호화 없이 refresh 만으로.

      ① PRE (PRE_SIGN_DEPTH)   모델 레벨에서 시작
      ② sign                   minimax 합성.  성분 사이 표준 bootstrap,
                               **마지막에 sign_bootstrap 내장** → 출력 레벨 고정
      ③ POST_FIXED + spread(R+1)

    ★ 마지막 sign_bootstrap 이 내장되어 **sign 이후 레벨이 항상 같다**.
      따라서 계획은 (a) sign 앞에 refresh 를 넣을지, (b) R 을 얼마로 할지 둘뿐이다.
      (이전에는 boot_post 유무로 경우가 갈렸으나 이제 불필요)

    ★ R 은 클수록 좋다 (임의 shift 회전이 2^k 의 5.8×).

    반환: dict(R, boot_pre, boot_post, stage, ok, why)
          boot_post 는 호환용으로 항상 0 (sign 에 내장).
    """
    from Hard_Tree.desilo.minimax_sign import sign_out_level
    best = None
    for bp in (0, 1):                       # sign 직전 refresh (표준 bootstrap)
        lv_at_sign = (l_small if bp else model_level) - PRE_SIGN_DEPTH
        if lv_at_sign < sign_depth:
            continue                        # sign 자체가 안 들어감
        # sign 출력 레벨 — 마지막 sign_bootstrap 이 정한다 (to_01 포함)
        after = sign_out_level(to_01=True)
        R = min(D - 1, after - POST_FIXED - 1)
        if R < 0:
            continue
        a, b = spread_rots(d, D, R)
        t = a * ROT_P2_MS + b * ROT_ANY_MS + bp * BOOT_MS + signboot_ms(SIGNBOOT_STAGE)
        if best is None or t < best["est_ms"]:
            best = dict(R=R, boot_pre=bp, boot_post=0, stage=SIGNBOOT_STAGE,
                        est_ms=t, ok=True)

    if best is None:
        return dict(R=0, boot_pre=0, boot_post=0, stage=SIGNBOOT_STAGE, ok=False,
                    why=f"모델 레벨 {model_level} 로는 sign({sign_depth}) 이 안 들어감. "
                        f"refresh 회복 {l_small} 도 부족. 차수를 낮추세요.")
    if verbose:
        print(f"    plan: R={best['R']}, boot_pre={best['boot_pre']}, "
              f"모델 lv={model_level}, 추정 {best['est_ms']/1000:.2f}s")
    best["why"] = ""
    return best


def auto_spread_bits(d, D, sign_depth, l_fresh=L_FRESH, l_small=L_SMALL,
                     model_level=None, allow_boot=True, verbose=False):
    """
    호환용 래퍼.  model_level 을 주면 plan_inference 로 정확히 계산한다.
    안 주면 fresh 모델(L_FRESH)로 가정.
    반환: (R, n_boot)
    """
    pl = plan_inference(d, D, sign_depth,
                        l_fresh if model_level is None else model_level,
                        l_small, verbose=verbose)
    if not pl["ok"]:
        raise RuntimeError(pl["why"])
    return pl["R"], pl["boot_pre"] + pl["boot_post"]


def plan_kwargs(pl):
    """plan_inference 결과 → infer_packed 인자."""
    return dict(boot_pre=pl["boot_pre"], boot_post=pl["boot_post"],
                boot_stage=pl.get("stage", SIGNBOOT_STAGE))


def est_rot_ms(d, D, R):
    """참고용 회전 시간 추정 (선택에는 쓰이지 않음). 곱셈·sign 은 제외."""
    a, b = spread_rots(d, D, R)
    return a * ROT_P2_MS + b * ROT_ANY_MS



# ══════════════════════════════════════════════════════════════════
# sign — SIGN_IMPL 로 전환
# ══════════════════════════════════════════════════════════════════
def _sign01(engine, ct, kp, secret_key=None, alpha=6, epsilon=None, spec=None,
            to_01=True, final_boot=True, boot_stage=None):
    """
    (sign(x) + 1) / 2  →  x>0 이면 1, 아니면 0.

    plain   : 복호 → np.sign → 재암호.
              ★ 파이프라인 한가운데서 복호·재암호가 일어난다. 보안 없음,
                레벨도 리셋된다. 회전 규약 검증 전용.
    minimax : Lee, Lee, No, Kim 합성 다항식 (minimax_sign.py).
    """
    if SIGN_IMPL == "plain":
        assert secret_key is not None, "plain sign 은 검증용이라 secret_key 필요"
        v = np.sign(np.real(engine.decrypt(ct, secret_key)))
        return engine.encrypt(list(v if not to_01 else (v + 1.0) / 2.0), kp.public_key)

    from Hard_Tree.desilo.minimax_sign import minimax_sign01
    return minimax_sign01(engine, ct, kp, alpha=alpha, epsilon=epsilon, spec=spec,
                          to_01=to_01, final_boot=final_boot, boot_stage=boot_stage)

# ══════════════════════════════════════════════════════════════════
# 추론 — layout_verify.py 이식 (200/200 검증됨)
# ══════════════════════════════════════════════════════════════════
def infer_packed(engine, tree: EncHardTreePacked, X_ct, masks, kp,
                 secret_key=None, refresh_below=None, verbose=False,
                 sign_spec=None, n_boot=0, sign_depth_hint=None,
                 boot_pre=0, boot_post=None, boot_stage=SIGNBOOT_STAGE):
    """
    반환: acc.  slot[ℓ·dp] 가 0 인 리프가 선택된 경로.
          (I5 리프 선택은 HBDT 방식으로 별도 — select_leaf 참고)

    깊이: ct·ct 1 + pt 1 + sign + spread 1 + beta 1  =  dep(sign) + 4
    """
    dm = tree.dm
    dp, D, NI, L = dm["dp"], dm["depth"], dm["NI"], dm["L"]
    rlk = kp.relinearization_key
    cache = {}

    if boot_post is None:
        boot_post = n_boot                      # 구버전 호출 호환

    # ── I1  입력을 전 노드 블록에 복제 ──                      깊이 0
    Xr = _replicate(engine, X_ct, kp, dp, 1 << D)

    # ── I2  feature 선택:  ⟨onehot_j, x⟩ ──                    깊이 2
    prod = engine.multiply(Xr, tree.Model, rlk)                # ct·ct
    prod = _block_sum(engine, prod, kp, dp)
    Sel = engine.multiply(prod, masks["fmask"])                # pt
    if _MEM_FREE:
        del Xr, prod

    # ── I3  비교:  트리 전체에 sign 1회 ──                     깊이 dep(sign)
    diff = engine.subtract(Sel, tree.Theta)
    if boot_pre > 0:                            # ★ sign 직전 refresh
        diff = refresh(engine, diff, kp)
    if refresh_below is not None:
        diff = _ensure(engine, diff, kp, refresh_below, "diff")
    # ★ minimax_sign01 이 마지막에 sign_bootstrap 을 내장한다 (레벨 회복 + ±1 정리).
    #   따로 boot_post 를 걸 필요가 없다.
    B = _sign01(engine, diff, kp, secret_key, spec=sign_spec)
    # ★ B×fmask 불필요: 아래 lvl[t] 마스크가 노드 슬롯만 남기므로 중복 (레벨 1 절감)
    if _MEM_FREE:
        del Sel, diff

    # ── I4  경로 평가: 덧셈 + 평문곱만 ──                      깊이 2
    acc = None
    for t in range(D):
        n_now, width = 1 << t, 1 << (D - t)
        base = n_now - 1

        Bt = engine.multiply(B, masks["lvl"][t])
        if base:
            Bt = _rot(engine, Bt, kp, -base * dp)              # 노드 p → slot p·dp

        # stride 확장 — ① 순차(깊이 R) → ② BSGS(깊이 1)
        seq, diag = masks["spread"][t]
        for mv, shift in seq:                          # ① 순차 (깊이 R)
            hi = engine.multiply(Bt, mv)
            Bt = engine.add(engine.subtract(Bt, hi), _rot(engine, hi, kp, shift))
        sp = None
        for mv, shift in diag:                         # ② BSGS (깊이 1)
            term = engine.multiply(_rot(engine, Bt, kp, shift), mv)
            sp = term if sp is None else engine.add(sp, term)
        Bt = sp

        # 자기 구간 width 개 리프 슬롯으로 복제
        #   최대 인덱스 (L−1)·dp < L·dp ≤ slot_count → wraparound 없음, lmask 불필요
        Bt = _replicate(engine, Bt, kp, dp, width)

        # 불일치 = β + b − 2βb   (XOR).  2β 는 미리 평문으로 만들어 뒀다 → pt곱 1회, 깊이 1
        term = engine.subtract(engine.add(masks["beta"][t], Bt),
                               engine.multiply(Bt, masks["beta2"][t]))
        acc = term if acc is None else engine.add(acc, term)
        if _MEM_FREE:
            del Bt, term

    if _MEM_FREE:
        del B; gc.collect()
    return acc


# ══════════════════════════════════════════════════════════════════
# symmetric 전용 추론 — 노드 2^D 개가 아니라 D 개만 평가
# ══════════════════════════════════════════════════════════════════
def make_sym_masks(engine, d, depth):
    """symmetric 추론용 평문 마스크.  dp = 2^⌈log2 d⌉."""
    sc = engine.slot_count
    dp, L = p2(d), 1 << depth
    lvl, beta, beta2 = [], [], []
    for t in range(depth):
        m = [0.0] * sc
        m[t * dp] = 1.0                                  # 레벨 t 의 분기 슬롯
        lvl.append(engine.encode(m))
        b = [0.0] * sc
        b2 = [0.0] * sc
        for l in range(L):
            v = float((l >> (depth - t - 1)) & 1)
            b[l * dp] = v
            b2[l * dp] = 2.0 * v
        beta.append(engine.encode(b)); beta2.append(engine.encode(b2))
    fm = [0.0] * sc
    for t in range(depth):
        fm[t * dp] = 1.0
    return dict(fmask=engine.encode(fm), lvl=lvl, beta=beta, beta2=beta2,
                dp=dp, L=L, depth=depth)


def infer_symmetric(engine, Model, Theta, X_ct, sm, kp, secret_key=None,
                    sign_spec=None, boot_pre=0, boot_post=0,
                    boot_stage=SIGNBOOT_STAGE):
    """
    symmetric tree 추론.

      비대칭:  노드 2^D−1 개 평가 + stride 확장(임의 shift 대각 2^t 개)
      symmetric: **분기 D 개만 평가**, 확장도 단순 복제(2^k shift)

      D=10 이면 노드 평가 1,023 → 10  (100배),  임의 shift 회전이 사라진다.

    깊이 = PRE(2) + sign + POST(2).   spread 가 없어 비대칭보다 1 적다.
    """
    dp, L, D = sm["dp"], sm["L"], sm["depth"]
    rlk = kp.relinearization_key

    # I1  입력을 D 개 블록에 복제
    Xr = _replicate(engine, X_ct, kp, dp, p2(D))

    # I2  feature 선택
    #   ★ fmask 곱은 불필요 — 아래 lvl[t] 마스크가 같은 일을 한다 (레벨 1 절감)
    prod = engine.multiply(Xr, Model, rlk)
    Sel = _block_sum(engine, prod, kp, dp)

    # I3  비교 — 트리 전체에 sign 1회
    diff = engine.subtract(Sel, Theta)
    if boot_pre:
        diff = refresh(engine, diff, kp)
    if boot_post:
        B = _sign01(engine, diff, kp, secret_key, spec=sign_spec, to_01=False,
)
        B = sign_refresh(engine, B, kp, stage_count=boot_stage)
        from Hard_Tree.desilo.minimax_sign import to01
        B = to01(engine, B)
    else:
        B = _sign01(engine, diff, kp, secret_key, spec=sign_spec,
)

    # I4  경로 평가 — b_t 를 전 리프에 복제 후 β 와 XOR
    #     ★ 레벨 내 모든 노드가 같은 분기라 stride 확장이 필요 없다
    acc = None
    for t in range(D):
        Bt = engine.multiply(B, sm["lvl"][t])
        if t:
            Bt = _rot(engine, Bt, kp, -t * dp)           # 슬롯 0 으로
        Bt = _replicate(engine, Bt, kp, dp, L)           # 전 리프에 복제
        term = engine.subtract(engine.add(sm["beta"][t], Bt),
                               engine.multiply(Bt, sm["beta2"][t]))
        acc = term if acc is None else engine.add(acc, term)
    return acc


def ref_leaf_symmetric(x_bins, splits, n_max):
    """평문 참조 — 레벨마다 공유 분기."""
    leaf = 0
    for (j, k) in splits:
        leaf = leaf * 2 + (1 if x_bins[j] > k else 0)
    return leaf


def select_leaf(engine, acc, tree, masks, kp):
    """
    I5 — HBDT 방식.  acc 가 0 인 리프만 살아남게 하고 라벨을 더한다.
    ★ 미구현.  랜덤 다항식 곱 + 라벨 덧셈.
    """
    raise NotImplementedError("S1 이후 구현")


# ══════════════════════════════════════════════════════════════════
# stride 변경 마스크 — spread / Stage A / Stage B 공용
# ══════════════════════════════════════════════════════════════════
def build_stride_masks(engine, n_items, s_src, s_dst, seq_bits,
                       base_shift=0, extra=None):
    """
    slot[q·s_src] → slot[q·s_dst + base_shift]  (q = 0 … n_items−1) 를
    회전+평문곱으로 실현하는 마스크를 만든다.

      q = q_hi·2^m + q_lo,  R = seq_bits (상위 비트 수)
      · 순차 doubling  shift = 2^k·2^m·(s_dst−s_src)   ← 2^k 배수라 싸다
      · BSGS 대각      shift = q_lo·(s_dst−s_src)      ← 임의 shift, 비싸다

    회전 수 = R + 2^m,  깊이 = R + 1.

    ★ **순서가 방향에 따라 다르다.**
        확장(s_dst > s_src) : 순차(상위 비트) → BSGS
        압축(s_dst < s_src) : BSGS(하위 비트) → 순차(상위 비트)
      압축에서 순차를 먼저 하면 상위 블록이 아직 안 옮겨진 하위 블록 자리로
      내려와 **충돌**한다.
        예) n_items=12, s_src=8, s_dst=1, R=1, m=3
            q=1 은 슬롯 8,  q=8 은 64−56=8  → 같은 슬롯 (실측 버그)
      n_items=8 에서는 우연히 안 겹쳐 통과했다.

    반환: (ops, )  — ops = [(kind, 평문마스크, shift), ...],  kind ∈ {"seq","diag"}
          apply_stride 가 순서대로 적용한다.
    """
    sc = engine.slot_count
    n = 1 << (max(1, n_items - 1)).bit_length()
    R = max(0, min(seq_bits, n.bit_length() - 1))
    m = max(0, (n.bit_length() - 1) - R)
    delta = s_dst - s_src
    expand = delta > 0
    off = extra if extra is not None else [0]
    cur = {q: q * s_src for q in range(n_items)}

    def _seq():
        out = []
        bits = range(R - 1, -1, -1) if expand else range(R)
        for k in bits:
            sel = [q for q in range(n_items) if ((q >> m) >> k) & 1]
            if not sel:
                continue
            mv = [0.0] * sc
            for q in sel:
                for ee in off:
                    mv[cur[q] + ee] = 1.0
            sh = (1 << k) * (1 << m) * delta
            out.append(("seq", engine.encode(mv), sh))
            for q in sel:
                cur[q] += sh
        return out

    def _diag():
        out = []
        for ql in range(1 << m):
            sel = [q for q in range(n_items) if (q & ((1 << m) - 1)) == ql]
            if not sel:
                continue
            mv = [0.0] * sc
            sh = ql * delta + base_shift
            for q in sel:
                for ee in off:
                    mv[cur[q] + sh + ee] = 1.0
            out.append(("diag", engine.encode(mv), sh))
        for q in range(n_items):                       # 위치 갱신
            cur[q] += (q & ((1 << m) - 1)) * delta + base_shift
        return out

    return (_seq() + _diag()) if expand else (_diag() + _seq())


def apply_stride(engine, ct, kp, ops):
    """
    build_stride_masks 결과를 **순서대로** 적용.
      "seq"  : 해당 블록만 떼어 옮긴다 (제자리 갱신).  깊이 1/단계
      "diag" : 원본에서 출발해 누적한다.              깊이 1 (묶음 전체)
    """
    acc = None
    for kind, mv, sh in ops:
        if kind == "seq":
            if acc is not None:                        # diag 결과를 이어받는다
                ct, acc = acc, None
            hi = engine.multiply(ct, mv)
            ct = engine.add(engine.subtract(ct, hi), _rot(engine, hi, kp, sh))
        else:
            term = engine.multiply(_rot(engine, ct, kp, sh), mv)
            acc = term if acc is None else engine.add(acc, term)
    return acc if acc is not None else ct


# ══════════════════════════════════════════════════════════════════
# S3 — 학습부 모델 갱신 (argmin 출력 → (onehot_j, θ))
# ══════════════════════════════════════════════════════════════════
def nodes_per_ct(engine, dm: dict) -> int:
    """전이 버퍼(argmin 출력) 한 암호문에 들어가는 노드 수."""
    return max(1, engine.slot_count // dm["blk"])


def extract_chunks(engine, dm: dict, level: int):
    """레벨 level 을 몇 개 청크로 나눠 처리해야 하는지.  [(q0, nq), ...]"""
    per = nodes_per_ct(engine, dm)
    tot = 1 << level
    return [(q0, min(per, tot - q0)) for q0 in range(0, tot, per)]


def split_extract_budget(d, nq, max_depth):
    """
    추출 깊이 = 1(kmask) + (R_A+1) + (R_B+1) = R_A + R_B + 3.
    max_depth 안에서 R_A, R_B 를 나눈다.  R 이 클수록 임의 shift 회전이 줄지만
    깊이를 먹으므로, 모델 레벨을 높게 유지해야 하면 R 을 줄인다.
    """
    fullA = (p2(d) - 1).bit_length()
    fullB = max(1, nq - 1).bit_length()
    if max_depth is None:
        return fullA, fullB
    room = max(0, max_depth - 3)
    RA = min(fullA, (room + 1) // 2)
    RB = min(fullB, room - RA)
    return RA, max(0, RB)


def make_extract_masks(engine, dm: dict, level: int, q0: int = 0,
                       nq: int = None, seq_A: int = None, seq_B: int = None,
                       max_depth: int = None):
    """
    레벨 level 의 노드 [q0, q0+nq) 를 새 좌표계로 옮기는 마스크.

      입력  slot[(q−q0)·blk + j·n_pad + k] = 1
      출력  Model  slot[(base+q)·dp + f],  Theta  slot[(base+q)·dp]
            base = 2^level − 1  (BFS 인덱스)

    ★ 전이 버퍼가 암호문 하나를 넘으면 extract_chunks 로 나눠 호출한다.
    ★ max_depth 를 주면 그 깊이 안에 들어가도록 seq_A/seq_B 를 자동으로 줄인다.
      깊이를 줄이면 모델 레벨이 높아져 추론에서 R 을 키울 수 있다 (대신 학습 회전 증가).
      학습은 1회, 추론은 여러 번이므로 대개 깊이를 아끼는 쪽이 유리하다.
    """
    sc = engine.slot_count
    dp, n_pad, blk = dm["dp"], dm["n_pad"], dm["blk"]
    d, n_max = dm["d"], dm["n_max"]
    if nq is None:
        nq = 1 << level
    base = (1 << level) - 1 + q0
    if seq_A is None or seq_B is None:
        aA, aB = split_extract_budget(d, nq, max_depth)
        seq_A = aA if seq_A is None else seq_A
        seq_B = aB if seq_B is None else seq_B

    # θ 가중치:  w[q·blk + j·n_pad + k] = (k+1)/n_max
    w = [0.0] * sc
    for q in range(nq):
        for j in range(d):
            for k in range(n_max):
                w[q * blk + j * n_pad + k] = (k + 1.0) / n_max
    # k 축 합 후 블록 첫 슬롯만
    kmask = [0.0] * sc
    tmask = [0.0] * sc
    for q in range(nq):
        tmask[q * blk] = 1.0
        for j in range(d):
            kmask[q * blk + j * n_pad] = 1.0

    # Stage A: 노드 내부  f·n_pad → f   (압축)
    opsA = build_stride_masks(engine, d, n_pad, 1, seq_A, base_shift=0,
                              extra=[q * blk for q in range(nq)])
    # Stage B: 노드 stride  q·blk → (base+q)·dp   (압축 + 균일 이동)
    opsB = build_stride_masks(engine, nq, blk, dp, seq_B,
                              base_shift=base * dp, extra=list(range(d)))
    opsT = build_stride_masks(engine, nq, blk, dp, seq_B,
                              base_shift=base * dp, extra=[0])

    return dict(w=engine.encode(w), kmask=engine.encode(kmask),
                tmask=engine.encode(tmask), nq=nq, base=base,
                A=opsA, B=opsB, T=opsT,
                depth=1 + len(opsA) + len(opsB))


def sign_refresh(engine, ct, kp, stage_count=SIGNBOOT_STAGE, verbose=False):
    """
    부호 값(±1) 전용 bootstrap.  구현은 keypack.sign_refresh 에 있다.
      · 회복 레벨 16 − stage_count (기본 13) — refresh(~10) 보다 높다
      · 부호 잡음을 줄여 비교 정확도가 오른다
      · non-NTT 입력을 요구하므로 keypack 쪽에서 intt 를 먼저 한다
    """
    from Hard_Tree.desilo.util.keypack import sign_refresh as _sr
    return _sr(engine, ct, kp, stage_count=stage_count, verbose=verbose)


def required_level(sign_depth, R):
    """추론 1회에 필요한 최소 모델 레벨."""
    return PRE_SIGN_DEPTH + sign_depth + (R + 1) + POST_FIXED


def finalize_model(engine, tree, kp, secret_key=None, sign_depth=None, R=0,
                   mode="reencrypt", verbose=False):
    """
    학습 산출 모델을 추론 가능한 레벨로 만든다.

    ★ 왜 필요한가
      extract_model 이 레벨을 8~13 소모하고 D 레벨에 걸쳐 누적되므로,
      학습 직후 Model/Theta 는 추론 예산(18~25)을 만족하지 못한다.

    ★ mode="reencrypt"  (기본, 프로토콜 정상 경로)
      클라이언트가 모델을 받아 복호 → 재암호화.  MODEL_PROCESS 의 T8 이
      바로 이 단계이고, 학습과 추론은 원래 분리된 페이즈다
      (HBDT / HEaaN-ID3 도 동일 구조).  fresh 레벨을 얻는다.

    ★ mode="refresh"    (서버 단독)
      클라이언트 개입 없이 서버가 처리.  단 small-key bootstrap 의
      회복 레벨(L_SMALL=10)이 추론 요구치보다 낮으면 실패한다.
      회복 레벨이 더 높은 full bootstrap 이 있으면 그것을 써야 한다.
    """
    need = required_level(sign_depth, R) if sign_depth is not None else None
    before = _lvl(tree.Model)

    if mode == "reencrypt":
        assert secret_key is not None, "reencrypt 는 클라이언트 단계라 secret_key 필요"
        for name in ("Model", "Theta", "Leaf"):
            ct = getattr(tree, name)
            if ct is None:
                continue
            v = np.real(engine.decrypt(ct, secret_key))
            setattr(tree, name, engine.encrypt(list(v), kp.public_key))
    elif mode == "refresh":
        tree.Model = refresh(engine, tree.Model, kp)
        tree.Theta = refresh(engine, tree.Theta, kp)
        if tree.Leaf is not None:
            tree.Leaf = refresh(engine, tree.Leaf, kp)
        if need is not None and _lvl(tree.Model) < need:
            raise RuntimeError(
                f"refresh 회복 레벨 부족: {_lvl(tree.Model)} < 필요 {need}. "
                f"mode='reencrypt' 를 쓰거나 회복 레벨이 더 높은 bootstrap 이 필요합니다.")
    else:
        raise ValueError(mode)

    if verbose:
        print(f"    finalize({mode}): {before} → {_lvl(tree.Model)}"
              + (f"  (필요 {need})" if need else ""))
    return tree


def _lvl(ct):
    try:
        return ct.level
    except Exception:
        return -1


def extract_model(engine, A_ct, ex, dm, kp):
    """
    A_ct : FindMinPos 출력.  slot[q·blk + j·n_pad + k] one-hot
    반환 : (Model_ct, Theta_ct)  — 새 좌표계, 해당 레벨 노드만 채워짐

    깊이 : Theta 2,  Model 1 + (R_A+1) + (R_B+1)
    회전 : k축 합 log2(n_pad) + Stage A (R_A + 2^mA) + Stage B (R_B + 2^mB) ×2
    """
    n_pad, blk = dm["n_pad"], dm["blk"]

    # ── θ ──
    T = engine.multiply(A_ct, ex["w"])                     # 깊이 1
    T = _block_sum(engine, T, kp, blk)                     # 노드 블록 전체 합
    T = engine.multiply(T, ex["tmask"])                    # 깊이 1
    Theta = apply_stride(engine, T, kp, ex["T"])

    # ── onehot_j ──
    O = _block_sum(engine, A_ct, kp, n_pad)                # k 축 합
    O = engine.multiply(O, ex["kmask"])                    # 깊이 1
    O = apply_stride(engine, O, kp, ex["A"])              # Stage A
    Model = apply_stride(engine, O, kp, ex["B"])          # Stage B
    return Model, Theta

# ══════════════════════════════════════════════════════════════════
# 평문 참조 (검증용)
# ══════════════════════════════════════════════════════════════════
def ref_leaf(x_bins, feat, thr_bin, depth):
    """노드 i 에서 bin(x_{j_i}) > k_i 이면 오른쪽."""
    gid, leaf = 0, 0
    for _ in range(depth):
        b = 1 if x_bins[feat[gid]] > thr_bin[gid] else 0
        leaf = leaf * 2 + b
        gid = 2 * gid + 1 + b
    return leaf


def encode_model_plain(engine, kp, feat, thr_bin, d, n_max, depth):
    """검증용: 평문 (j,k) 로부터 Model / Theta 직접 인코딩.
       실제 학습에서는 extract_model() 이 암호문 상태로 만든다 (S3)."""
    dm = dims(d, n_max, depth)
    sc, dp, NI = engine.slot_count, dm["dp"], dm["NI"]
    mv = [0.0] * sc
    tv = [0.0] * sc
    for i in range(NI):
        mv[i * dp + int(feat[i])] = 1.0
        tv[i * dp] = (float(thr_bin[i]) + 1.0) / n_max          # ★ bin 경계
    return EncHardTreePacked(engine.encrypt(mv, kp.public_key),
                             engine.encrypt(tv, kp.public_key),
                             None, dm)
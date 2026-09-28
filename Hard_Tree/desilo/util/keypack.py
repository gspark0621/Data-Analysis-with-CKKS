# Hard_Tree/keypack.py
#
# FHE hard decision tree — KeyPack + 엔진 셋업.
# DBSCAN 코드베이스(util/keypack.py, Client_main.setup_fhe_engine)와 동일 규약.
#
# ★ 학습에는 full bootstrap_key 필요 (small 은 회복 레벨이 낮음, 실측 ~10).
#   bootstrap 은 내부에 곱셈(mod 다항 근사)이 있어 레벨을 소비하므로,
#   회복 후 도달 레벨 = (bootstrap 목표레벨). small 은 이 값이 작아 학습 1스텝(~18 소비)에 부족.
#
# ★ 추론의 sign 직후에는 sign_bootstrap 을 쓴다 (refresh 보다 유리).
#     · 회복 레벨 = 16 − stage_count  (기본 3 → 13).  small refresh(~10) 보다 높다
#     · 부호 값(±1) 전용이라 잡음까지 줄여준다 → 비교 정확도 상승
#     · 필요한 키: conjugation_key + (smallbootstrap_key 또는 lossy_bootstrap_key)
#   ※ sign_bootstrap 은 use_bootstrap_to_14_levels 파라미터와 함께 쓸 수 없다.

from dataclasses import dataclass
from typing import Optional
try:
    import desilofhe
    from desilofhe import Engine
except Exception:
    desilofhe = None; Engine = object   # 평문 폴백


@dataclass
class KeyPack:
    """DBSCAN 과 동일 필드 구성. 학습/추론에서 공유."""
    public_key:            object
    rotation_key:          object
    relinearization_key:   object
    conjugation_key:       object
    smallbootstrap_key:    object
    bootstrap_key:         Optional[object] = None   # 학습용 full bootstrap (회복 레벨 높음)
    lossy_bootstrap_key:   Optional[object] = None   # sign_bootstrap 가속용 (메모리 더 씀)


def _gpu_used_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0


def setup_fhe_engine(mode: str = "gpu", verbose: bool = True,
                     create_full_bootstrap_key: bool = False,
                     create_lossy_bootstrap_key: bool = False,
                     lossy_stage_count: int = 3):
    """
    엔진 + KeyPack 생성. DBSCAN setup_fhe_engine 과 동일 패턴.

    Parameters
    ----------
    create_full_bootstrap_key : 학습용. 회복 레벨이 높지만 메모리를 많이 쓴다.
    create_lossy_bootstrap_key: sign_bootstrap 을 더 빠르게. 메모리를 더 쓴다.
                                없으면 smallbootstrap_key 로 동작한다(느림, 메모리 적음).

    Returns
    -------
    engine, secret_key, keypack
    """
    engine     = desilofhe.Engine(use_bootstrap=True, mode=mode)
    secret_key = engine.create_secret_key()

    def _create(name, fn, *a, **kw):
        before = _gpu_used_mb() if verbose else 0.0
        key = fn(secret_key, *a, **kw)
        if verbose:
            after = _gpu_used_mb()
            print(f"  [MEM] {name:<40}  delta={after-before:+.0f} MB  (used={after:.0f} MB)")
        return key

    if verbose:
        print(f"  [MEM] Engine 초기화  used={_gpu_used_mb():.0f} MB  (slot_count={engine.slot_count:,})")

    pk  = _create("public_key",          engine.create_public_key)
    rk  = _create("rotation_key",        engine.create_rotation_key)
    rlk = _create("relinearization_key", engine.create_relinearization_key)
    ck  = _create("conjugation_key",     engine.create_conjugation_key)
    sbk = _create("smallbootstrap_key",  engine.create_small_bootstrap_key)

    if create_full_bootstrap_key:
        bk = _create("bootstrap_key",    engine.create_bootstrap_key)
    else:
        bk = None
        if verbose:
            print(f"  [MEM] bootstrap_key  생략 (small 키로 대체)  (used={_gpu_used_mb():.0f} MB)")

    if create_lossy_bootstrap_key:
        lbk = _create("lossy_bootstrap_key", engine.create_lossy_bootstrap_key,
                      stage_count=lossy_stage_count)
    else:
        lbk = None
        if verbose:
            print(f"  [MEM] lossy_bootstrap_key 생략 (sign_bootstrap 은 small 키 사용)")

    keypack = KeyPack(
        public_key=pk, rotation_key=rk, relinearization_key=rlk,
        conjugation_key=ck, smallbootstrap_key=sbk, bootstrap_key=bk,
        lossy_bootstrap_key=lbk,
    )
    if verbose:
        print(f"  [MEM] 모든 키 생성 완료  used={_gpu_used_mb():.0f} MB")
    return engine, secret_key, keypack


def refresh(engine: Engine, ct, keypack: KeyPack):
    """
    값 보존 bootstrap(레벨 회복).
    ★ bootstrap 계열 함수는 non-NTT(coefficient) 입력을 요구 →
      NTT 상태 ciphertext 를 그대로 넘기면
      'the input ciphertext should not be in NTT form' 에러 발생.
      (evaluate_chebyshev_polynomial 과 동일 제약 — gate_eval.py 참고)
    ★ full bootstrap_key 있으면 4-인자, 없으면 small 로 5-인자(rotation 포함).
    ★ **full bootstrap_key 는 만들지 않는다.**  small 키의 5-인자 형태가
      값 보존 bootstrap 으로 동작한다 ('small' 은 EvalSign 전용 키가 아니라
      키 변형).  13.4GB 를 아낀다.
    """
    ct_nonNTT = engine.intt(ct)          # ★ NTT → coefficient 변환

    if keypack.bootstrap_key is not None:
        return engine.bootstrap(
            ct_nonNTT,
            keypack.relinearization_key,
            keypack.conjugation_key,
            keypack.bootstrap_key,
        )
    else:
        return engine.bootstrap(
            ct_nonNTT,
            keypack.relinearization_key,
            keypack.conjugation_key,
            keypack.rotation_key,
            keypack.smallbootstrap_key,
        )


# ── sign_bootstrap 실측 (A6000 / DesiloFHE) ──
#   stage  입력≥  회복   시간(ms)   최대오차
#     3      3     13     2280.3    7.69e-07
#     4      4     12     1461.4    8.08e-07
#     5      5     11     1256.5    7.85e-07
#   refresh              2814.4              ← 모든 stage 보다 비싸다
#   ★ 정밀도는 stage 와 무관.  hard_core.plan_inference 가 자동 선택한다.
SIGNBOOT_MS = {3: 2280.3, 4: 1461.4, 5: 1256.5}
SIGNBOOT_STAGE = 5          # 3~5.  클수록 빠르지만 회복 레벨(16−sc)이 낮다


def signboot_level(stage_count: int = SIGNBOOT_STAGE) -> int:
    """sign_bootstrap 이후 도달 레벨."""
    return 16 - stage_count


def sign_refresh(engine: Engine, ct, keypack: KeyPack,
                 stage_count: int = SIGNBOOT_STAGE, verbose: bool = False):
    """
    부호 값(±1) 전용 bootstrap.  refresh 대비 두 가지 이점:
      ① 회복 레벨이 높다        16 − stage_count  (기본 13)  vs  small refresh ~10
      ② 부호 잡음을 줄여준다     비교 정확도가 오른다

    ★ 입력은 반드시 ±1 이어야 한다.
      minimax 합성 다항식의 출력을 (z+1)/2 하기 **전에** 걸 것.
    ★ refresh 와 마찬가지로 non-NTT 입력을 요구하므로 intt 를 먼저 한다.

    키 우선순위:  lossy_bootstrap_key (빠름) → smallbootstrap_key (메모리 적음)
    """
    try:
        if ct.level < stage_count:
            raise RuntimeError(f"sign_bootstrap 입력 레벨 부족: "
                               f"{ct.level} < stage_count {stage_count}")
    except AttributeError:
        pass
    ct_nonNTT = engine.intt(ct)          # ★ refresh 와 동일한 제약

    if keypack.lossy_bootstrap_key is not None:
        out = engine.sign_bootstrap(
            ct_nonNTT,
            keypack.relinearization_key,
            keypack.conjugation_key,
            keypack.lossy_bootstrap_key,
        )
    else:
        out = engine.sign_bootstrap(
            ct_nonNTT,
            keypack.relinearization_key,
            keypack.conjugation_key,
            keypack.rotation_key,
            keypack.smallbootstrap_key,
            stage_count,
        )
    if out is None:
        raise RuntimeError("engine.sign_bootstrap 이 None 을 반환했습니다.")
    if verbose:
        try:
            print(f"    sign_bootstrap(sc={stage_count}) → level {out.level}")
        except Exception:
            pass
    return out


# ══════════════════════════════════════════════════════════════════
# 부트스트랩 성능 · 회복 레벨 실측  (python -m Hard_Tree.keypack)
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    import numpy as np

    engine, sk, kp = setup_fhe_engine(mode="gpu", verbose=True)
    sc = engine.slot_count
    one = engine.encode([1.0] * sc)

    def drop_to(ct, target):
        while ct.level > target:
            ct = engine.multiply(ct, one)
        return ct

    print()
    print("=" * 64)
    print("부트스트랩 비교  (hard_core.py 상단 상수에 반영)")
    print("=" * 64)
    fresh = engine.encrypt([0.5] * sc, kp.public_key)
    print(f"  fresh level = {fresh.level}")

    # ── refresh ──
    ct = drop_to(engine.encrypt(list(np.random.default_rng(0).normal(0, .3, sc)),
                                kp.public_key), 4)
    t0 = time.time(); r = refresh(engine, ct, kp); t_ref = (time.time() - t0) * 1000
    print(f"  refresh          {ct.level:>3} → {r.level:<3}  {t_ref:>8.1f} ms")

    # ── sign_bootstrap ──
    v = np.sign(np.random.default_rng(1).normal(0, 1, sc))
    for stage in (3, 4, 5):
        ct = drop_to(engine.encrypt(list(v), kp.public_key), stage + 2)
        try:
            t0 = time.time()
            out = sign_refresh(engine, ct, kp, stage_count=stage)
            dt = (time.time() - t0) * 1000
            err = float(np.abs(np.real(engine.decrypt(out, sk)) - v).max())
            print(f"  sign_bootstrap sc={stage}  {ct.level:>3} → {out.level:<3}"
                  f"  {dt:>8.1f} ms   최대오차 {err:.2e}")
        except Exception as e:
            print(f"  sign_bootstrap sc={stage}  실패 — {type(e).__name__}: {e}")

    print()
    print("  hard_core.py 에 반영할 값")
    print(f"    L_FRESH        = {fresh.level}")
    print(f"    L_SMALL        = {r.level}        # refresh 회복")
    print(f"    BOOT_MS        = {t_ref:.1f}")
    print(f"    SIGNBOOT_MS    = (위 sign_bootstrap 시간)")
    print(f"    SIGNBOOT_STAGE = (오차·시간 보고 선택)")


# ══════════════════════════════════════════════════════════════════
# Cleaning — 이산값 보정 (Drucker et al. / Cheon et al.)
# ══════════════════════════════════════════════════════════════════
#   bootstrap 이 주입한 잡음을 {0,1} 또는 {±1} 로 이차수렴시킨다.
#   noise 를 **주입하지 않고** 정리한다는 점이 일반 bootstrap 과 다르다.
#   레벨 소비 2/회.  1회면 ~2^-17, 2회면 CKKS 한계.

def bit_cleaning(engine, ct, keypack, n_iters=1):
    """{0,1} 도메인:  h(x) = 3x² − 2x³ = x²(3 − 2x).   레벨 2/회."""
    sc = engine.slot_count
    rlk = keypack.relinearization_key
    three = engine.encode([3.0] * sc)
    two = engine.encode([2.0] * sc)
    x = ct
    for _ in range(n_iters):
        x2 = engine.multiply(x, x, rlk)
        x = engine.multiply(x2, engine.subtract(three, engine.multiply(x, two)), rlk)
    return x


def sign_cleaning(engine, ct, keypack, n_iters=1):
    """{−1,+1} 도메인:  g(x) = (3x − x³)/2 = x(3 − x²)/2.   레벨 2/회."""
    sc = engine.slot_count
    rlk = keypack.relinearization_key
    three = engine.encode([3.0] * sc)
    half = engine.encode([0.5] * sc)
    x = ct
    for _ in range(n_iters):
        inner = engine.subtract(three, engine.multiply(x, x, rlk))
        x = engine.multiply(engine.multiply(x, inner, rlk), half)
    return x
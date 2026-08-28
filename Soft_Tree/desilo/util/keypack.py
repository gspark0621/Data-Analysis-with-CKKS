# soft_tree/keypack.py
#
# FHE oblivious soft tree — KeyPack + 엔진 셋업.
# DBSCAN 코드베이스(util/keypack.py, Client_main.setup_fhe_engine)와 동일 규약.
#
# ★ 학습에는 full bootstrap_key 필요 (small 은 회복 레벨이 낮음, 실측 ~10).
#   bootstrap 은 내부에 곱셈(mod 다항 근사)이 있어 레벨을 소비하므로,
#   회복 후 도달 레벨 = (bootstrap 목표레벨). small 은 이 값이 작아 학습 1스텝(~18 소비)에 부족.
#   → setup_fhe_engine(create_full_bootstrap_key=True) 기본. small 은 sign_bootstrap 등에만.

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


def _gpu_used_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
    except Exception:
        return 0.0


def setup_fhe_engine(mode: str = "gpu", verbose: bool = True,
                     create_full_bootstrap_key: bool = False):  # ★ 방향 B: small 유지(OOM 회피). full 은 옵션.
    """
    엔진 + KeyPack 생성. DBSCAN setup_fhe_engine 과 동일 패턴.

    Returns
    -------
    engine, secret_key, keypack
    """
    engine     = desilofhe.Engine(use_bootstrap=True, mode=mode)
    secret_key = engine.create_secret_key()

    def _create(name, fn):
        before = _gpu_used_mb() if verbose else 0.0
        key = fn(secret_key)
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

    keypack = KeyPack(
        public_key=pk, rotation_key=rk, relinearization_key=rlk,
        conjugation_key=ck, smallbootstrap_key=sbk, bootstrap_key=bk,
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
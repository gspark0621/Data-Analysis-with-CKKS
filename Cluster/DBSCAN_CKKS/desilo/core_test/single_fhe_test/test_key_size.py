# benchmark_key_memory.py
# 완전 독립 실행 파일 - 프로젝트 내 어떤 모듈도 import하지 않음
# 실행 방법:
#   python benchmark_key_memory.py bootstrap
#   python benchmark_key_memory.py sign_bootstrap
#   python benchmark_key_memory.py both

import gc
import sys
import traceback
import math

import torch
import pynvml

from desilofhe import Engine

# ─────────────────────────────────────────────
# 유틸: GPU 메모리 (NVML 우선, torch fallback)
# ─────────────────────────────────────────────
def gpu_used_mb() -> float:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
    except Exception:
        try:
            return torch.cuda.memory_allocated(0) / (1024 ** 2)
        except Exception:
            return 0.0

def gpu_total_mb() -> float:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 2)
    except Exception:
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        except Exception:
            return 0.0

def gpu_peak_mb() -> float:
    try:
        return torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    except Exception:
        return 0.0

def gpu_reset():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass

SEP  = "=" * 65
SEP2 = "-" * 65
def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ─────────────────────────────────────────────
# GPU 메모리 델타로 key size 측정
# ─────────────────────────────────────────────
def measure_key_size(label: str, create_fn):
    gpu_reset()
    before = gpu_used_mb()
    key    = create_fn()
    after  = gpu_used_mb()
    mb     = after - before
    print(f"  {label:<26}  {mb:>10.2f} MB")
    return key, mb

# ─────────────────────────────────────────────
# Engine 파라미터 출력
# ─────────────────────────────────────────────
def print_engine_params(engine: Engine):
    section("Engine 파라미터 (메모리 관련)")

    slot_count = engine.slot_count
    log_n      = int(math.log2(slot_count * 2))   # N = 2 * slot_count → log2(N)

    total_gpu  = gpu_total_mb()
    used_gpu   = gpu_used_mb()

    print(f"\n  {'파라미터':<30}  {'값':>16}")
    print(f"  {'-'*30}  {'-'*16}")
    print(f"  {'slot_count':<30}  {slot_count:>16,}")
    print(f"  {'log N  (=log2(2*slots))':<30}  {log_n:>16}")
    print(f"  {'GPU 총 메모리':<30}  {total_gpu:>14.2f} MB")
    print(f"  {'GPU 현재 사용 메모리':<30}  {used_gpu:>14.2f} MB")
    print(f"  {'GPU 여유 메모리':<30}  {total_gpu - used_gpu:>14.2f} MB")

    return slot_count

# ─────────────────────────────────────────────
# 케이스 A: bootstrap
# ─────────────────────────────────────────────
def run_bootstrap():
    section("[ CASE A ] bootstrap")

    engine     = Engine(use_bootstrap=True, mode="gpu")
    slot_count = print_engine_params(engine)

    section("[ CASE A ] Key Size (GPU 메모리 델타 기준)")
    print(f"\n  {'Key 이름':<26}  {'크기 (MB)':>10}")
    print(f"  {'-'*26}  {'-'*10}")

    sk,  sk_mb  = measure_key_size("secret_key",          lambda: engine.create_secret_key())
    pk,  pk_mb  = measure_key_size("public_key",          lambda: engine.create_public_key(sk))
    rk,  rk_mb  = measure_key_size("rotation_key",        lambda: engine.create_rotation_key(sk))
    rlk, rlk_mb = measure_key_size("relinearization_key", lambda: engine.create_relinearization_key(sk))
    ck,  ck_mb  = measure_key_size("conjugation_key",     lambda: engine.create_conjugation_key(sk))
    bk,  bk_mb  = measure_key_size("bootstrap_key",       lambda: engine.create_bootstrap_key(sk))

    sizes = {
        "secret_key":          sk_mb,
        "public_key":          pk_mb,
        "rotation_key":        rk_mb,
        "relinearization_key": rlk_mb,
        "conjugation_key":     ck_mb,
        "bootstrap_key":       bk_mb,
    }

    # --- bootstrap() 메모리 측정 ---
    section("[ CASE A ] bootstrap() 연산 메모리 측정")
    print(f"  시그니처: engine.bootstrap(ct, rlk, ck, bootstrap_key)\n")

    dummy_ct = engine.encrypt(engine.encode([0.5] * slot_count), pk)

    gpu_reset()
    mem_before = gpu_used_mb()
    print(f"  [연산 전]  GPU 사용 메모리 : {mem_before:.2f} MB")

    delta, peak = None, None
    try:
        _ = engine.bootstrap(dummy_ct, rlk, ck, bk)
        mem_after = gpu_used_mb()
        peak      = gpu_peak_mb()
        delta     = mem_after - mem_before
        print(f"  [연산 후]  GPU 사용 메모리 : {mem_after:.2f} MB")
        print(f"  ▶ 증가량 (after - before) : {delta:+.2f} MB")
        print(f"  ▶ Peak 메모리 (torch 기준): {peak:.2f} MB")
    except Exception as e:
        print(f"  ✖ bootstrap() 실패: {e}")
        traceback.print_exc()

    del bk, dummy_ct, sk, pk, rk, rlk, ck
    gpu_reset()

    return sizes, delta, peak

# ─────────────────────────────────────────────
# 케이스 B: sign_bootstrap
# ─────────────────────────────────────────────
def run_sign_bootstrap():
    section("[ CASE B ] sign_bootstrap")

    engine     = Engine(use_bootstrap=True, mode="gpu")
    slot_count = print_engine_params(engine)

    section("[ CASE B ] Key Size (GPU 메모리 델타 기준)")
    print(f"\n  {'Key 이름':<26}  {'크기 (MB)':>10}")
    print(f"  {'-'*26}  {'-'*10}")

    sk,  sk_mb  = measure_key_size("secret_key",          lambda: engine.create_secret_key())
    pk,  pk_mb  = measure_key_size("public_key",          lambda: engine.create_public_key(sk))
    rk,  rk_mb  = measure_key_size("rotation_key",        lambda: engine.create_rotation_key(sk))
    rlk, rlk_mb = measure_key_size("relinearization_key", lambda: engine.create_relinearization_key(sk))
    ck,  ck_mb  = measure_key_size("conjugation_key",     lambda: engine.create_conjugation_key(sk))
    sbk, sbk_mb = measure_key_size("sign_bootstrap_key",  lambda: engine.create_lossy_bootstrap_key(sk))

    sizes = {
        "secret_key":          sk_mb,
        "public_key":          pk_mb,
        "rotation_key":        rk_mb,
        "relinearization_key": rlk_mb,
        "conjugation_key":     ck_mb,
        "sign_bootstrap_key":  sbk_mb,
    }

    # --- sign_bootstrap() 메모리 측정 ---
    section("[ CASE B ] sign_bootstrap() 연산 메모리 측정")
    print(f"  시그니처: engine.sign_bootstrap(ct, rlk, ck, sign_bootstrap_key)\n")

    dummy_ct = engine.encrypt(engine.encode([0.5] * slot_count), pk)

    gpu_reset()
    mem_before = gpu_used_mb()
    print(f"  [연산 전]  GPU 사용 메모리 : {mem_before:.2f} MB")

    delta, peak = None, None
    try:
        _ = engine.sign_bootstrap(dummy_ct, rlk, ck, sbk)
        mem_after = gpu_used_mb()
        peak      = gpu_peak_mb()
        delta     = mem_after - mem_before
        print(f"  [연산 후]  GPU 사용 메모리 : {mem_after:.2f} MB")
        print(f"  ▶ 증가량 (after - before) : {delta:+.2f} MB")
        print(f"  ▶ Peak 메모리 (torch 기준): {peak:.2f} MB")
    except Exception as e:
        print(f"  ✖ sign_bootstrap() 실패: {e}")
        traceback.print_exc()

    del sbk, dummy_ct, sk, pk, rk, rlk, ck
    gpu_reset()

    return sizes, delta, peak

# ─────────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────────
def print_summary(sizes_a, delta_a, peak_a, sizes_b, delta_b, peak_b):
    section("최종 요약")

    common_keys = ["secret_key", "public_key", "rotation_key",
                   "relinearization_key", "conjugation_key"]

    print(f"\n  [ 공통 Key Size (두 케이스 동일 조건) ]")
    print(f"  {'Key 이름':<26}  {'크기 (MB)':>10}")
    print(f"  {'-'*26}  {'-'*10}")
    for name in common_keys:
        print(f"  {name:<26}  {sizes_a[name]:>10.2f} MB")

    print(f"\n  [ Bootstrap Key Size 비교 ]")
    print(f"  {'Key 이름':<26}  {'크기 (MB)':>10}  {'비고'}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*30}")
    print(f"  {'bootstrap_key':<26}  {sizes_a['bootstrap_key']:>10.2f} MB  create_bootstrap_key")
    print(f"  {'sign_bootstrap_key':<26}  {sizes_b['sign_bootstrap_key']:>10.2f} MB  create_lossy_bootstrap_key")

    key_diff  = sizes_b["sign_bootstrap_key"] - sizes_a["bootstrap_key"]
    key_ratio = sizes_b["sign_bootstrap_key"] / max(sizes_a["bootstrap_key"], 1e-9)
    print(f"\n  ▶ sign_bootstrap_key - bootstrap_key : {key_diff:+.2f} MB")
    print(f"  ▶ sign_bootstrap_key / bootstrap_key  : {key_ratio:.3f}x")

    print(f"\n  [ 연산 메모리 비교 (GPU) ]")
    print(f"  {'연산':<22}  {'증가량 (MB)':>12}  {'Peak (MB)':>12}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}")

    str_da = f"{delta_a:>+12.2f}" if delta_a is not None else f"{'실패':>12}"
    str_pa = f"{peak_a:>12.2f}"   if peak_a  is not None else f"{'N/A':>12}"
    str_db = f"{delta_b:>+12.2f}" if delta_b is not None else f"{'실패':>12}"
    str_pb = f"{peak_b:>12.2f}"   if peak_b  is not None else f"{'N/A':>12}"

    print(f"  {'bootstrap()':<22}  {str_da}  {str_pa}")
    print(f"  {'sign_bootstrap()':<22}  {str_db}  {str_pb}")

    if delta_a is not None and delta_b is not None:
        op_diff = delta_b - delta_a
        pk_diff = (peak_b or 0) - (peak_a or 0)
        print(f"\n  ▶ 증가량 차이 (sign - boot)  : {op_diff:+.2f} MB")
        print(f"  ▶ Peak 차이  (sign - boot)  : {pk_diff:+.2f} MB")

    print(f"\n{SEP}\n")

# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "bootstrap":
        run_bootstrap()

    elif mode == "sign_bootstrap":
        run_sign_bootstrap()

    elif mode == "both":
        sizes_a, delta_a, peak_a = run_bootstrap()
        sizes_b, delta_b, peak_b = run_sign_bootstrap()
        print_summary(sizes_a, delta_a, peak_a, sizes_b, delta_b, peak_b)

    else:
        print("사용법: python benchmark_key_memory.py [bootstrap|sign_bootstrap|both]")
        sys.exit(1)
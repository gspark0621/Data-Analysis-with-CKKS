# test_sign_comparison.py
#
# ─────────────────────────────────────────────────────────────────────────────
# 두 sign 근사 함수의 비교 테스트
#
# 방법 1: f(x) = (3/2)x - (1/2)x³  를 14번 iteration + 일반 bootstrap
#         - 레벨 budget 관리를 위해 bootstrap_interval=3마다 중간 bootstrap
#         - 마지막에 명시적으로 일반 bootstrap 1회 추가 (방법 2의 sign_bootstrap에 대응)
#
# 방법 2: mcp_alpha15_lp_cheb.json (이미 생성된 파일)에서 계수 추출
#         → eval_mcp_full_chebyshev (Chebyshev BSGS) → engine.sign_bootstrap
#         (= Label_Propagation.fhe_sgn 과 완전히 동일한 파이프라인)
#
# 입력 데이터:
#   [-1, 1] 구간의 uniform random 100개 (seed 고정 → 재현 가능)
#
# 출력:
#   각 점별 (x, true_sign, M1_val, M1_err, M2_val, M2_err)
#   요약: max error / mean error
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
from time import time

import desilofhe
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev


# ─── 경로 설정 (test code와 동일) ─────────────────────────────────────────────
_MCP_PATH       = "mcp_alpha15_lp_cheb.json"   # 이미 생성된 JSON 사용
_NUM_POINTS     = 100
_NUM_ITERATIONS = 14
_BS_INTERVAL    = 3                            # 방법 1의 중간 bootstrap 주기
_RANDOM_SEED    = 42


# ═════════════════════════════════════════════════════════════════════════════
# 방법 1: (3/2)x - (1/2)x³ 반복
# ═════════════════════════════════════════════════════════════════════════════

def method1_iterative_poly(engine, ct, num_iters, keypack,
                            bootstrap_interval: int = 3):
    """
    f(x) = (3/2)x - (1/2)x³ 를 num_iters번 반복.

    한 iteration 레벨 소비 (DesiloFHE lazy rescaling 기준):
       - x_sq  = square(x)              : ~1 level
       - x_cub = multiply(x_sq, x)      : ~1 level
       - term1 = multiply(x, 1.5_pt)    : ~1 level
       - term2 = multiply(x_cub, 0.5_pt): ~1 level
       - subtract(term1, term2)         : 0 level
       → 한 iteration 당 약 2~4 level 소비
       → 안전을 위해 bootstrap_interval=3 권장

    마지막 iteration 후에는 명시적으로 일반 bootstrap 1회 추가하여
    fresh ciphertext 상태로 마무리.
    """
    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    boot_key   = keypack.bootstrap_key
    slot_count = engine.slot_count

    # 평문 상수 인코딩 (재사용)
    c15_pt = engine.encode([1.5] * slot_count)
    c05_pt = engine.encode([0.5] * slot_count)

    current = ct
    for i in range(num_iters):
        # f(x) = 1.5x - 0.5x³
        x_sq    = engine.square(current, relin_key)
        x_cub   = engine.multiply(x_sq, current, relin_key)
        term1   = engine.multiply(current, c15_pt)
        term2   = engine.multiply(x_cub,   c05_pt)
        current = engine.subtract(term1, term2)

        # 중간 bootstrap (마지막 iter는 건너뜀 — 어차피 아래에서 한 번 더 함)
        if (i + 1) % bootstrap_interval == 0 and (i + 1) != num_iters:
            print(f"    [M1] iter {i+1}/{num_iters}: 중간 bootstrap")
            current = engine.bootstrap(
                engine.intt(current),
                relin_key, conj_key, boot_key,
            )
        else:
            print(f"    [M1] iter {i+1}/{num_iters} 완료")

    # ── 마지막 일반 bootstrap (사용자 요구사항: "+ 일반 bootstrap") ──────
    print(f"    [M1] 최종 일반 bootstrap")
    current = engine.bootstrap(
        engine.intt(current),
        relin_key, conj_key, boot_key,
    )
    return current


# ═════════════════════════════════════════════════════════════════════════════
# 방법 2: MCP (Chebyshev) + sign_bootstrap
# ═════════════════════════════════════════════════════════════════════════════

def method2_mcp_sign_bootstrap(engine, ct, components, keypack):
    """
    Label_Propagation.fhe_sgn 과 완전히 동일한 파이프라인:
       1. eval_mcp_full_chebyshev (Chebyshev BSGS, Bossuat Algorithm 1)
       2. engine.sign_bootstrap (smallbootstrap_key 사용)
    """
    slot_count = engine.slot_count

    # 1) Chebyshev MCP 평가
    result = eval_mcp_full_chebyshev(
        engine, ct, components, slot_count, keypack,
        tag="M2 ", debug=False,
    )

    # 2) sign_bootstrap
    print(f"    [M2] sign_bootstrap")
    result = engine.sign_bootstrap(
        engine.intt(result),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 메인
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("  Sign 근사 함수 비교 테스트")
    print(f"    데이터: U(-1, 1) {_NUM_POINTS}개 (seed={_RANDOM_SEED})")
    print(f"    방법 1: (3/2)x - (1/2)x³ × {_NUM_ITERATIONS} iter + 일반 bootstrap")
    print(f"            (중간 bootstrap 주기={_BS_INTERVAL})")
    print(f"    방법 2: {_MCP_PATH} + sign_bootstrap")
    print("=" * 78)

    # ── Step 1: MCP 파일 로드 ─────────────────────────────────────────
    print(f"\n[Step 1] MCP 파일 로드: {_MCP_PATH}")
    if not os.path.exists(_MCP_PATH):
        raise FileNotFoundError(
            f"{_MCP_PATH} 가 존재하지 않습니다.\n"
            f"먼저 test_Server_main_call_dataset.py 또는 _ensure_mcp_files()를 "
            f"한 번 실행하여 JSON을 생성하세요."
        )
    components = load_mcp(_MCP_PATH)
    basis = components[0].get("basis", "power")
    if basis != "chebyshev":
        raise ValueError(
            f"{_MCP_PATH} 의 basis='{basis}', expected 'chebyshev'. "
            f"compute_mcp_for_label_prop_chebyshev() 로 재생성하세요."
        )
    print(f"  components       = {len(components)}")
    print(f"  degrees          = {[c['degree'] for c in components]}")
    print(f"  domain_a (= τ)   = {components[0]['domain_a']:.4e} "
          f"(= 2^{np.log2(components[0]['domain_a']):.2f})")
    print(f"  final sign_err   = {components[-1]['error']:.4e}")

    # ── Step 2: 테스트 데이터 생성 ───────────────────────────────────
    print(f"\n[Step 2] [-1, 1] uniform 데이터 생성")
    rng        = np.random.default_rng(_RANDOM_SEED)
    x_values   = rng.uniform(-1.0, 1.0, size=_NUM_POINTS)
    true_signs = np.sign(x_values)  # 정확한 ±1 (x=0 확률 0)

    n_pos = int((true_signs > 0).sum())
    n_neg = int((true_signs < 0).sum())
    print(f"  x range : [{x_values.min():+.6f}, {x_values.max():+.6f}]")
    print(f"  positive: {n_pos}개,  negative: {n_neg}개")
    print(f"  min |x| : {np.min(np.abs(x_values)):.4e} "
          f"(2^{np.log2(np.min(np.abs(x_values))):.2f})")

    # ── Step 3: FHE 엔진/키 생성 (직접 선언, mode="cpu") ────────────
    print(f"\n[Step 3] FHE 엔진 및 키 생성 (직접 선언, mode='cpu')")
    engine     = desilofhe.Engine(use_bootstrap=True, mode="cpu")
    secret_key = engine.create_secret_key()

    public_key          = engine.create_public_key(secret_key)
    rotation_key        = engine.create_rotation_key(secret_key)
    relinearization_key = engine.create_relinearization_key(secret_key)
    conjugation_key     = engine.create_conjugation_key(secret_key)
    bootstrap_key       = engine.create_bootstrap_key(secret_key)
    smallbootstrap_key  = engine.create_small_bootstrap_key(secret_key)

    keypack = KeyPack(
        public_key=public_key,
        rotation_key=rotation_key,
        relinearization_key=relinearization_key,
        conjugation_key=conjugation_key,
        bootstrap_key=bootstrap_key,
        smallbootstrap_key=smallbootstrap_key,
    )
    slot_count = engine.slot_count
    print(f"  모든 키 생성 완료 (public/rotation/relin/conj/bootstrap/smallbootstrap)")
    print(f"  slot_count = {slot_count:,}  (100개 < slot_count ✓)")

    # ── Step 4: 데이터 암호화 (한 ciphertext에 100개 packing) ────────
    print(f"\n[Step 4] 데이터 암호화 (한 ciphertext에 100개 packing)")
    padded   = list(x_values) + [0.0] * (slot_count - _NUM_POINTS)
    encoded  = engine.encode(padded)
    ct_x     = engine.encrypt(encoded, secret_key)
    print(f"  encrypted: {_NUM_POINTS}개 데이터 + {slot_count - _NUM_POINTS}개 0.0 padding")

    # 암호화된 데이터의 sanity check (decrypt 후 잘 들어갔는지)
    dec_check = np.real(engine.decrypt(ct_x, secret_key))[:_NUM_POINTS]
    enc_err   = np.max(np.abs(dec_check - x_values))
    print(f"  encryption sanity: max|dec - x| = {enc_err:.4e}")

    # ── Step 5: 방법 1 실행 ──────────────────────────────────────────
    print(f"\n[Step 5] 방법 1 실행: (3/2)x - (1/2)x³ × {_NUM_ITERATIONS} iter + 일반 bootstrap")
    t0 = time()
    ct_m1 = method1_iterative_poly(
        engine, ct_x, _NUM_ITERATIONS, keypack,
        bootstrap_interval=_BS_INTERVAL,
    )
    time_m1 = time() - t0
    print(f"  → 완료: {time_m1:.2f}초")

    decrypted_m1 = np.real(engine.decrypt(ct_m1, secret_key))[:_NUM_POINTS]

    # ── Step 6: 방법 2 실행 ──────────────────────────────────────────
    print(f"\n[Step 6] 방법 2 실행: MCP + sign_bootstrap")
    t0 = time()
    ct_m2 = method2_mcp_sign_bootstrap(engine, ct_x, components, keypack)
    time_m2 = time() - t0
    print(f"  → 완료: {time_m2:.2f}초")

    decrypted_m2 = np.real(engine.decrypt(ct_m2, secret_key))[:_NUM_POINTS]

    # ── Step 7: 점별 오차 출력 ───────────────────────────────────────
    err_m1 = np.abs(decrypted_m1 - true_signs)
    err_m2 = np.abs(decrypted_m2 - true_signs)

    print(f"\n[Step 7] 점별 오차 (vs ideal sign(x) = ±1)")
    print()
    header = (f"  {'idx':>3} | {'x':>10} | {'sign(x)':>7} | "
              f"{'M1_val':>12} | {'M1_err':>11} | "
              f"{'M2_val':>12} | {'M2_err':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    # |x| 오름차순으로 정렬해 0 근방 어려운 점이 먼저 보이도록
    order_by_absx = np.argsort(np.abs(x_values))
    for i in order_by_absx:
        flag_m1 = "" if err_m1[i] < 1e-2 else " ⚠"
        flag_m2 = "" if err_m2[i] < 1e-2 else " ⚠"
        print(f"  {i:>3} | {x_values[i]:>+10.6f} | {int(true_signs[i]):>+7d} | "
              f"{decrypted_m1[i]:>+12.6f} | {err_m1[i]:>11.4e}{flag_m1:<2} | "
              f"{decrypted_m2[i]:>+12.6f} | {err_m2[i]:>11.4e}{flag_m2:<2}")

    # ── Step 8: 요약 통계 ────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  요약 통계 (각 방법의 |출력 - sign(x)| 분포)")
    print("=" * 78)

    print(f"\n[방법 1] (3/2)x - (1/2)x³ × {_NUM_ITERATIONS} iter + 일반 bootstrap")
    print(f"  Max  error : {err_m1.max():.6e}")
    print(f"  Mean error : {err_m1.mean():.6e}")
    print(f"  Median err : {np.median(err_m1):.6e}")
    print(f"  Min  error : {err_m1.min():.6e}")
    print(f"  소요 시간  : {time_m1:.2f}초")
    print(f"  argmax     : idx={int(err_m1.argmax())},  "
          f"x={x_values[err_m1.argmax()]:+.6f}, "
          f"|x|={abs(x_values[err_m1.argmax()]):.4e}")

    print(f"\n[방법 2] MCP({_MCP_PATH}) + sign_bootstrap")
    print(f"  Max  error : {err_m2.max():.6e}")
    print(f"  Mean error : {err_m2.mean():.6e}")
    print(f"  Median err : {np.median(err_m2):.6e}")
    print(f"  Min  error : {err_m2.min():.6e}")
    print(f"  소요 시간  : {time_m2:.2f}초")
    print(f"  argmax     : idx={int(err_m2.argmax())},  "
          f"x={x_values[err_m2.argmax()]:+.6f}, "
          f"|x|={abs(x_values[err_m2.argmax()]):.4e}")

    print(f"\n[직접 비교]")
    print(f"  Max  error 비율 (M2 / M1): {err_m2.max() / max(err_m1.max(), 1e-30):.4e}")
    print(f"  Mean error 비율 (M2 / M1): {err_m2.mean() / max(err_m1.mean(), 1e-30):.4e}")
    print(f"  시간 비율  (M2 / M1)     : {time_m2 / max(time_m1, 1e-30):.4f}")

    # ── (참고) 영역별 오차 분석 ──────────────────────────────────────
    print()
    print("=" * 78)
    print("  영역별 오차 (|x| 구간별)")
    print("=" * 78)
    bins = [0.0, 2**-15, 2**-10, 2**-5, 2**-2, 1.0]
    bin_labels = [f"[0, 2^-15)", f"[2^-15, 2^-10)",
                  f"[2^-10, 2^-5)", f"[2^-5, 2^-2)", f"[2^-2, 1]"]
    print(f"\n  {'구간':<18} | {'count':>5} | "
          f"{'M1 max':>11} | {'M1 mean':>11} | "
          f"{'M2 max':>11} | {'M2 mean':>11}")
    print("  " + "-" * 84)
    for lo, hi, lab in zip(bins[:-1], bins[1:], bin_labels):
        mask = (np.abs(x_values) >= lo) & (np.abs(x_values) < hi)
        if not mask.any():
            print(f"  {lab:<18} | {0:>5} | {'-':>11} | {'-':>11} | "
                  f"{'-':>11} | {'-':>11}")
            continue
        print(f"  {lab:<18} | {int(mask.sum()):>5} | "
              f"{err_m1[mask].max():>11.4e} | {err_m1[mask].mean():>11.4e} | "
              f"{err_m2[mask].max():>11.4e} | {err_m2[mask].mean():>11.4e}")

    print("\n" + "=" * 78)
    print("  ✓ 비교 테스트 완료")
    print("=" * 78)


if __name__ == "__main__":
    main()
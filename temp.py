"""
GLEngine (desilofhe) 부트스트래핑 벤치마크 스크립트 (v2)
========================================================

변경점 (v1 -> v2):
  - 암호화 직후(레벨 최대)가 아니라, level을 0까지 완전히 소모한 뒤
    bootstrap을 호출하도록 수정했습니다. 이게 실제 사용 패턴과 일치하는
    조건입니다 (bootstrap은 보통 더 이상 곱셈을 할 수 없는 상태에서 호출).
  - 참고: bootstrap 자체의 소요 시간은 "level을 몇 번의 곱셈으로 소모했는지"와는
    거의 무관합니다 (bootstrap 비용은 파라미터 구조에 의존하는 고정 비용).
    다만 level=0이라는 조건에서 실행해야 실제 라이브러리 사용법과 맞는
    측정이 되므로, 아래 코드는 그 조건을 맞추는 데 초점을 둡니다.
  - 대칭키(secret key) 기반 암호화(encrypt_symmetric 계열)를 기본으로 사용합니다.
    (공개키 암호화가 필요 없는 단일 사용자 시나리오를 가정)

주의 (실행 전 반드시 확인):
  - 이 코드는 desilofhe의 공개된 CKKS `Engine` API 패턴을 GLEngine에도
    그대로 적용한다고 "가정"하고 작성되었습니다. 실제 함수명(특히
    symmetric encrypt, multiply, rescale, bootstrap_key 생성 관련)은
    사용 중인 버전의 https://fhe.desilo.dev/<version>/api/gl-engine/
    문서에서 확인 후 KEY_SETUP / ENCRYPT / CONSUME_LEVEL / BOOTSTRAP_CALL
    표시된 부분만 맞게 고치면 됩니다.
"""

import time
import csv
import traceback

import numpy as np
from desilofhe import GLEngine

# -----------------------------------------------------------------engine, 
# 실험 설정
# -----------------------------------------------------------------

SHAPES = [
    (256, 16, 16),
    (16, 256, 256),
    (4, 1024, 1024),
    (256, 32, 32),
    (16, 512, 512),
    (4, 2048, 2048),
    (256, 64, 64),
]

MODE = "cpu"          # "cpu" 또는 "gpu"
THREAD_COUNT = None    # GPU 사용 시 예: 512
DEVICE_ID = None       # GPU 사용 시 예: 1
CUDA_GRID_SIZE_MULTIPLIER = -1  # -1 = 자동

OUTPUT_CSV = "gl_bootstrap_benchmark.csv"


def build_engine(shape):
    kwargs = {"shape": shape}
    if MODE == "gpu":
        kwargs["mode"] = "gpu"
        if THREAD_COUNT is not None:
            kwargs["thread_count"] = THREAD_COUNT
        if DEVICE_ID is not None:
            kwargs["device_id"] = DEVICE_ID
        kwargs["cuda_grid_size_multiplier"] = CUDA_GRID_SIZE_MULTIPLIER
    return GLEngine(**kwargs)


def consume_all_levels(engine, ciphertext, plaintext_one):
    """
    level이 0이 될 때까지 Hadamard product(ct x pt) + rescale을 반복.
    plaintext_one은 값이 폭주하지 않도록 1에 가까운 값으로 encode된 평문.
    """
    steps = 0
    while ciphertext.level > 0:
        ciphertext = engine.multiply(ciphertext, plaintext_one)
        ciphertext = engine.rescale(ciphertext)
        steps += 1
        if steps > 200:  # 무한루프 방지 안전장치
            raise RuntimeError("level이 0으로 줄어들지 않습니다 (API 확인 필요)")
    return ciphertext, steps


def run_case(shape):
    engine = build_engine(shape)

    # ---------------------------------------------------------
    # 1) 키 생성 (KEY_SETUP)
    # ---------------------------------------------------------
    secret_key = engine.create_secret_key()
    matrix_multiplication_key = engine.create_matrix_multiplication_key(secret_key)
    bootstrap_key = engine.create_bootstrap_key(secret_key)

    # ---------------------------------------------------------
    # 2) 대칭키 방식으로 fresh ciphertext 생성 (ENCRYPT)
    #    -> 비밀키 소유자가 직접 암호화하는 일반적인 시나리오
    # ---------------------------------------------------------
    batch, n1, n2 = shape
    message = np.random.uniform(-1, 1, size=shape)
    encoded = engine.encode(message)
    ciphertext = engine.encrypt(encoded, secret_key)  # symmetric-key encryption

    fresh_level = ciphertext.level

    # 곱셈에 쓸 "1에 가까운" 평문 (값 폭주 방지)
    one_message = np.full(shape, 0.999)
    plaintext_one = engine.encode(one_message)

    # ---------------------------------------------------------
    # 3) level을 0까지 완전히 소모 (CONSUME_LEVEL)
    # ---------------------------------------------------------
    ciphertext_at_zero, mult_steps = consume_all_levels(
        engine, ciphertext, plaintext_one
    )
    level_before_bootstrap = ciphertext_at_zero.level  # 기대값: 0

    # ---------------------------------------------------------
    # 4) Bootstrap 실행 및 시간 측정 (BOOTSTRAP_CALL)
    # ---------------------------------------------------------
    start = time.perf_counter()
    bootstrapped = engine.bootstrap(
        ciphertext_at_zero,
        matrix_multiplication_key,
        bootstrap_key,
    )
    elapsed = time.perf_counter() - start

    level_after_bootstrap = bootstrapped.level

    return {
        "shape": str(shape),
        "batch": batch,
        "rows": n1,
        "cols": n2,
        "fresh_level": fresh_level,
        "mult_steps_to_zero": mult_steps,
        "level_before_bootstrap": level_before_bootstrap,
        "level_after_bootstrap": level_after_bootstrap,
        "bootstrap_time_sec": round(elapsed, 6),
        "status": "ok",
    }


def main():
    results = []

    for shape in SHAPES:
        print(f"\n=== shape={shape} ({MODE}) ===")
        try:
            row = run_case(shape)
            print(
                f"  fresh_level={row['fresh_level']}, "
                f"consumed_in={row['mult_steps_to_zero']} mults, "
                f"level_before_bootstrap={row['level_before_bootstrap']}, "
                f"level_after_bootstrap={row['level_after_bootstrap']}, "
                f"bootstrap_time={row['bootstrap_time_sec']}s"
            )
        except Exception as e:
            traceback.print_exc()
            row = {
                "shape": str(shape),
                "batch": shape[0],
                "rows": shape[1],
                "cols": shape[2],
                "fresh_level": None,
                "mult_steps_to_zero": None,
                "level_before_bootstrap": None,
                "level_after_bootstrap": None,
                "bootstrap_time_sec": None,
                "status": f"error: {type(e).__name__}: {e}",
            }
            print(f"  실패: {row['status']}")
        results.append(row)

    fieldnames = [
        "shape",
        "batch",
        "rows",
        "cols",
        "fresh_level",
        "mult_steps_to_zero",
        "level_before_bootstrap",
        "level_after_bootstrap",
        "bootstrap_time_sec",
        "status",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n결과를 '{OUTPUT_CSV}'에 저장했습니다.")

    print("\n{:<20}{:>10}{:>10}{:>10}{:>15}".format(
        "shape", "fresh_lv", "pre_boot", "post_boot", "time(sec)"
    ))
    for r in results:
        print("{:<20}{:>10}{:>10}{:>10}{:>15}".format(
            r["shape"],
            str(r["fresh_level"]),
            str(r["level_before_bootstrap"]),
            str(r["level_after_bootstrap"]),
            str(r["bootstrap_time_sec"]),
        ))


if __name__ == "__main__":
    main()
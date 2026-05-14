import desilofhe
import numpy as np

engine = desilofhe.Engine(use_bootstrap=True, mode="gpu")
secret_key = engine.create_secret_key()
relin_key  = engine.create_relinearization_key(secret_key)
conj_key   = engine.create_conjugation_key(secret_key)
rot_key    = engine.create_rotation_key(secret_key)
boot_key   = engine.create_bootstrap_key(secret_key)
sboot_key  = engine.create_small_bootstrap_key(secret_key)

N = engine.slot_count
data = [0.9] * N

# 1) sign_bootstrap가 최소 몇 레벨을 요구하는지 탐색
ct_fresh = engine.encrypt(data, engine.create_public_key(secret_key))
ct = ct_fresh
for i in range(24):
    ct = engine.multiply(ct, engine.encode([1.0]*N))
print(f"fresh 암호문 레벨: {ct_fresh.level}")

# 레벨을 하나씩 소진하면서 sign_bootstrap 시도
for i in range(ct_fresh.level + 1):
    try:
        ct_test = engine.sign_bootstrap(
            engine.intt(ct), relin_key, conj_key, rot_key, sboot_key
        )
        print(f"레벨 {ct.level} → sign_bootstrap 성공, 출력 레벨: {ct_test.level}")
        break
    except RuntimeError as e:
        print(f"레벨 {ct.level} → 실패: {e}")
        # 레벨 하나 소진
        ct = engine.multiply(ct, engine.encode([1.0]*N))

# 2) 일반 bootstrap 후 레벨 확인
ct2 = engine.encrypt(data, engine.create_public_key(secret_key))
ct2_after = engine.bootstrap(engine.intt(ct2), relin_key, conj_key, boot_key)
print(f"\n일반 bootstrap 출력 레벨: {ct2_after.level}")

# 3) sign_bootstrap 후 레벨 확인 (성공한 경우)
print(f"sign_bootstrap 출력 레벨: {ct_test.level}")
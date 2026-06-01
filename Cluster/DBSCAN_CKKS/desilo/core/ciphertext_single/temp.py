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

ct_1 = engine.encrypt(engine.encode(data), secret_key)
ct_2 = engine.multiply(ct_1, ct_1)
ct_3 = engine.add(ct_1, ct_2)
print(engine.decrypt(ct_3, secret_key)[:10])
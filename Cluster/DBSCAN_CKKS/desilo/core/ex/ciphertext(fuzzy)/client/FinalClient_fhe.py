"""
FinalClientFHE
  Phase 1 : Multiparty 키 생성 (코디네이터 역할)
  Phase 4 : Multiparty 복호화 + 라벨 추출
"""
from desilofhe import Engine
from core.ciphertext.shared.keypack import KeyPack
from typing import List, Any, Dict
import numpy as np

BOOTSTRAP_STAGE_COUNT = 3   # sign_bootstrap 정밀도 설정


class FinalClientFHE:
    def __init__(self):
        self.engine     = Engine(use_bootstrap=True, mode="gpu")
        self.secret_key = None
        self.keypack    = None

    # ── Phase 1: 키 생성 ──────────────────────────────────────────
    def generate_keys(self) -> KeyPack:
        e = self.engine
        self.secret_key = e.create_secret_key()

        self.keypack = KeyPack(
            engine              = e, 
            public_key          = e.create_public_key(self.secret_key),
            rotation_key        = e.create_rotation_key(self.secret_key),
            relinearization_key = e.create_relinearization_key(self.secret_key),
            conjugation_key     = e.create_conjugation_key(self.secret_key),
            lossy_bootstrap_key = e.create_lossy_bootstrap_key(self.secret_key),
        )
        return self.keypack

    # ── Phase 4: 복호화 ───────────────────────────────────────────
    def decrypt(self, ct: Any) -> np.ndarray:
        result = self.engine.decrypt(ct, self.secret_key)
        return np.array(result)
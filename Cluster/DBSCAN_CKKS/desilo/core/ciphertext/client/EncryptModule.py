# core/client/EncryptModule.py
import desilofhe
from typing import List
from util.keypack import KeyPack


class DimensionalEncryptor:
    def __init__(self, engine: desilofhe.Engine, keypack: KeyPack):
        self.engine = engine
        self.keypack = keypack

    def encrypt_data(self, pt: List[List[float]], dim: int):
        if not pt:
            raise ValueError("Input plaintext 'pt' is empty.")
        if len(pt[0]) != dim:
            raise ValueError(f"Input data dimension ({len(pt[0])}) does not match dim ({dim}).")

        transposed_data = list(zip(*pt))
        if len(transposed_data) != dim:
            raise ValueError("Transposed data dimensions mismatch.")

        ciphertexts = []
        print(f"[Client] Starting encryption for {dim} dimensions...")
        for i in range(dim):
            vector_data = list(transposed_data[i])
            ctxt = self.engine.encrypt(vector_data, self.keypack.public_key)
            ciphertexts.append(ctxt)

        return ciphertexts
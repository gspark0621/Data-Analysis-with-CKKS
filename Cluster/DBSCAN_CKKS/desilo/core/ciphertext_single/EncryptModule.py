# EncryptModule.py
import desilofhe
from typing import List
from util.keypack import KeyPack


# Encryption Module
# packing 방식 구현(차원에 따라 같은 좌표끼리 묶는 ciphertext(x좌표 끼리, y좌표끼리 ...) 생성)

class DimensionalEncryptor:
    def __init__(self, engine: desilofhe.Engine, keypack: KeyPack):
        """
        초기화 메서드
        :param engine: desilofhe의 Engine 인스턴스 (암호화 수행 주체)
        :param keypack: public_key 등이 포함된 KeyPack 객체
        """
        self.engine = engine
        self.keypack = keypack

    def encrypt_data(self, pt: List[List[float]], dim: int) -> List[desilofhe.Ciphertext]:     
        # 1. 데이터 검증 (Optional but recommended)
        if not pt:
            raise ValueError("Input plaintext 'pt' is empty.")
        if len(pt[0]) != dim:
            raise ValueError(f"Input data dimension ({len(pt[0])}) does not match the provided dim ({dim}).")
        # 예: [[x1, y1], [x2, y2]] -> [[x1, x2], [y1, y2]]
        # 2. Python의 zip(*pt)를 사용하여 리스트를 전치(transpose)합니다.
        transposed_data = list(zip(*pt))
        # 차원 수 검증 (전치된 결과의 길이가 dim과 같아야 함)
        if len(transposed_data) != dim:
            raise ValueError("Transposed data dimensions mismatch.")


        ciphertexts = []
        # 3. 차원별 암호화 진행
        print(f"[Client] Starting encryption for {dim} dimensions...")
        for i in range(dim):
            # 해당 차원의 벡터 (예: 모든 점의 x좌표들)
            vector_data = list(transposed_data[i])
            
            # KeyPack에서 Public Key를 가져와 암호화 수행
            # desilofhe의 engine.encrypt는 (data, key)를 인자로 받습니다.
            try:
                ctxt = self.engine.encrypt(vector_data, self.keypack.public_key)
                ciphertexts.append(ctxt)
            except Exception as e:
                print(f"Error encrypting dimension {i}: {e}")
                raise e

        return ciphertexts

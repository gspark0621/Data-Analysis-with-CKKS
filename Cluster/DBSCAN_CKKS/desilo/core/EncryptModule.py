# EncryptModule.py
from desilofhe import Engine
class CKKSEncryptor:
    def __init__(self):
        self.engine = Engine(use_bootstrap=True, mode="gpu")
        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relinearization_key = self.engine.create_relinearization_key(self.secret_key)
        self.conjugation_key = self.engine.create_conjugation_key(self.secret_key)
        self.bootstrap_key = self.engine.create_bootstrap_key(self.secret_key)
        self.rotation_key = self.engine.create_rotation_key(self.secret_key)
        self.slot_count = self.engine.slot_count

    def get_engine(self):
        return self.engine

    def get_secret_key(self):
        return self.secret_key

    def get_public_key(self):
        return self.public_key

    def get_relinearization_key(self):
        return self.relinearization_key
    
    def get_conjugation_key(self):
        return self.conjugation_key

    def get_bootstrap_key(self):
        return self.bootstrap_key
    
    def get_rotation_key(self):
        return self.rotation_key
    
    
from dataclasses import dataclass
import desilofhe

@dataclass
class KeyPack:
    public_key: desilofhe.PublicKey
    rotation_key: desilofhe.RotationKey
    relinearization_key: desilofhe.RelinearizationKey
    conjugation_key: desilofhe.ConjugationKey
    lossy_bootstrap_key: desilofhe.LossyBootstrapKey
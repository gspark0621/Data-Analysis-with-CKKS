# util/keypack.py

from dataclasses import dataclass
import desilofhe


@dataclass
class KeyPack:
    public_key:          desilofhe.PublicKey
    rotation_key:        desilofhe.RotationKey
    relinearization_key: desilofhe.RelinearizationKey
    conjugation_key:     desilofhe.ConjugationKey
    sign_bootstrap_key:  desilofhe.LossyBootstrapKey 
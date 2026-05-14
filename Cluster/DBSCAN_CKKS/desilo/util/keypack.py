# util/keypack.py

from dataclasses import dataclass
import desilofhe


@dataclass
class KeyPack:
    public_key:          desilofhe.PublicKey
    rotation_key:        desilofhe.RotationKey
    relinearization_key: desilofhe.RelinearizationKey
    conjugation_key:     desilofhe.ConjugationKey
    bootstrap_key:       desilofhe.BootstrapKey  # sign_bootstrap_key 제거, bootstrap_key만 사용
    smallbootstrap_key:  desilofhe.SmallBootstrapKey
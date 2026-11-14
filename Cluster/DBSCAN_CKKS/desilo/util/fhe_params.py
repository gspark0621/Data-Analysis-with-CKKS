"""
fhe_params.py -- recover the CKKS parameters (N, log Q, log PQ, secret
distribution) of the DesiloFHE engine used by the protocol, and write them
to fhe_params.json for the security estimate (estimate_security.sage).

DesiloFHE does not expose the moduli through its API (only slot_count and
max_level), so they are read from the serialized keys:
  * serialize_*  ->  [small header][uint64 residues, one row of N per RNS prime]
  * N            =  2 * slot_count
  * secret key   :  the first rows hold s in coefficient form ({0, 1, -1}),
                    which gives the Hamming weight of s
  * relin. key   :  dnum digits x 2 polys x (#primes of PQ) rows; residues
                    mod q_i are uniform, so log2(max over ~14N samples)
                    equals log2(q_i) up to ~1e-4 bits.
  * public key   :  2 polys x (#primes of the top-level modulus Q).
The largest modulus in any published key (PQ of the key-switching keys)
is the one that determines the RLWE security level.

Usage (pyenv env "dbscan"):
    python fhe_params.py              # Engine(use_bootstrap=True), as in the experiments
    python fhe_params.py --compact    # Engine(use_bootstrap=True, compact=True)
"""
import argparse
import json
import math

import numpy as np
import desilofhe


def _rows(buf: bytes, n: int) -> np.ndarray:
    buf = bytes(buf)
    hdr = len(buf) % (8 * n)
    return np.frombuffer(buf[hdr:], dtype=np.uint64).reshape(-1, n)


def _prime_log2(rows: np.ndarray, n_primes: int, n: int) -> list:
    m = rows.reshape(-1, n_primes, n).max(axis=(0, 2))
    return [math.log2(int(x)) for x in m]


def recover(compact: bool) -> dict:
    kw = dict(use_bootstrap=True, mode="cpu")
    if compact:
        kw["compact"] = True
    engine = desilofhe.Engine(**kw)
    N = 2 * engine.slot_count
    sk = engine.create_secret_key()

    # secret key: Hamming weight from the coefficient-form row
    sk_rows = _rows(engine.serialize_secret_key(sk), N)
    coeff_rows = [r for r in sk_rows if len(np.unique(r)) == 3]
    # (the non-compact engine stores s only in NTT form -> weight unknown)
    hw = int((coeff_rows[0] != 0).sum()) if coeff_rows else None

    # relinearization key: modulus PQ of key switching
    rlk_rows = _rows(engine.serialize_relinearization_key(
        engine.create_relinearization_key(sk)), N)
    R = rlk_rows.shape[0]
    n_pq = next(k for k in range(40, 0, -1)
                if R % (2 * k) == 0 and (R // (2 * k)) in range(1, 16)
                and k <= sk_rows.shape[0])
    pq_bits = _prime_log2(rlk_rows, n_pq, N)

    # public key: top-level modulus
    pk_rows = _rows(engine.serialize_public_key(engine.create_public_key(sk)), N)
    n_q = pk_rows.shape[0] // 2
    q_bits = _prime_log2(pk_rows, n_q, N)

    return {
        "engine_kwargs": {k: v for k, v in kw.items() if k != "mode"},
        "N": N,
        "log2_N": int(math.log2(N)),
        "slot_count": engine.slot_count,
        "max_level": engine.max_level,
        "secret": {"distribution": "ternary", "hamming_weight": hw,
                   "hamming_weight_ratio": hw / N if hw else None},
        "pk_modulus": {"prime_count": n_q, "prime_log2": q_bits,
                       "log2": sum(q_bits)},
        "keyswitch_modulus_PQ": {"prime_count": n_pq,
                                 "dnum": R // (2 * n_pq),
                                 "prime_log2": pq_bits,
                                 "log2": sum(pq_bits)},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--out", default="fhe_params.json")
    a = ap.parse_args()
    p = recover(compact=a.compact)
    with open(a.out, "w") as f:
        json.dump(p, f, indent=2)
    print(f"N = 2^{p['log2_N']}  slots = {p['slot_count']}  max_level = {p['max_level']}")
    hw = p["secret"]["hamming_weight"]
    print(f"secret: ternary, h = {hw} ({hw / p['N']:.3f} N)" if hw
          else "secret: ternary, h not readable (NTT form)")
    print(f"log2 Q  (public key, {p['pk_modulus']['prime_count']} primes) "
          f"= {p['pk_modulus']['log2']:.2f}")
    print(f"log2 PQ (key switching, {p['keyswitch_modulus_PQ']['prime_count']} primes, "
          f"dnum = {p['keyswitch_modulus_PQ']['dnum']}) "
          f"= {p['keyswitch_modulus_PQ']['log2']:.2f}")
    print(f"-> {a.out}")

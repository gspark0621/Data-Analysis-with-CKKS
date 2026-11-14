# estimate_security.sage -- RLWE security level lambda of the recovered
# CKKS parameters, with the lattice estimator of Albrecht et al.
#   https://github.com/malb/lattice-estimator   (tested at commit 7d59f86)
#
# Usage:
#   sage estimate_security.sage <path/to/lattice-estimator> [fhe_params.json] [sigma]
# The modulus is the key-switching modulus PQ (largest modulus in any key);
# the secret is ternary (sparse only if fhe_params.py read a weight < N/2);
# the error is a discrete Gaussian with standard deviation sigma (default
# 3.19, the HE-standard value; DesiloFHE does not document its own).
import sys, json
sys.path.insert(0, sys.argv[1])
from estimator import *

path  = sys.argv[2] if len(sys.argv) > 2 else "fhe_params.json"
sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 3.19
p = json.load(open(path))

N, h = p["N"], p["secret"]["hamming_weight"]
logq = p["keyswitch_modulus_PQ"]["log2"]
Xs = (ND.SparseTernary(h // 2, h - h // 2, N) if h and h < N // 2
      else ND.UniformMod(3))
params = LWE.Parameters(n=N, q=2**ceil(logq), Xs=Xs,
                        Xe=ND.DiscreteGaussian(sigma))
print(f"N = {N}, log2 PQ = {logq:.2f}, secret = {Xs}, sigma = {sigma}")
res = {}
for name, f in (("usvp", LWE.primal_usvp), ("bdd", LWE.primal_bdd),
                ("dual_hybrid", LWE.dual_hybrid)):
    for model_name, model in (("BDGL16", RC.BDGL16), ("core-SVP", RC.ADPS16)):
        r = f(params, red_cost_model=model)
        res[(name, model_name)] = float(log(r["rop"], 2))
        print(f"  {name:12s} {model_name:9s} lambda = {res[(name, model_name)]:.1f}")
for m in ("BDGL16", "core-SVP"):
    print(f"-> lambda ({m}) = {min(v for (a, b), v in res.items() if b == m):.1f} bits")

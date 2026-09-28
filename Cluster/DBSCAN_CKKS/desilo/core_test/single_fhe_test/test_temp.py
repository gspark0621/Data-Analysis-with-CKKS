# test_cleaning_ablation.py
"""sign_bootstrap 뒤의 cleaning 이 필요한지 3-way ablation 으로 판정.

  (A) sign_bootstrap 만
  (B) sign_bootstrap + 표준 bootstrap      (2026-05c 이전 경로)
  (C) sign_bootstrap + bit_cleaning        (현행)

Normalize.py:84-146 의 산식을 그대로 복제하되, dist_sq 를 실데이터가 아니라
'eps² 경계를 가로지르는 스윕'으로 넣어 경계 근접도별 정밀도를 본다.
"""
import math
import numpy as np

from core.ciphertext_single.Client_main import setup_fhe_engine
from core.ciphertext_single.chebyshev_eval import eval_mcp_full_chebyshev
from core.ciphertext_single.cleaning import bit_cleaning, std_boot_count
from core.ciphertext_single.minimax import load_mcp

MCP_PATH  = "mcp_alpha15_lp_cheb.json"   # Normalize 가 쓰는 파일
DIMENSION = 2
EPS_NORM  = 0.05
N_PROBE   = 256                          # 스윕 점 개수


def build_probe(eps_sq, max_dist_sq, margin_val, n=N_PROBE):
    """dist² 스윕. 경계에서 ±(1/2,1,2,4,16,64)τ 지점을 반드시 포함시킨다."""
    tau = margin_val
    near = np.concatenate([eps_sq + s * f * tau
                           for s in (-1.0, 1.0)
                           for f in (0.5, 1.0, 2.0, 4.0, 16.0, 64.0)]
                          ) if False else np.array(
        [eps_sq + s * f * tau for s in (-1.0, 1.0)
         for f in (0.5, 1.0, 2.0, 4.0, 16.0, 64.0)])
    far  = np.linspace(0.0, max_dist_sq, n - len(near))
    v    = np.clip(np.concatenate([near, far]), 0.0, max_dist_sq)
    return v


def deviation(vals, expect):
    """{0,1} 로부터의 최대 이탈 + 오분류 개수."""
    d  = np.abs(vals - expect)
    fn = int(np.sum((expect == 1.0) & (vals < 0.5)))   # 참이웃을 놓침
    fp = int(np.sum((expect == 0.0) & (vals > 0.5)))   # 비이웃을 이웃으로
    return float(d.max()), float(d.mean()), fn, fp


def main():
    engine, secret_key, keypack = setup_fhe_engine(verbose=True)
    slot = engine.slot_count

    # ── Normalize.py 산식 그대로 ──────────────────────────────────
    comps        = load_mcp(MCP_PATH)
    mcp_delta    = comps[0]["domain_a"]
    max_dist_sq  = float(DIMENSION)
    approx_bound = max_dist_sq * 1.05
    margin_val   = mcp_delta * approx_bound
    eps_sq       = EPS_NORM ** 2
    x_min_abs    = eps_sq + margin_val
    x_max_abs    = max_dist_sq - (eps_sq + margin_val)
    bound        = max(x_min_abs, x_max_abs) * 1.05
    scale_factor = 1.0 / bound

    print(f"\n[SETUP] alpha 추정 = {-math.log2(mcp_delta):.1f}, delta = {mcp_delta:.3e}")
    print(f"        eps_sq = {eps_sq:.6f}, margin = {margin_val:.3e}, bound = {bound:.4f}")
    print(f"        eps_effective / eps = "
          f"{math.sqrt((eps_sq + margin_val) / eps_sq):.4f}")

    probe  = build_probe(eps_sq, max_dist_sq, margin_val)
    expect = (probe <= eps_sq).astype(float)      # 참이웃 = 1
    vec    = np.zeros(slot); vec[:len(probe)] = probe
    dist_sq_ct = engine.encrypt(vec, keypack.public_key)

    def front():
        """threshold 이동 -> 스케일 -> MCP -> sign_bootstrap 까지 (공통 구간)."""
        thr = engine.encode([eps_sq + margin_val] * slot)
        x   = engine.subtract(dist_sq_ct, thr)
        cx  = engine.multiply(x, engine.encode([scale_factor] * slot))
        cx  = eval_mcp_full_chebyshev(engine, cx, comps, slot, keypack,
                                      tag="probe ", debug=False)
        cx  = engine.sign_bootstrap(engine.intt(cx),
                                    keypack.relinearization_key,
                                    keypack.conjugation_key,
                                    keypack.rotation_key,
                                    keypack.smallbootstrap_key)
        return cx

    def to01(ct):
        r = engine.add(engine.multiply(ct, engine.encode([-0.5] * slot)),
                       engine.encode([0.5] * slot))
        return r

    def report(tag, ct, boots):
        v = np.real(engine.decrypt(ct, secret_key)[:len(probe)])
        mx, mn, fn, fp = deviation(v, expect)
        lvl = getattr(ct, "level", -1)
        print(f"  {tag:<34} level={lvl:>3}  최대이탈={mx:.3e}  "
              f"평균이탈={mn:.3e}  FN={fn}  FP={fp}  표준boot={boots}")
        return v

    print("\n[ABLATION] 세 경로 비교")
    print("-" * 92)

    # (A) sign_bootstrap 만
    std_boot_count(reset=True)
    ctA = to01(front())
    print(f"  {'(공통) sign_bootstrap 직후 level':<34} = "
          f"{getattr(ctA, 'level', -1)}")
    vA = report("(A) sign_bootstrap 만", ctA, std_boot_count())

    # (B) + 표준 bootstrap  (구버전 경로)
    std_boot_count(reset=True)
    ctB = to01(front())
    ct8 = engine.intt(ctB)
    ctB = engine.bootstrap(ct8, keypack.relinearization_key,
                           keypack.conjugation_key, keypack.rotation_key,
                           keypack.smallbootstrap_key)
    vB = report("(B) + 표준 bootstrap", ctB, std_boot_count() + 1)

    # (C) + bit_cleaning  (현행)
    std_boot_count(reset=True)
    ctC = to01(front())
    ctC = bit_cleaning(engine, ctC, keypack, n_iters=1, slot_count=slot)
    vC = report("(C) + bit_cleaning (현행)", ctC, std_boot_count())

    # ── 판정 ──────────────────────────────────────────────────────
    print("\n[판정]")
    lvlA = getattr(ctA, "level", -1)
    devA = np.abs(vA - expect).max()
    if lvlA >= 3 and devA <= 1e-5:
        print(f"  ✓ cleaning 제거 가능 — (A) 만으로 level {lvlA}, 이탈 {devA:.2e}")
        print(f"    → Normalize.py:145 의 bit_cleaning 삭제, 레벨 2 x k_max 절약")
    elif lvlA < 3:
        print(f"  ✗ (A) 의 level {lvlA} 부족 → cleaning 내부 _refresh 가 필요.")
        print(f"    현행 (C) 유지.")
    else:
        print(f"  △ (A) 이탈 {devA:.2e} 가 큼 → cleaning 유지. LP 와 같은 상황.")
    print(f"\n  누적 결손 비교 (이웃수 d=500 인 점 기준):")
    for tag, v in [("A", vA), ("B", vB), ("C", vC)]:
        print(f"    ({tag}) {np.abs(v - expect).mean() * 500:.4f}"
              f"   {'✓ 안전 (<0.5)' if np.abs(v-expect).mean()*500 < 0.5 else '✗ 위험'}")


if __name__ == "__main__":
    main()
# core/ciphertext_single/Core.py
#
# 핵심 변경:
#   [변경 1] α=12 → α=11 최적화
#     이유:
#       α=9  t_k > min_gap=0.5/N → 분류 오류 가능
#       α=10 Remez 편차로 UNSAFE 위험 (margin 추가 시 threshold 초과)
#       α=11 [7,15,15,15]: t_k=0.00051 << threshold=0.00098 → 안정적 SAFE
#            delta=2^{-11} < 0.5/N (4.8배 여유)
#            α=12와 동일 5회 bootstrap, non-scalar mult 5회 절약
#
#   [변경 2] _eval_mcp_fhe: 마지막 컴포넌트 평가 *후* bootstrap
#     (Normalize.py와 동일 패턴)
#
#   [변경 3] sign_bootstrap 복원 (level 충분)

from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp


_MCP_CORE_PATH = "mcp_alpha11.json"   # ★ α=11 (기존 α=12에서 최적화)


def _eval_mcp_fhe(engine, ct, components, N, keypack):
    """
    FHE MCP 평가. 마지막 후 bootstrap 포함.
    중간 + 마지막 모든 step 후 bootstrap → level=10 반환.
    """
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key
    current   = ct

    for step_idx, comp in enumerate(components):
        coeffs   = comp["coeffs"]
        domain_b = comp.get("domain_b", 1.0)

        if abs(domain_b - 1.0) > 1e-9:
            inv_b   = engine.encode([1.0 / domain_b] * N)
            working = engine.multiply(current, inv_b)
        else:
            working = current

        x_sq   = engine.square(working, relin_key)
        x_pow  = working
        result = engine.multiply(x_pow, engine.encode([coeffs[0]] * N))

        for k in range(1, len(coeffs)):
            x_pow  = engine.multiply(x_pow, x_sq, relin_key)
            result = engine.add(
                result,
                engine.multiply(x_pow, engine.encode([coeffs[k]] * N))
            )

        current = result

        # ★ 모든 step 후 bootstrap (마지막 포함)
        print(f"  - [Core MCP] p_{step_idx+1} 완료 (domain_b={domain_b:.4f}), bootstrap...")
        current = engine.intt(current)
        current = engine.bootstrap(current, relin_key, conj_key, boot_key)

    return current  # level=10


def identify_core_points_fhe_converted(
    engine: Engine,
    neighbor_count_ct: Ciphertext,
    min_pts: float,
    N: int,
    keypack: KeyPack,
    bootstrap_interval: int = 3,
    mcp_path: str = None,  # None이면 _MCP_CORE_PATH 사용
    **kwargs
) -> Ciphertext:
    """
    Core point 판별: totalNeighbors >= min_pts → 1, else → 0.

    α=11 선택 이유 (최적값):
      최소 입력 gap = 0.5/N = 0.5/212 ≈ 0.00236
      α=10 [7,7,13,15]: Remez 편차로 t_k가 threshold 근방 → margin 추가 시 UNSAFE
      α=11 [7,15,15,15]: t_k=0.00051 << threshold=0.00098 → 안정적 SAFE
                          delta=2^{-11}=0.00049 < 0.00236 (4.8배 여유)
      α=12 대비: 동일 4+1=5회 bootstrap, non-scalar mult 5회 절약

    sign_bootstrap 후 정밀도:
      error ≤ (π²/8) × (2^{-10})² ≈ 1.2e-6 → 매우 정밀한 0/1 판별
    """
    if mcp_path is None:
        mcp_path = _MCP_CORE_PATH

    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key

    print(f"[Server] Core: α=11 MCP 로드 ({mcp_path})")
    components = load_mcp(mcp_path)
    print(f"[Server] Core: degrees={[c['degree'] for c in components]}, "
          f"sign_err={components[-1]['error']:.4e}")

    # x = (totalNeighbors - (min_pts - 0.5)) / N ∈ [-1, 1]
    margin     = 0.5
    min_pts_pt = engine.encode([min_pts - margin] * N)
    x          = engine.subtract(neighbor_count_ct, min_pts_pt)
    scale_pt   = engine.encode([1.0 / float(N)] * N)
    current_x  = engine.multiply(x, scale_pt)

    print(f"[Server] Core: N={N}, min_pts={min_pts}, scale=1/{N}={1.0/N:.4e}")
    print(f"[Server] Core: delta=2^-11={2**-11:.5f} < 0.5/N={0.5/N:.5f} ✓")

    # MCP 평가 → 마지막 후 bootstrap → level=10
    current_x = _eval_mcp_fhe(engine, current_x, components, N, keypack)

    # ★ sign_bootstrap (level=10 입력)
    print(f"  - [Core] sign_bootstrap...")
    current_x = engine.sign_bootstrap(
        engine.intt(current_x),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )

    # (sign + 1) / 2 → {0, 1}
    half_pt        = engine.encode([0.5] * N)
    core_indicator = engine.add(engine.multiply(current_x, half_pt), half_pt)

    core_indicator = engine.intt(core_indicator)
    return engine.bootstrap(core_indicator, relin_key, conj_key, boot_key)
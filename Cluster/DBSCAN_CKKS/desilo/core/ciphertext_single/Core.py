# core/ciphertext_single/Core.py

from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp


def _eval_mcp_fhe(engine, ct, components, N, keypack):
    """
    FHE 상에서 Minimax Composite Polynomial 평가:
        p(x) = p_k ∘ ... ∘ p_1(x)

    각 p_i: c[0]·x + c[1]·x³ + c[2]·x⁵ + ...  (홀수 다항식)
    중간 단계마다 bootstrap 수행 (마지막 제외).
    """
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key
    current   = ct

    for step_idx, comp in enumerate(components):
        coeffs = comp["coeffs"]   # [c0, c1, ...] → x^1, x^3, x^5, ...

        x_sq   = engine.square(current, relin_key)
        x_pow  = current                                         # x^1
        result = engine.multiply(
            x_pow, engine.encode([coeffs[0]] * N)
        )                                                        # c[0]·x

        for k in range(1, len(coeffs)):
            x_pow  = engine.multiply(x_pow, x_sq, relin_key)    # x^(2k+1)
            result = engine.add(
                result,
                engine.multiply(x_pow, engine.encode([coeffs[k]] * N))
            )

        current = result

        # 마지막 컴포넌트 제외하고 중간 bootstrap
        if step_idx < len(components) - 1:
            print(f"  - [Core MCP] p_{step_idx+1} 완료, 중간 부트스트래핑...")
            current = engine.intt(current)
            current = engine.bootstrap(current, relin_key, conj_key, boot_key)

    return current


def identify_core_points_fhe_converted(
    engine            : Engine,
    neighbor_count_ct : Ciphertext,
    min_pts           : float,
    N                 : int,
    keypack           : KeyPack,
    bootstrap_interval: int = 3,    # 하위 호환성 유지 (내부 미사용)
    mcp_path          : str = "mcp_alpha12.json",
    **kwargs
) -> Ciphertext:
    """
    Core point 판별: totalNeighbors >= min_pts 이면 1, 아니면 0.

    α=12 MCP (degrees=[3,5,5,5,5,5,9]) 사용:
      - sign_err ≈ 2.4e-4
      - N=134: 누적 오차 = 134 × 2.4e-4 = 0.032  → 안전
      - N=2048: 누적 오차 = 2048 × 2.4e-4 = 0.49 → 안전

    사전 준비:
      test_Server_main_call_dataset.py 실행 시 자동으로
      mcp_alpha12.json 을 생성하므로 별도 작업 불필요.
    """
    relin_key = keypack.relinearization_key
    conj_key  = keypack.conjugation_key
    boot_key  = keypack.bootstrap_key

    # ── 1. α=12 MCP 계수 로드 ────────────────────────────
    print(f"[Server] Core: α=12 MCP 계수 로드 ({mcp_path})")
    components = load_mcp(mcp_path)
    print(f"[Server] Core: degrees={[c['degree'] for c in components]}, "
          f"sign_err={components[-1]['error']:.4e}")

    # ── 2. 입력 준비: x = (totalNeighbors - (min_pts - 0.5)) / N ──
    # margin=0.5: totalNeighbors 는 정수값 → 경계 0.5 버퍼
    # /N: sign 입력을 [-1, 1] 로 스케일
    margin         = 0.5
    min_pts_pt     = engine.encode([min_pts - margin] * N)
    x              = engine.subtract(neighbor_count_ct, min_pts_pt)
    scale_pt       = engine.encode([1.0 / float(N)] * N)
    current_x      = engine.multiply(x, scale_pt)

    print(f"[Server] Core: N={N}, min_pts={min_pts}, margin={margin}, "
          f"scale=1/{N}={1.0/N:.4e}")

    # ── 3. α=12 MCP 로 sign 근사 ─────────────────────────
    current_x = _eval_mcp_fhe(engine, current_x, components, N, keypack)

    # ── 4. sign → {0,1}: (sign + 1) / 2 ─────────────────
    half_pt        = engine.encode([0.5] * N)
    core_indicator = engine.add(engine.multiply(current_x, half_pt), half_pt)

    core_indicator = engine.intt(core_indicator)
    return engine.bootstrap(core_indicator, relin_key, conj_key, boot_key)
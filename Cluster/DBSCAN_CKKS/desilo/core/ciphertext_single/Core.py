# core/ciphertext_single/Core.py
#
# 변경: _eval_mcp_fhe → bsgs_poly.eval_mcp_full (BSGS 기반)
#   naive 루프: deg=15에서 8레벨 소비, deg=27이면 14레벨 → budget=10 초과
#   BSGS:       dep(d)×2 레벨 소비 → dep(15)=4 → 8레벨, dep(27)=5 → 10레벨
#   α=11 degrees=[7,15,15,15]: dep 최대=4 → 8레벨 소비 → budget 내 ✓
#   (Core는 α=11 사용 → deg≤15 → naive도 작동하지만 BSGS로 통일)

from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack
from core.ciphertext_single.minimax import load_mcp
from core.ciphertext_single.bsgs_poly import eval_mcp_full   # ★ BSGS 공용 모듈


_MCP_CORE_PATH = "mcp_alpha11.json"   # α=11: dep(15)=4 → 8레벨 ✓


def identify_core_points_fhe_converted(
    engine: Engine,
    neighbor_count_ct: Ciphertext,
    min_pts: float,
    N: int,
    keypack: KeyPack,
    bootstrap_interval: int = 3,
    mcp_path: str = None,
    **kwargs
) -> Ciphertext:
    """
    Core point 판별: totalNeighbors >= min_pts → 1, else → 0.

    α=11 선택 이유 (N=212):
      최소 입력 gap = 0.5/N ≈ 0.00236
      t_k=0.00051 << threshold=0.00098 → SAFE ✓  delta=2^{-11}=0.00049 (4.8배 여유)

    Pipeline:
      1. x = (totalNeighbors - (min_pts-0.5)) / N  → x ∈ [-1,1] 정규화
      2. MCP 평가 (BSGS): sign 근사 (각 컴포넌트 후 bootstrap)
      3. sign_bootstrap: level=10 입력 → level≈13 출력
      4. (sign+1)/2 → {0,1} 변환
      5. 최종 bootstrap
    """
    if mcp_path is None:
        mcp_path = _MCP_CORE_PATH

    relin_key  = keypack.relinearization_key
    conj_key   = keypack.conjugation_key
    boot_key   = keypack.bootstrap_key
    slot_count = engine.slot_count

    print(f"[Server] Core: BSGS MCP 로드 ({mcp_path})")
    components = load_mcp(mcp_path)
    print(f"[Server] Core: degrees={[c['degree'] for c in components]}, "
          f"sign_err={components[-1]['error']:.4e}")

    # x = (totalNeighbors - (min_pts - 0.5)) / N ∈ [-1, 1]
    margin     = 0.5
    min_pts_pt = engine.encode([min_pts - margin] * slot_count)
    x          = engine.subtract(neighbor_count_ct, min_pts_pt)
    scale_pt   = engine.encode([1.0 / float(N)] * slot_count)
    current_x  = engine.multiply(x, scale_pt)

    print(f"[Server] Core: N={N}, min_pts={min_pts}, scale=1/{N}={1.0/N:.4e}")
    print(f"[Server] Core: delta=2^-11={2**-11:.5f} < 0.5/N={0.5/N:.5f} ✓")

    # MCP 평가 (BSGS) → 각 컴포넌트 후 bootstrap → level=10 반환
    current_x = eval_mcp_full(engine, current_x, components, slot_count, keypack, tag="Core ")

    # sign_bootstrap (level=10 입력 → level≈13 출력)
    print(f"  - [Core] sign_bootstrap...")
    current_x = engine.sign_bootstrap(
        engine.intt(current_x),
        keypack.relinearization_key,
        keypack.conjugation_key,
        keypack.rotation_key,
        keypack.smallbootstrap_key,
    )

    # (sign + 1) / 2 → {0, 1}
    half_pt        = engine.encode([0.5] * slot_count)
    core_indicator = engine.add(engine.multiply(current_x, half_pt), half_pt)

    core_indicator = engine.intt(core_indicator)
    return engine.bootstrap(core_indicator, relin_key, conj_key, boot_key)
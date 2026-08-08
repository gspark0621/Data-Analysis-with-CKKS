# diag_precision.py
#
# 목적: ÷N 패치(라벨 (0,1])가 aliasing 은 없애지만, gap 이 1/N 로 좁아져
#       bootstrap 잔여 드리프트에 인접 라벨이 뭉개지지 않는지 **실측**으로 확정.
#
# 앞선 diag_decay.py 는 32768슬롯을 '가득' 채워 드리프트를 ~6배 과장했음.
# 여기선 실제 파이프라인처럼:
#   · 희소  : N=800 슬롯만 채움 (라운드 사이 상태)
#   · 패킹  : ~20800 슬롯 채움  (트리-max 중, n_region≈26)
# 그리고 상수가 아니라 '램프((i+1)·scale)'를 넣어 **인접 라벨 구분/단조성**을 직접 본다.
#
# 판정 (L=1 케이스):
#   · 인접 gap 최소값이 계속 +  &  단조유지 ≈ 전체 → ÷N 패치 안전 (precision OK)
#   · 인접 gap 이 0/음수로 무너지면              → ÷N 만으론 부족 (중간 스케일 검토)
#   · L=N 케이스가 크게 aliasing 되면            → bootstrap 한계 = 버그 재확인
#
# 실행:  python diag_precision.py    (프로젝트 루트, FHE 환경)

import numpy as np
from core.ciphertext_single.Client_main import setup_fhe_engine
from core.ciphertext_single.Label_Propagation import (
    _refresh, _scaled_refresh, _optimal_refresh_scale,
)

engine, sk, kp = setup_fhe_engine(verbose=False)
sc = engine.slot_count
N  = 800
REFRESHES = 40   # 한 라벨 lineage 가 실제로 겪는 refresh 수 대략 (라운드×수회)

def enc_ramp(scale, fill, period=None):
    # 값 = ((i % period)+1)*scale,  나머지 슬롯 0.  period=None 이면 전체 램프.
    p = period or fill
    vals = [((i % p) + 1) * scale for i in range(fill)] + [0.0] * (sc - fill)
    return engine.encrypt(engine.encode(vals), sk)

def dec(ct, fill):
    return np.real(engine.decrypt(ct, sk))[:fill]

cases = [
    # (scale,   fill,   period, S,                          tag)
    (1.0,       N,      N,      _optimal_refresh_scale(N, 1), "L=N  희소(800)  램프  S=opt(현행)"),
    (1.0 / N,   N,      N,      1,                            "L=1  희소(800)  램프  S=1 (÷N 패치)"),
    (1.0 / N,   20800,  N,      1,                            "L=1  패킹(20800) 램프  S=1 (트리-max 상황)"),
]

for scale, fill, period, S, tag in cases:
    ct   = enc_ramp(scale, fill, period)
    true = np.array([((i % period) + 1) * scale for i in range(fill)])
    print(f"\n[{tag}]")
    print(f"    참 gap={scale:.3e}  max={true.max():.4f}  (bootstrap 안전범위 |m|≲1 대비 "
          f"{'초과!' if true.max() > 1.5 else 'OK'})")
    for it in range(1, REFRESHES + 1):
        ct = _scaled_refresh(engine, ct, kp, S) if S > 1 else _refresh(engine, ct, kp)
        if it in (1, 5, 10, 20, 40):
            v = dec(ct, fill)
            drift = np.abs(v - true).max()
            # region 0 (첫 period 슬롯) 안에서 인접 라벨 gap/단조성 (경계 제외)
            seg   = v[:period]
            d     = np.diff(seg)
            print(f"    it{it:2d}: max드리프트={drift:.3e}  "
                  f"region0 인접gap[min={d.min():+.3e}={d.min()/scale:+.2f}x참값]  "
                  f"단조유지 {int((d > 0).sum())}/{period-1}")

print("\n" + "=" * 72)
print("  해석:")
print("   · L=1 두 케이스에서 region0 인접gap 최소가 계속 + & 단조 거의 유지")
print("       → ÷N 패치 precision 안전. 3.4h full 런 진행 OK.")
print("   · 반대로 gap 이 0/음수로 무너지면 → ÷N 만으론 부족,")
print("       라벨을 (0,1] 대신 (0,c] (c>1, 단 aliasing 임계 아래)로 올리는 중간 스케일 필요.")
print("   · L=N 케이스가 심하게 aliasing → bootstrap 한계 = 근본원인 재확인.")
print("=" * 72)
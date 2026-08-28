# soft_tree/test/bench_boot.py
#
# bootstrap 옵션 전수 비교.  fhe.desilo.dev 문서 확인 후 작성.
#
# [문서 사실]
#   bootstrap(...)              → 출력 레벨 10 (기본).  full key 는 "더 빠르고 메모리 더 씀".
#   bootstrap + use_bootstrap_to_14_levels → 출력 레벨 14, stage_count 는 3만 지원.
#   lossy_bootstrap(...)        → 출력 레벨 = 16 - stage_count  (stage3→13, 4→12, 5→11)
#                                 입력 레벨 >= stage_count 필요.  14레벨 파라미터와 병용 불가.
#   stage_count ↑ → 부트스트랩은 빨라지지만 결과 레벨은 낮아짐 (3~5 지원)
#   모든 bootstrap: 입력 암호문의 "값" 이 [-1, 1] 이내여야 함.
#
# [측정 지표]
#   회복 레벨 L, 시간 T, 정밀도 오차 e, 그리고 ms/level = T / L
#   최종 판단은 n_refresh = n_chains * ceil(D_step / L) 로 환산한 예상 벽시계.
#
# 실행:  python -m Soft_Tree.test.bench_boot

from __future__ import annotations
import gc, math, traceback
from time import perf_counter
import numpy as np
from desilofhe import Engine

REPS = 2
D_STEPS = [17, 15, 13]     # 현재 / 상수접기 / +평문뺄셈 마스킹
N_CHAINS = 10              # depth=2: 파라미터 5 + velocity 5
EPOCHS = 20                # iris 실측 조건


def med(fn, reps=REPS):
    fn()
    ts = []
    for _ in range(reps):
        t0 = perf_counter(); fn(); ts.append(perf_counter() - t0)
    return float(np.median(ts) * 1000.0)


def make_engine(use14: bool):
    kw = dict(use_bootstrap=True)
    if use14:
        kw["use_bootstrap_to_14_levels"] = True
    t0 = perf_counter()
    eng = Engine(**kw)
    sk = eng.create_secret_key()
    K = dict(sk=sk, pk=eng.create_public_key(sk),
             rlk=eng.create_relinearization_key(sk),
             ck=eng.create_conjugation_key(sk),
             rot=eng.create_rotation_key(sk),
             sbk=eng.create_small_bootstrap_key(sk))
    print(f"  [엔진{' +14levels' if use14 else ''}] max_level={eng.max_level}, "
          f"slot_count={eng.slot_count}, 키생성 {perf_counter()-t0:.1f}s")
    return eng, K


def drain_to(eng, K, target_level, amp=0.5):
    """값을 [-amp,amp] 로 유지하면서 레벨만 target 까지 낮춘다 (bootstrap 입력 제약)."""
    sc = eng.slot_count
    vec = np.random.default_rng(0).uniform(-amp, amp, sc)
    one = eng.encode([1.0] * sc)
    c = eng.encrypt(vec.tolist(), K["pk"])
    while c.level > target_level:
        c = eng.multiply(c, one)          # 값 보존, 레벨만 -1
    return c, vec


def measure(eng, K, name, call, in_level, amp=0.5):
    try:
        c, vec = drain_to(eng, K, in_level, amp)
        lv_in = c.level
        f = lambda: call(c)
        t = med(f)
        out = f()
        d = np.real(eng.decrypt(out, K["sk"]))[:4000]
        err = float(np.max(np.abs(d - vec[:4000])))
        L = out.level - lv_in
        per = t / L if L > 0 else float("inf")
        print(f"  {name:<42} in={lv_in:>2} out={out.level:>2} (+{L:>2})"
              f" {t:>8.1f} ms  {per:>7.1f} ms/lv  오차 {err:.2e}")
        r = dict(name=name, t=t, out=out.level, L=L, per=per, err=err)
        del c, out; gc.collect()
        return r
    except Exception as e:
        print(f"  {name:<42} 실패 — {type(e).__name__}: {str(e)[:60]}")
        return None


def main():
    results = []

    # ══ A. 기본 엔진 ══
    print("\n" + "=" * 100)
    print("A. 기본 엔진 (use_bootstrap=True)")
    print("=" * 100)
    eng, K = make_engine(False)

    # A-1. 현재 방식: small key bootstrap, stage 3/4/5
    for st in [3, 4, 5]:
        r = measure(eng, K, f"bootstrap(small key, stage={st})",
                    lambda c, s=st: eng.bootstrap(c, K["rlk"], K["ck"], K["rot"], K["sbk"], s),
                    in_level=1)
        if r: results.append(r)

    # A-2. full bootstrap key
    try:
        t0 = perf_counter()
        K["bk"] = eng.create_bootstrap_key(K["sk"], stage_count=3)
        print(f"  [full bootstrap key 생성 {perf_counter()-t0:.1f}s, "
              f"{getattr(K['bk'],'nbytes',0)/2**20:.0f} MB]")
        r = measure(eng, K, "bootstrap(FULL key)",
                    lambda c: eng.bootstrap(c, K["rlk"], K["ck"], K["bk"]), in_level=1)
        if r: results.append(r)
    except Exception as e:
        print(f"  full bootstrap key: 실패 — {type(e).__name__}: {str(e)[:70]}")

    # A-3. ★ lossy_bootstrap — 출력 레벨 16-stage
    for st in [3, 4, 5]:
        # small key 경로 (입력 레벨 >= stage_count 필요)
        r = measure(eng, K, f"lossy_bootstrap(small key, stage={st})",
                    lambda c, s=st: eng.lossy_bootstrap(c, K["rlk"], K["ck"], K["rot"], K["sbk"], s),
                    in_level=st)
        if r: results.append(r)
    try:
        t0 = perf_counter()
        lbk = eng.create_lossy_bootstrap_key(K["sk"], stage_count=3)
        print(f"  [lossy bootstrap key 생성 {perf_counter()-t0:.1f}s, "
              f"{getattr(lbk,'nbytes',0)/2**20:.0f} MB]")
        r = measure(eng, K, "lossy_bootstrap(LOSSY key, stage=3)",
                    lambda c: eng.lossy_bootstrap(c, K["rlk"], K["ck"], lbk), in_level=3)
        if r: results.append(r)
    except Exception as e:
        print(f"  lossy bootstrap key: 실패 — {type(e).__name__}: {str(e)[:70]}")

    del eng, K; gc.collect()

    # ══ B. 14레벨 엔진 (lossy 와 병용 불가) ══
    print("\n" + "=" * 100)
    print("B. use_bootstrap_to_14_levels=True  (stage_count=3 만, lossy 불가)")
    print("=" * 100)
    try:
        eng, K = make_engine(True)
        r = measure(eng, K, "bootstrap(small key, 14levels)",
                    lambda c: eng.bootstrap(c, K["rlk"], K["ck"], K["rot"], K["sbk"], 3),
                    in_level=1)
        if r: results.append(r)
        try:
            K["bk"] = eng.create_bootstrap_key(K["sk"], stage_count=3)
            r = measure(eng, K, "bootstrap(FULL key, 14levels)",
                        lambda c: eng.bootstrap(c, K["rlk"], K["ck"], K["bk"]), in_level=1)
            if r: results.append(r)
        except Exception as e:
            print(f"  full key(14levels): 실패 — {str(e)[:70]}")
        del eng, K; gc.collect()
    except Exception as e:
        print(f"  14레벨 엔진 생성 실패 — {type(e).__name__}: {str(e)[:80]}")

    # ══ 요약 ══
    print("\n" + "=" * 100)
    print("요약 — 스텝당 refresh 횟수와 예상 벽시계 (iris depth2 K1 ep20, 실측 기준선 1152.6 s)")
    print("=" * 100)
    print(f"{'방법':<42}{'회복':>6}{'ms/lv':>8}{'오차':>10}"
          + "".join(f"{'D='+str(d):>10}" for d in D_STEPS))
    results.sort(key=lambda r: r["per"])
    for r in results:
        row = ""
        for D in D_STEPS:
            n = N_CHAINS * math.ceil(D / max(r["L"], 1))
            row += f"{n*r['t']/1000.0*EPOCHS:>10.0f}"
        print(f"{r['name']:<42}{r['L']:>6}{r['per']:>8.1f}{r['err']:>10.1e}{row}")
    print("\n  D=17 현재 / D=15 상수접기 / D=13 +마스크를 평문뺄셈으로")
    print("  ★ D <= 회복레벨 이면 체인당 refresh 2회 → 1회 (약 2배)")
    print("  ★ 정밀도: 현재 학습 최대오차 7.7e-5, 상관계수 0.999999.")
    print("            lossy 오차가 1e-3 이하면 AUC 에 영향 없을 가능성 높음 (검증 필요)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
# Hard_Tree/desilo/opcount.py
"""
FHE 연산 카운터.

  용도 ①  병목 진단 — 어느 단계가 회전/곱셈을 얼마나 쓰는지
  용도 ②  하드웨어 정규화 — 논문에서 시간 대신 연산 수로 비교

  engine 을 감싸기만 하므로 코어 코드를 건드리지 않는다.

사용:
    from Hard_Tree.desilo.opcount import CountingEngine
    engine = CountingEngine(engine)
    ...
    engine.report()                      # 전체
    with engine.section("T1"):           # 구간별
        ...
    engine.report_sections()
"""
from __future__ import annotations
from contextlib import contextmanager
from collections import defaultdict


class CountingEngine:
    """engine 위임 래퍼.  세어야 할 메서드만 가로챈다."""

    _WATCH = ("rotate", "multiply", "add", "subtract", "encode", "encrypt",
              "decrypt", "intt", "ntt", "bootstrap", "sign_bootstrap",
              "evaluate_chebyshev_polynomial")

    def __init__(self, engine):
        object.__setattr__(self, "_e", engine)
        object.__setattr__(self, "n", defaultdict(int))
        object.__setattr__(self, "_sec", None)
        object.__setattr__(self, "sec", defaultdict(lambda: defaultdict(int)))
        object.__setattr__(self, "shifts", defaultdict(int))   # 분해 안 된 shift

    # ── 위임 ──
    def __getattr__(self, name):
        target = getattr(self._e, name)
        if name not in self._WATCH or not callable(target):
            return target

        def wrapped(*a, **kw):
            key = name
            if name == "rotate" and len(a) >= 3:
                # 2^k shift 인지 임의 shift 인지 구분 — 비용이 5.8배 다르다
                s = abs(int(a[2]))
                key = "rot_p2" if (s & (s - 1)) == 0 else "rot_any"
                if key == "rot_any":
                    self.shifts[a[2]] += 1
            elif name == "multiply":
                key = "mul_ct" if len(a) >= 3 else "mul_pt"
            self.n[key] += 1
            if self._sec is not None:
                self.sec[self._sec][key] += 1
            return target(*a, **kw)

        return wrapped

    @property
    def slot_count(self):
        return self._e.slot_count

    # ── 구간 ──
    @contextmanager
    def section(self, name):
        prev = self._sec
        object.__setattr__(self, "_sec", name)
        try:
            yield
        finally:
            object.__setattr__(self, "_sec", prev)

    def reset(self):
        self.n.clear()
        self.sec.clear()
        self.shifts.clear()

    def report_shifts(self, top=10):
        """분해되지 않은 임의 shift 목록 — 최적화 대상 찾기용."""
        if not self.shifts:
            return
        from Hard_Tree.desilo.hard_core import _naf
        print(f"\n  분해 안 된 임의 shift (NAF weight > 임계값)")
        print(f"    {'shift':>10}{'횟수':>7}{'NAF':>6}{'이진':>6}")
        for sh, cnt in sorted(self.shifts.items(), key=lambda x: -x[1])[:top]:
            a = abs(sh)
            print(f"    {sh:>10}{cnt:>7}{len(_naf(a)):>6}{bin(a).count('1'):>6}")

    # ── 보고 ──
    ORDER = ("rot_p2", "rot_any", "mul_ct", "mul_pt", "add", "subtract",
             "encode", "encrypt", "decrypt", "intt",
             "bootstrap", "sign_bootstrap", "evaluate_chebyshev_polynomial")
    SHORT = {"evaluate_chebyshev_polynomial": "cheb", "sign_bootstrap": "signboot",
             "subtract": "sub", "encrypt": "enc", "decrypt": "dec",
             "bootstrap": "boot"}

    # ── 단가 (ms) ──  test_unit_cost.py 로 머신마다 재서 갱신할 것.
    #   ★ 실측 (A6000 / DesiloFHE, level 14)
    #     ct·ct 곱(9.84)이 회전(6.24)보다 비싸다 — 추측값 1.20 은 8배 틀렸다.
    #     decrypt 105ms — 평문 argmin 이 생각보다 비싸다.
    UNIT = dict(rot_p2=6.242, rot_any=33.918, mul_ct=9.844, mul_pt=3.928,
                add=0.413, sub=0.412, encode=2.346, encrypt=5.197,
                decrypt=105.562, intt=1.225,
                bootstrap=2814.4, sign_bootstrap=1256.5,
                evaluate_chebyshev_polynomial=0.0)

    def est_ms(self, counts=None, **override):
        u = dict(self.UNIT)
        u.update(override)
        c = counts if counts is not None else self.n
        g = c.get if hasattr(c, "get") else (lambda k, d=0: c[k])
        return sum(g(k, 0) * v for k, v in u.items())

    def report(self, title="연산 수"):
        print(f"\n  {title}")
        print("  " + "-" * 52)
        for k in self.ORDER:
            if self.n.get(k, 0):
                print(f"    {k:<32}{self.n[k]:>12,}")
        print(f"    {'추정 시간 (ms)':<32}{self.est_ms():>12,.0f}")

    def report_sections(self):
        if not self.sec:
            return
        keys = [k for k in self.ORDER if any(v.get(k, 0) for v in self.sec.values())]
        print(f"\n  구간별")
        print("  " + "-" * (24 + 10 * len(keys) + 12))
        print(f"    {'구간':<20}"
              + "".join(f"{self.SHORT.get(k, k):>10}" for k in keys) + f"{'추정 ms':>12}")
        for name, v in self.sec.items():
            print(f"    {name:<20}" + "".join(f"{v.get(k, 0):>10,}" for k in keys)
                  + f"{self.est_ms(v):>12,.0f}")
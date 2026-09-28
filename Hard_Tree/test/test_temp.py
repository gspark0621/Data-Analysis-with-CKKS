# Hard_Tree/test/test_mem_probe.py
"""
GPU 메모리 실측 — n 을 올려가며 **어디서 터지는지** 찾는다.

  ★ 추정으로는 n=4096 도 6.3 GB (24 GB 여유) 지만, S9 에서 이미 추정보다
    훨씬 일찍 OOM 이 났다.  키·bootstrap 작업공간·단편화가 추정에 안 들어간다.
    → 작은 n 부터 올려가며 **실제로** 확인한다.

  각 n 마다
    ① 마스크 생성 → ② 트리 1 개 학습 → ③ 부스팅 1 라운드
  를 돌리고, 단계별 nvidia-smi 사용량을 찍는다.

  OOM 이 나면 그 지점을 보고하고 다음 n 은 건너뛴다.

실행:  python -m Hard_Tree.test.test_mem_probe
"""
from __future__ import annotations
import gc
import subprocess
import time
import numpy as np

from Hard_Tree.desilo.util.keypack import setup_fhe_engine
from Hard_Tree.desilo import train_core as TC
from Hard_Tree.desilo import hard_core as HC
from Hard_Tree.desilo import minimax_sign as MS

HC.SIGN_IMPL = "minimax"

# telco/bank/adult 모양.  n 을 올려가며 한계를 찾는다.
D_FEAT, N_MAX, DEPTH = 19, 8, 4
N_LIST = [256, 512, 1024, 2048, 4096]
MODE = "gpu"        # "gpu" | "cpu".  CPU 는 메모리 여유가 크지만 느리다
                    #   → test_cpu_vs_gpu.py 로 배율을 먼저 잴 것
ALPHA = TC.ARGMAX_ALPHA


def gpu_mb():
    """
    현재 사용량(MB).  MODE="cpu" 면 프로세스 RSS, 아니면 nvidia-smi.
    """
    if MODE == "cpu":
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) // 1024
            with open("/proc/meminfo") as f:
                tot = int(f.readline().split()[1]) // 1024
            return rss, tot
        except Exception:
            return None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0]
        used, total = (int(x) for x in out.split(","))
        return used, total
    except Exception:
        return None, None


def probe(engine, sk, kp, n, d, n_max, D):
    rng = np.random.default_rng(0)
    Xb = rng.integers(0, n_max, (n, d))
    y = rng.integers(0, 2, n).astype(float)
    td = TC.train_dims(n, d, n_max, 2, engine.slot_count)
    dp = HC.p2(d)
    marks = []

    def mark(tag):
        gc.collect()
        u, t = gpu_mb()
        marks.append((tag, u))
        return u

    mark("시작")
    tm = TC.make_train_masks(engine, td)
    am = TC.make_argmax_masks(engine, td, TC.auto_mtry(engine, td))
    leaf_oh = TC.make_leaf_onehots(engine, td, dp, D)
    spec = MS.build_composite(ALPHA, 2.0 ** (-ALPHA), "depth")
    mark("마스크")

    tiles = TC.encode_onehot(engine, kp, Xb, td)
    pre = TC.prefix_cumsum(engine, tiles, kp, td, tm)
    del tiles
    gv = TC.encode_sample_all(engine, kp, y - y.mean(), td)
    mark("prefix")

    t0 = time.time()
    sel2 = TC.encode_boost_sel_ct(engine, gv, kp, td)
    masks = [TC.encode_sample_all(engine, kp, np.ones(n), td)]
    Lq = [TC.root_stats(engine, pre, masks[0], sel2, kp, td, tm)]
    Nq = [TC.broadcast_all(engine, Lq[0], kp, td, tm)]
    for lv in range(D):
        score = TC.level_score(engine, Lq, Nq, kp, td, criterion="boost",
                               level=lv, normalize=True)
        sc = TC.compact_scores(engine, score, am, kp, td)
        ind = TC.argmax_league(engine, sc, am, kp, spec, maximize=True, secret_key=sk)
        if lv == 0:
            mark("argmax 1회")
        iv = np.real(engine.decrypt(ind, sk))
        j, k = am["cands"][int(np.argmax([iv[i * am["nsp"]] for i in am["idx"]]))]
        bvec = TC.extract_branch(engine, pre, j, k, kp, td)
        masks, Lq, Nq = TC.next_level(engine, pre, masks, Lq, Nq, bvec, sel2,
                                      kp, td, tm, with_stats=(lv < D - 1))
        del score, sc, ind, bvec
    G, H = TC.leaf_stats(engine, masks, gv, kp, td, dp)
    Leaf = TC.leaf_values(engine, G, H, kp, n, D, lr=0.3)
    peak_tree = mark("트리 완성")
    t_tree = time.time() - t0

    t1 = time.time()
    pred = TC.scatter_leaves(engine, Leaf, masks, leaf_oh, kp, td)
    gv = TC.update_residual(engine, gv, pred, kp, td, lr=1.0)
    peak = mark("부스팅 1라운드")
    t_boost = time.time() - t1

    del tm, am, leaf_oh, pre, gv, sel2, masks, Lq, Nq, G, H, Leaf, pred
    gc.collect()
    return dict(nct=td["nct"], marks=marks, peak=peak,
                t_tree=t_tree, t_boost=t_boost)


def main():
    print("=" * 92)
    print(f"메모리 실측 [{MODE.upper()}]  (d={D_FEAT}, n_max={N_MAX}, D={DEPTH}, α={ALPHA})")
    print("=" * 92)
    u, t = gpu_mb()
    if t:
        print(f"  GPU 총 {t:,} MB,  시작 시 사용 {u:,} MB")
    engine, sk, kp = setup_fhe_engine(mode=MODE, verbose=False)
    u0, _ = gpu_mb()
    print(f"  엔진·키 로드 후 {u0:,} MB" if u0 else "  (nvidia-smi 사용 불가)")
    print()
    print(f"  {'n':>6}{'nct':>5} | {'마스크':>9}{'prefix':>9}{'argmax':>9}"
          f"{'트리':>9}{'부스팅':>9} | {'학습(s)':>9}{'라운드(s)':>10}")
    print("  " + "-" * 84)
    for n in N_LIST:
        try:
            r = probe(engine, sk, kp, n, D_FEAT, N_MAX, DEPTH)
        except RuntimeError as ex:
            print(f"  {n:>6}      OOM/에러: {str(ex)[:56]}")
            print("  → 여기가 한계.  이보다 작은 n 을 쓰세요.")
            break
        m = dict(r["marks"])
        f = lambda k: f"{m.get(k, 0):,}" if m.get(k) else "-"
        print(f"  {n:>6}{r['nct']:>5} | {f('마스크'):>9}{f('prefix'):>9}"
              f"{f('argmax 1회'):>9}{f('트리 완성'):>9}{f('부스팅 1라운드'):>9}"
              f" | {r['t_tree']:>9.1f}{r['t_boost']:>10.1f}")
        gc.collect()
    print()
    print("  · 숫자는 nvidia-smi 사용량(MB) — 키·작업공간·단편화 전부 포함")
    print("  · 학습(s) = 트리 1개,  라운드(s) = 잔차 갱신 추가분")
    print("  · K=20 총시간 ≈ 20 × (학습 + 라운드)")


if __name__ == "__main__":
    main()
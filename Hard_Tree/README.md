# Data-Analysis-with-CKKS
- pip list
    - pip install numpy
    - pip install desilofhe-cu128
- Test
    1. pyenv activate hardtree
    2. (hardtree) junhyung@cryptolab2:~/study/Data_Analysis_with_CKKS$ python3 -m Hard_Tree.test.test_cheby_levels

# Model
# Hard_Tree — CatBoost-FHE

symmetric tree + 비교 기반 노드 표현으로 FHE 결정트리의 depth 벽을 깨는 구현.

## 배치

```
Hard_Tree/
  desilo/
    util/keypack.py      ★ 기존 파일 교체 (refresh + sign_refresh)
    hard_core.py         레이아웃 · plan_inference · extract_model · 추론
    minimax_sign.py      Remez + minimax composite sign
    train_core.py        학습 루프 T0·T1·T2·T3·T4·T5·T6 (league argmax 포함)
    opcount.py           FHE 연산 카운터 (병목 진단 · 하드웨어 정규화)
  test/
    test_rot_cost.py     하드웨어 상수 실측      ← 제일 먼저
    test_cheb_levels.py  체비셰프 레벨 소비 실측
    test_s1_inference.py 추론 검증               (20/20)
    test_s3_extract.py   학습부 모델 갱신 검증    (20/20)
    test_s4_train.py     학습 루프 검증          (오차 0)
    test_s5_endtoend.py  학습→모델→추론 전 경로   (20/20)
    test_s6_boostsplit.py 부스팅 분할 기준 검증
    test_s7_argmax.py    FHE league argmax 검증
    test_opcount.py      구간별 연산 수 계측
    test_unit_cost.py    단위 연산 비용 실측 (opcount 상수 확정)
```

`__init__.py` 는 빈 파일. 이미 있으면 덮어쓰지 마세요.

## 의존 관계

```
keypack.py          (외부 의존 없음)
   ↑
hard_core.py  ──→ minimax_sign.py
   ↑
train_core.py       (hard_core 의 p2 · _rot · _lvl 만 사용)
```

**테스트는 전부 위 4개 모듈에만 의존합니다.**
`packed_core.py` · `tree_core.py` · `gate_eval.py` 등 soft tree 시절 파일은
**더 이상 쓰지 않습니다.**

## 실행 순서

```bash
# 1) 하드웨어 상수 실측 → hard_core.py 상단에 반영
python -m Hard_Tree.test.test_rot_cost
python -m Hard_Tree.test.test_unit_cost      # ★ opcount 단가 확정
python -m Hard_Tree.test.test_cheb_levels
python -m Hard_Tree.desilo.util.keypack      # refresh / sign_bootstrap 비교

# 2) 검증
python -m Hard_Tree.test.test_s1_inference   # 추론
python -m Hard_Tree.test.test_s3_extract     # 학습부 모델 갱신
python -m Hard_Tree.test.test_s4_train       # 학습 루프
python -m Hard_Tree.test.test_s5_endtoend   # 전 경로 (symmetric)
python -m Hard_Tree.test.test_s6_boostsplit  # 부스팅 분할 기준
python -m Hard_Tree.test.test_s7_argmax      # FHE argmax
python -m Hard_Tree.test.test_opcount        # 구간별 연산 수
```

## keypack 에 필요한 키

`sign_bootstrap` 을 쓰려면 두 개가 필요합니다 (교체본에 포함).

```python
kp.conjugation_key    = engine.create_conjugation_key(secret_key)
kp.smallbootstrap_key = engine.create_small_bootstrap_key(secret_key)
# 선택 (빠르지만 메모리 더 씀)
kp.lossy_bootstrap_key = engine.create_lossy_bootstrap_key(secret_key, stage_count=3)
```

## 조정 상수 (`hard_core.py` 상단)

```python
L_FRESH   = 26        # fresh 암호문 레벨
L_SMALL   = 10        # refresh 회복 레벨
BOOT_MS   = 2814.4    # refresh 1회 (실측)
ROT_P2_MS  = 3.49     # 2^k shift 회전 (level 13, 실측)
ROT_ANY_MS = 20.25    # 임의 shift 회전 — 5.8배 비싸다
SPREAD_SEQ_BITS = None       # None = 깊이 예산에서 자동
SIGNBOOT_TABLE  = {3: 2280.3, 4: 1461.4, 5: 1256.5}   # 실측
SIGNBOOT_STAGES = (5, 4, 3)  # planner 가 자동 선택
```

**부트스트랩 실측 (A6000/Desilo)**

| | 입력≥ | 회복 | 시간 | 최대오차 |
|---|---|---|---|---|
| refresh | — | 10 | 2814.4 ms | — |
| sign_bootstrap sc=3 | 3 | 13 | 2280.3 ms | 7.69e-07 |
| sign_bootstrap sc=4 | 4 | 12 | 1461.4 ms | 8.08e-07 |
| sign_bootstrap sc=5 | 5 | 11 | 1256.5 ms | 7.85e-07 |

★ `sign_bootstrap` 이 `refresh` 보다 **항상 싸다.** ±1 상태에서 걸 수 있으면 그쪽을 쓴다.
★ 정밀도는 stage 와 무관. 레벨 1개당 ~800ms 인데 그 1레벨로 아끼는 회전은 60ms 수준
  → 큰 stage 가 유리하지만 입력 레벨 제약이 있어 `plan_inference` 가 자동 선택한다.

`test_s3_extract.py` 의 `EXTRACT_MAX_DEPTH` (기본 4) 는
낮출수록 모델 레벨이 높아져 추론이 유리하고, 학습 회전이 늘어납니다.

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| S1 | 비교 기반 추론 | ✅ 실제 FHE 20/20 |
| S2 | minimax composite sign | ✅ 평문 검증 + FHE 연동 |
| S3 | 좌표계 전환 (argmin → (onehot_j, θ)) | ✅ 실제 FHE 20/20 |
| S4 | 학습 루프 T0·T1·T2·T5 (방식 C) | ✅ 실제 FHE 오차 0 |
| S5 | T4 연결 + symmetric 추론 | ✅ 목 20/20 |
| S6 | 부스팅 분할 기준 (나눗셈 없는 대리) | ✅ FHE 4/4 |
| S7 | league argmax (전쌍, sign 1회) | 🟡 레벨 0 통과, 깊은 레벨 보정 필요 |
| — | T3 를 학습 루프에 연결 | ⚠️ `ARGMAX_MODE="plain"` 이 기본 |
| — | T4-b 하이브리드 부스팅 (M=4) | ⚠️ 다음 단계 |
| T6 | 리프값 −lr·G/(H+λ), Goldschmidt 역수 | ✅ 오차 1e-16 |

## symmetric 추론 (`infer_symmetric`)

레벨 내 모든 노드가 같은 분기이므로 **분기 D 개만 평가**한다.
비대칭의 stride 확장(임의 shift 대각 2^t 개)이 통째로 사라진다.

```
            비대칭 노드평가   symmetric   배율
   D=6           63             6       10.5x
   D=8          255             8       31.9x
   D=10       1,023            10      102.3x
```

모델도 `D·dp` 슬롯이면 된다 (비대칭은 `2^D·dp`).


## 핵심 기법 세 가지 (2026-08-30 추가)

**① 성분 사이 sign_bootstrap (`inter_boot`)** — Hong et al. §4.1
minimax 합성 다항식을 통째로 평가하면 α=12 에 깊이 25 가 필요하지만,
성분마다 `sign_bootstrap` 을 끼우면 **한 번에 필요한 깊이가 성분 하나(~5)** 다.
```
α        차수                 한번에   성분별   sign_boot
 6    [15,15]                  11       6         1
12    [7,15,15,15,15]          25       6         4
```
덤으로 sin 이 유계라 중간값 폭주를 클램프하고, τ→(π²/8)τ² 이차 수렴이 공짜다.
**→ α 의 레벨 천장이 사라진다.**

**② full bootstrap_key 불필요**
small 키의 5-인자 `engine.bootstrap(ct, relin, conj, rotation, small)` 이
값 보존 bootstrap 으로 동작한다. 13.4GB 를 아낀다.

**③ 점수 정규화를 휴리스틱 없이 계산으로**
```
g′ = g/‖g‖ (단위노름)  →  |G′_L| ≤ √H_L   (Cauchy-Schwarz)
score ≤ 4·Σ_p H′_p³                        ← 암호문에서 계산 가능
```
`Σ_p H′_p³` 로 나누면 상한이 정확히 1 이고, 그 나눗셈의 동적 범위가
`4^t` 로 알려져 있어 Goldschmidt `2t+5` 회면 된다. **데이터셋별 보정표가 없다.**

## sign_bootstrap 의 실체 (2026-08-30 정정)

```
sign_bootstrap = 주기적 EvalSign = sin(π·z/2)
   sin(π/2)=1,  sin(π)=0,  sin(3π/2)=−1     (z≈2→0, z≈3→−1 접힘)
   z=1+τ 이면 cos(πτ/2) ≈ 1 − (π²/8)τ²      (Hong et al. Thm 1)
```

★ **부호를 강제하지 않는다.**  minimax 합성으로 **이미 ±1 근처에 온 값**을
  더 밀어주는 + 레벨 복구다.  먼 값(0.045)은 ±1 이 되지 않고 0.07 이 된다.
★ 따라서 부트는 **τ < 8/π² = 0.811 인 지점에만** 건다.
  τ 가 크면 다음 성분이 자기 도메인 밖 값을 받아 발산 (실측 오차 4.1e+03).

| α | 차수 | 부트 | 최대오차 | sdep |
|---|---|---|---|---|
| 4 | [27] | `.` | 9.7e-02 | 7 |
| 6 | [15,15] | `B.` | 5.9e-03 | 6 |
| 8 | [7,15,15] | `.B.` | 3.7e-04 | 10 |
| 12 | [7,15,15,15,15] | `..BB.` | 8.9e-16 | 15 |

## 정규화 — 레벨별 분기

`level` 은 공개 파라미터이므로 분기해도 된다.

```
t=0 : 파티션이 루트 하나 → H′=1 → Σ_p H′_p³ = 1 → 분모가 **공개 상수 4**
      나눗셈 불필요.  깊이 1 (pt 곱만)
t≥1 : Σ_p H′_p³ ∈ [1/4^t, 1] → 범위 4^t → Goldschmidt 2t+5 회
```

```
  t   반복   깊이   이전(max8)   절감
  0     0      1        12        11     ★
  1     7     11        12         1
  2     9     14        14         0
```

이 11 레벨 덕분에 α=8~12 가 league argmax 에 들어간다.

## league argmax — 확정된 설계

`test_s7_argmax` 가 **12/12 통과** (4 구성 × α ∈ {8,10,12}).
실측 진단으로 α 요구치가 나왔다.

```
구성            score max   top gap   필요 α
n=64  ns=28      1.5e-01   3.0e-03      9
n=100 ns=42      7.3e-01   2.7e-01      2
n=200 ns=35      1.6e-01   1.1e-02      7
n=379 ns=35      2.3e-01   1.0e-01      4
```

★ **α=8 이면 충분하다.**  α ∈ {8,10,12} 전부 12/12 통과.
★ `mtry` 는 직교하는 축이다 — 후보를 줄이면 필요 α 가 낮아지고 T1 도 준다.
  둘 중 택일이 아니라 **같이 쓴다.**

### 잡은 버그 세 개

**① 스케일 폴딩 ↔ inter_boot 충돌**
`build_composite` 은 `1/w_{i+1}` 을 이전 성분 계수에 접어넣는데, 이는
`z_{i+1} = p_i(z_i)/w_{i+1}` 을 전제한다.  `sign_bootstrap` 이 z 를 ±1 로
만들어 전제가 깨진다. **증상: 성분 수가 짝수일 때만 실패.**
→ `inter_boot=True` 면 폴딩하지 않고, 각 성분 도메인을 **부트 이후 τ**
  (`τ → (π²/8)τ²`, Hong et al. Thm 1) 로 잡는다.

**② 블록 곱이 경계를 넘음**
`ind_i = Π_j b_ij` 의 블록 곱은 블록 크기가 2의 거듭제곱일 때만 성립.
`ns=35, 42` 처럼 아니면 회전이 옆 블록을 침범한다.
→ `nsp = 2^⌈log2 ns⌉` 로 패딩하고 남는 슬롯은 **1** 로 채운다 (곱에 무영향).

**③ 평문끼리 연산 불가**
DesiloFHE 는 `engine.add(pt, pt)` / `subtract(pt, pt)` 를 지원하지 않는다.
→ 마스크 생성 시점에 numpy 로 미리 합쳐 하나의 평문으로 인코딩한다
  (`am["diag_pad"] = diag + pad`).  연산도 하나 줄어든다.
→ 재발 방지: 목 엔진이 평문끼리 연산을 **TypeError 로 거부**하도록 했다.

**④ 곱이 오차를 ns 제곱으로 증폭**
`b=0.9, ns=35` 면 `0.9^35 = 0.025`.  단계마다 `bit_cleaning`
(`h(x)=3x²−2x³`, 이차 수렴, 레벨 2) 으로 {0,1} 로 되돌린다.

## ⚠️ 남은 것

**league 경로는 모델 레벨을 많이 소모한다.**  `extract_split` 이후 Model 이
레벨 3~4 까지 내려가 추론 전에 refresh 가 필요하고, 일부 구성(d=12, D=4)은
그래도 부족하다.  → `plan_inference` 를 학습 경로까지 확장하는 것이 다음 작업.
기본값 `ARGMAX_MODE="plain"` 은 전 구성 통과.
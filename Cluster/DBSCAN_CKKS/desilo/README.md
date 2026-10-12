python: 3.12.10

### desilo version 사용법
1. core_test directory의 test_{}을 실행
    - 1. pyenv activate dbscan(dbscan 이름의 python 가상환경 실행)
    - 2. DBSCAN_CKKS/desilo로 이동
    - 3. python -m pytest core_test/test_normalize.py
2. dataset은 dataset directory에 넣은 후 test_main파일에서 DATASETPATH를 해당 파일 위치로 설정
3. EPS와 minpoints 입력

### Original_DBSCAN 사용법
1. Original_DBSCAN directory의 test_main실행
    - 1. python3 -m Original_DBSCAN.test_main

### test_{} 사용볍
1. core_test directory의 test_{}을 실행
    - 1. pyenv activate dbscan(dbscan 이름의 python 가상환경 실행)
    - 2. DBSCAN_CKKS/desilo로 이동
    - 3. python3 -m core_test.test_normalize

### Dataset 별 DBSCAN parameter
1. hepta(212개)
    - max_iter:3
    - eps: 1.0
    - minpts: 4
    - 결과: 군집 7개(ARI 100점)
2. tetra(400개)
    - max_iter: 4
    - eps: 0.43
    - minpts: 3
    - 결과: 군집 4개/정답: 4개(ARI 97.25점)
3. TwoDiamonds(800개)
    - round: 16
    - eps: 0.14
    - minpts: 6
    - 결과: 군집 2개/정답: 2개(ARI 99.75점)
4. Lsun(400개)
    - round: 16
    - eps: 0.5
    - minpts: 4
    - 결과: 군집 3개(ARI 100점)
5. target(770개)
    - max_iter: ?
    - eps: 0.25
    - minpts: 4
    - 결과: 군집 6개(ARI 99.96점 - 이게 맞음. noise로 분류되어야 할 점들이 ground truth에서는 각 cluster로 표현됨(eps 0.25 minpoints 3으로 하면 ARI 100 나옴))
6. 3_spiral(312개)
    - max_iter: ?
    - eps: 1.4
    - minpts: 3
    - 결과: 군집 3개(ARI 100점)
7. atom(800개)
    - max_iter: ?
    - eps: 20
    - minpts: 4
    - 결과: 군집 3개(ARI 100점)
8. Longsquare(900개)
    - round: 12
    - eps: 1.15
    - minpts: 6
    - 결과: 군집 6개/정답: 6개(ARI 95.02점)
8. S1(5000개)
    - round: 5
    - eps: 44000
    - minpts: 50
    - 결과: 군집 6개/정답: 6개(ARI 95.02점)
9. twomoons(200개)
    - round: 29
    - eps: 0.28
    - minpts: 5
    - 결과: 군집 2개/정답: 2개(ARI 100점)
10. chainlink
    - round: 12
    - eps: 0.15
    - minpts: 6
    - 결과: 군집 2개/정답: 2개(ARI 100점)
-------------------------------------------
1. Iris(150개)
    - max_iter: 4
    - eps: 0.4
    - minpts: 6
    - 결과: 군집 4개/정답: 3개(ARI 59.34점 - 원래 DBSCAN이 Iris같은 밀도가 다른 점에 대해서 약함)
2. banknote_auth(1372개)
    - max_iter: 6
    - eps: 1.8
    - minpts: 8
    - 결과: 군집 10개/정답: 7개(ARI 66.06점)
2. seeds(210개)
    - max_iter: 3
    - eps: 2.6
    - minpts: 3
    - 결과: 군집 3개/정답: 3개(ARI 80.39점)
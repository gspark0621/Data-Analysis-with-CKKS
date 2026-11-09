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

### pip install
- pip install numpy scikit-learn pynvml 
### Dataset 별 DBSCAN parameter
1. hepta(212개)
    - eps: 1.0
    - minpts: 4
    - round: 3
    - 결과: 군집 4개 / ARI 1.0
2. tetra(400개)
    - eps: 0.819
    - minpts: 45
    - round: 1
    - 결과: 군집 4개 / ARI 1.0
3. TwoDiamonds(800개)
    - eps: 0.14
    - minpts: 6
    - round: 13
    - 결과: 군집 2개 / ARI 0.9975
4. Lsun(400개)
    - eps: 0.5
    - minpts: 4
    - round: 8
    - 결과: 군집 3개 / ARI 1.0
8. Longsquare(900개)
    - eps: 1.15
    - minpts: 6
    - round: 11
    - 결과: 군집 6개 / ARI 0.9502
8. S1(5000개)
    - eps: 44000
    - minpts: 50
    - round: 2
    - 결과: 군집 6개 / ARI 
11. leukemia(554개)
    - eps: 0.85
    - minpts: 4
    - round: 4
    - 결과: 군집 4개 / ARI 0.9959
11. swiss_banknote(200개)
    - eps: 0.85
    - minpts: 4
    - round: 6
    - 결과: 군집 2개 / ARI 0.9701
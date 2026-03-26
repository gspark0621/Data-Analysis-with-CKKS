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
    - max_iter: 10
    - eps: 0.11
    - minpts: 4
    - 결과: 군집 3개/정답: 2개(ARI 95.55점)
4. Lsun(400개)
    - max_iter: 3
    - eps: 0.5
    - minpts: 4
    - 결과: 군집 3개(ARI 100점)
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

### TODO
1. scaling factor를 이용하여 Normalize.py와 Core.py에서 근사함수 input의 원하는 범위로 scaling
    - 1. plaintext를 곱할때에도 plaintext를 encoding해서 넣기 때문에 scaling factor에 영향을 주는지 확인
    - 2. Core.py에서는 (#neighbor - minpoint)가 발산하기 때문에 이를 scaling해야 하는데, 이를 어떻게 해야할지 고민
2. 비교연산의 precision이 어떻게 되는지 확인
3. 노이즈 제거 알고리즘 추가
4. Client_main.py

### 문제점 - 해결점
1. Normalize.py(dist^2- eps^2의 값을 1(이웃 O) 또는 0(이웃 X)로 매핑)와 Core.py(한 점의 이웃의 개수를 통해 1(core point) 와 0(border or noise)인지 판별)에서 매핑을 할 때 입력값(dist^2-eps^2 또는 이웃의 개수)가 -1~1사이 값이어야 함.
    - 3가지 방법이 존재했었음
        1. client측에서 server에게 각 좌표계의 최댓값을 알려주고, server는 이를 이용하여 각 값을 최댓값으로 나눠주어 scaling하는 방식
            - 서버가 특정 차원의 최댓값, 최솟값을 통해 범위를 알게 되고, 이를 통해 데이터의 분포나 특정 이상치의 존재를 알 수도 있음
        2. FHE의 최댓값(word size)로 나눠주어 scaling 하는 방식
            - data가 모두 0에 몰려 있어, 매핑 시 많은 반복을 요하고, 이는 depth의 증가(연산의 증가)로 귀결
        3. (현재 진행 방식)client가 사전 정규화하여, input을 0과 1사이로 scaling 한 후 넘김
            - global scaling 방식: 각 좌표계의 최댓값을 모아, 하나의 최댓값을 정한 후, 그 값으로 sclaing하는 방식
                - 각 좌표별로 최댓값의 scaling을 진행하면, 원(또는 구) 상에서 진행하는 Euclidean distance 방식이 불가능함(∵원형이 아닌 타원상으로 좌표계가 변하기 때문)
            - Client_main.py에서 input data와 eps에 대한 정규화 시행
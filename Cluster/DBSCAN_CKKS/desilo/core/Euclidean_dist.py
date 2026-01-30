from desilofhe import Engine

def Euclidean_Distance_ct(
    engine: Engine, 
    p_enc_list: list, 
    q_enc_list: list, 
    relinearization_key
):
    #차원별 패킹 방식에 Tree Summation(Logarithmic Depth)을 적용한 최적화 버전.
    
    # 입력 개수 검증
    if len(p_enc_list) != len(q_enc_list):
        raise ValueError("Dimension mismatch")
    
    # 1. 각 차원별 제곱 차이(Squared Difference) 계산(SIMD)
    squared_diffs = []
    for p, q in zip(p_enc_list, q_enc_list):
        diff = engine.subtract(p, q)
        # multiply는 noise를 많이 증가시키므로, 여기서 최대한 평평하게 시작하는 것이 좋음
        sq = engine.square(diff,relinearization_key)
        squared_diffs.append(sq)
    
    # 2. 트리 구조 합산 (Tree-based Reduction)
    # Ex. [A, B, C, D] -> [A+B, C+D] -> [A+B+C+D]
    current_list = squared_diffs
    
    while len(current_list) > 1:
        next_list = []
        for i in range(0, len(current_list), 2):
            if i + 1 < len(current_list):
                # 짝이 있으면 더함
                summed = engine.add(current_list[i], current_list[i+1])
                next_list.append(summed)
            else:
                # 짝이 없으면(홀수 개일 때) 그대로 다음 라운드로 토스
                next_list.append(current_list[i])
        current_list = next_list
        
    # 마지막 남은 하나가 최종 결과
    return current_list[0]

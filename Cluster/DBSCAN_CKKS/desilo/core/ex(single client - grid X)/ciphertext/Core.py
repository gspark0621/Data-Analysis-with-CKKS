import math
from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack

#TODO: 현재방식과 최적화된 합성 미니맥스(Minimax) 다항식(5-3. Optimization of Homomorphic Comparison... 논문 참고)
def identify_core_points_fhe_converted(
    engine: Engine, 
    neighbor_count_ct: Ciphertext, 
    min_pts: float, 
    N: int, 
    keypack: KeyPack, 
    bootstrap_interval: int = 3, 
    **kwargs
) -> Ciphertext:
    
    relin_key = keypack.relinearization_key
    conj_key = keypack.conjugation_key
    boot_key = keypack.bootstrap_key
    
    # 1. x = neighbor_count - (min_pts - margin)
    # Core.txt의 마진 로직 적용
    margin = 0.5
    min_pts_margin = min_pts - margin
    min_pts_pt = engine.encode([min_pts_margin for _ in range(N)])
    x = engine.subtract(neighbor_count_ct, min_pts_pt)
    
    # 2. 발산 방지를 위한 극단적 스케일링
    # 최대 이웃 수는 N을 넘을 수 없으므로 N으로 나누어 [-1, 1] 범위로 압축
    scale_factor = 1.0 / float(N)
    scale_pt = engine.encode([scale_factor for _ in range(N)])
    current_x = engine.multiply(x, scale_pt)
    
    # 3. 데이터 수(N)에 비례하는 다항식 깊이 자동 산출
    required_depth = math.ceil(math.log(N / margin, 1.5)) + 1
    print(f"[Server] Core 판별을 위한 Sign 근사 반복 횟수 (N={N}): {required_depth}회 자동 설정")
    
    # 동형암호 최적화를 위한 평문 상수 사전 인코딩 (참조 코드 적용)
    c15_pt = engine.encode([1.5 for _ in range(N)])
    c05_pt = engine.encode([0.5 for _ in range(N)])
    
    # 4. Sign 다항식 근사 반복 (1.5x - 0.5x^3)
    for i in range(required_depth):
        x_sq = engine.square(current_x, relin_key)
        x_cub = engine.multiply(x_sq, current_x, relin_key)
        
        term1 = engine.multiply(current_x, c15_pt)
        term2 = engine.multiply(x_cub, c05_pt)
        
        current_x = engine.subtract(term1, term2)
        
        # 노이즈 폭발을 막기 위한 주기적 부트스트래핑
        if (i + 1) % bootstrap_interval == 0 and (i + 1) != required_depth:
            print(f"  - [Core] {i+1}회 반복 완료. 중간 부트스트래핑 수행...")
            current_x = engine.intt(current_x)
            current_x = engine.bootstrap(current_x, relin_key, conj_key, boot_key)
            
    # 5. 매핑 로직: Core(x=+1) -> 1.0, Non-core(x=-1) -> 0.0 
    # Core.txt의 half_x + 0.5 수식을 동형암호 연산으로 치환
    half_pt = engine.encode([0.5 for _ in range(N)])
    half_x = engine.multiply(current_x, half_pt)
    core_indicator = engine.add(half_x, half_pt)
    
    # 최종 데이터 무결성 확보를 위한 부트스트래핑
    core_indicator = engine.intt(core_indicator)
    return engine.bootstrap(core_indicator, relin_key, conj_key, boot_key)

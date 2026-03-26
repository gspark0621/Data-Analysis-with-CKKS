from dataclasses import dataclass

@dataclass
class ClusteringConfig:
    """
    Clustering Benchmark Dataset Configuration
    Ref: Optimized Privacy-Preserving Clustering with Fully Homomorphic Encryption
    """
    name: str           # 데이터셋 이름 (예: 'S1', 'A1')
    n_samples: int      # 데이터 포인트 개수 (N)
    n_features: int     # 데이터 차원 (D)
    n_clusters: int     # 클러스터 개수 (k) - 참고용
    slot_counts: int    # FHE 슬롯 개수 (예: 32768)

class ComparisonConfig:
    def __init__(self, name, n_samples, d, m, t, slot_counts=1<<15):
        self.name = name
        self.n_samples = n_samples
        self.d = d          # Goldschmidt 반복 횟수
        self.m = m          # Plaintext 상한값 (inv_goldschmidts용)
        self.t = t          # Scaling factor 또는 반복 횟수 (Comp용)
        self.slot_counts = slot_counts
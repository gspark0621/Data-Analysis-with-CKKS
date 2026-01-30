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

CLUSTERING_DATASETS = [
    # Synthetic Datasets (S-sets)
    ClusteringConfig("G2-1-20", 2048, 1, 2, 1 << 15),
    ClusteringConfig("G2-2-20", 2048, 2, 2, 1 << 15),
    ClusteringConfig("G2-4-20", 2048, 4, 2, 1 << 15),
    ClusteringConfig("G2-8-20", 2048, 8, 2, 1 << 15),
    ClusteringConfig("G2-16-20", 2048, 16, 2, 1 << 15),
    
    # Real-world Datasets (UCI 등) - 논문 실험용
    ClusteringConfig("Iris", 150, 4, 3, 1 << 15),
    ClusteringConfig("Wine", 178, 13, 3, 1 << 15),
    ClusteringConfig("Cancer", 569, 30, 2, 1 << 15), # Breast Cancer Wisconsin
    
    # Large scale (A-sets)
    ClusteringConfig("A1", 3000, 2, 20, 1 << 15),
    ClusteringConfig("A2", 5250, 2, 35, 1 << 15),
    ClusteringConfig("A3", 7500, 2, 50, 1 << 15),
]

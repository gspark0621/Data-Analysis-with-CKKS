# client/kdca_client.py
import desilofhe
from desilofhe import Engine

def decrypt_and_visualize_clusters(engine: Engine, secret_key, encrypted_cluster_labels, grid_centers):
    """
    질병관리청이 최종 Cluster Label을 복호화하고 결과를 분석합니다.
    """
    print("[KDCA Client] 서버로부터 암호화된 군집 결과 수신 완료.")
    print("[KDCA Client] 복호화 진행 중...")
    
    # 서버의 연산 과정에서 발생한 근사 오차(Noise)를 포함하여 복호화됨
    decrypted_labels = engine.decrypt(encrypted_cluster_labels, secret_key)
    
    final_clusters = {}
    num_grids = len(grid_centers)
    
    for i in range(num_grids):
        # CKKS의 근사 오차 보정 (예: 0.999 -> 1.0, 0.001 -> 0.0)
        label = round(decrypted_labels[i])
        
        if label <= 0:
            continue # 노이즈(Noise) 지역은 무시
            
        if label not in final_clusters:
            final_clusters[label] = []
        final_clusters[label].append(grid_centers[i])
        
    print("\n================ [ 최종 방역 군집 지도 ] ================")
    if not final_clusters:
        print("위험 군집(Core)이 발견되지 않았습니다. 전국이 안전합니다.")
    else:
        for cluster_id, centers in final_clusters.items():
            print(f"▶ Cluster {cluster_id}: {len(centers)}개의 격자가 하나로 연결된 대형 핫스팟입니다.")
            for c in centers:
                print(f"   - 격자 중심 좌표: {c}")
    print("=========================================================\n")
    
    return final_clusters

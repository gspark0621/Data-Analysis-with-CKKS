# core/plaintext/FinalClient_plain.py
def reconstruct_results_plain(server_result, global_min, scale_factor):
    all_points = server_result["all_points"]
    labels = server_result["labels"]
    core_mask = server_result["core_mask"]
    neighbor_counts = server_result["neighbor_counts"]

    results = []
    for item in all_points:
        gidx = item["global_idx"]
        
        # Scaling up (역정규화): v_raw = (v_norm * scale_factor) + global_min
        point_reconstructed = [
            round((v * scale_factor) + global_min, 6) 
            for v in item["point_norm"]
        ]
        
        results.append({
            "global_idx": gidx,
            "owner_id": item["owner_id"],
            "owner_local_idx": item["owner_local_idx"],
            "point_norm": item["point_norm"],
            "point_reconstructed": point_reconstructed,  # 복원된 좌표
            "grid_idx": item["grid_idx"],
            "block_idx": item["block_idx"],
            "neighbor_count": neighbor_counts[gidx],
            "is_core": core_mask[gidx],
            "label": labels[gidx]
        })
    return results
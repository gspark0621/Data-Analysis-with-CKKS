# core/plaintext/MultipartyOwner_plain.py
from core.plaintext.GridIndex_plain import normalize_points_global, bucketize_points_by_grid

def prepare_owner_blocks_plain(owner_points_raw,
                               domain_mins_norm,
                               domain_maxs_norm,
                               epsilon_norm,
                               axis_cell_counts,
                               bucket_size,
                               max_blocks_per_grid,
                               global_min,
                               global_max,
                               owner_id):
    # 정규화 진행
    owner_points_norm, _ = normalize_points_global(owner_points_raw, global_min, global_max)
    
    blocks = bucketize_points_by_grid(
        points_norm=owner_points_norm,
        domain_mins_norm=domain_mins_norm,
        domain_maxs_norm=domain_maxs_norm,
        epsilon_norm=epsilon_norm,
        axis_cell_counts=axis_cell_counts,
        bucket_size=bucket_size,
        max_blocks_per_grid=max_blocks_per_grid,
        owner_id=owner_id
    )
    return blocks, owner_points_norm
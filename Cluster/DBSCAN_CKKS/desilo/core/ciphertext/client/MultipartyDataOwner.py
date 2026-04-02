# core/client/MultipartyDataOwner.py
from core.client.EncryptModule import DimensionalEncryptor
from core.server.GridIndex import bucketize_points_by_grid

def normalize_points_global(points, global_min, global_max):
    scale = global_max - global_min
    if scale == 0.0:
        scale = 1.0
    out = []
    for row in points:
        out.append([(v - global_min) / scale for v in row])
    return out, scale

def encrypt_owner_blocks(engine,
                         keypack,
                         owner_points,
                         grid_centers,
                         epsilon,
                         bucket_size,
                         max_blocks_per_grid,
                         global_min,
                         global_max):
    normalized_points, scale = normalize_points_global(owner_points, global_min, global_max)

    blocks, block_meta = bucketize_points_by_grid(
        normalized_points,
        grid_centers,
        epsilon / scale,
        bucket_size,
        max_blocks_per_grid
    )

    encryptor = DimensionalEncryptor(engine, keypack)

    encrypted_blocks = []
    for blk in blocks:
        enc_coords = encryptor.encrypt_data(blk["points"], dim=len(blk["points"][0]))
        encrypted_blocks.append({
            "enc_coords": enc_coords,
            "selection_mask_pt": blk["selection_mask"],  # 평문 유지
            "grid_idx": blk["grid_idx"],
            "block_idx": blk["block_idx"]
        })

    return encrypted_blocks, scale
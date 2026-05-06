# core/ciphertext/client/MultipartyDataOwner.py
from core.ciphertext.client.GridIndex import (
    point_to_grid_index,
    pack_points_column_major,
    normalize_points_global
)


def prepare_and_encrypt_owner_blocks(
    engine,
    keypack,
    owner_points_raw:  list,
    grid_centers_norm: list,   # FinalClient 가 계산 후 전달
    epsilon_norm:      float,  # = base_epsilon_norm (격자 셀 크기)
    bucket_size:       int,
    global_min:        float,
    global_max:        float,
    owner_id:          int,
):
    """
    DO 데이터를 컬럼-메이저로 패킹 후 FinalClient PK 로 암호화.

    반환:
      client_pack ─ FinalClient 전용
                    slot_to_point_norm 포함 (서버에 절대 전달하지 말 것)
      server_pack ─ 서버 전용
                    암호문 + selection_mask 평문만 (좌표·ref 없음)
    """
    dim     = len(owner_points_raw[0]) if owner_points_raw else 2
    G       = len(grid_centers_norm)
    N_batch = bucket_size * G

    # ── 1. 전역 정규화 ────────────────────────────────────────
    owner_points_norm, _ = normalize_points_global(
        owner_points_raw, global_min, global_max
    )

    # ── 2. 격자별 포인트 할당 (bucket_size 초과 버림) ────────────
    grid_to_pairs = {g: [] for g in range(G)}
    for local_idx, pt in enumerate(owner_points_norm):
        g = point_to_grid_index(pt, grid_centers_norm, epsilon_norm)
        if g is not None and len(grid_to_pairs[g]) < bucket_size:
            ref = {"owner_id": owner_id, "owner_local_idx": local_idx}
            grid_to_pairs[g].append((pt, ref))

    # ── 3. 컬럼-메이저 패킹 ──────────────────────────────────
    packed_coords, packed_mask, slot_to_ref, slot_to_point_norm = \
        pack_points_column_major(grid_to_pairs, G, bucket_size, dim)

    # ── 4. FinalClient PK 로 암호화 ──────────────────────────
    enc_coords = []
    for d in range(dim):
        ct = engine.encrypt(engine.encode(packed_coords[d]), keypack.public_key)
        engine.to_cpu(ct)
        enc_coords.append(ct)

    enc_mask = engine.encrypt(engine.encode(packed_mask), keypack.public_key)
    engine.to_cpu(enc_mask)

    # ── 5. 패킷 구성 ─────────────────────────────────────────
    client_pack = {
        "owner_id":           owner_id,
        "slot_to_ref":        slot_to_ref,        # {(i,g): ref}
        "slot_to_point_norm": slot_to_point_norm, # {slot_idx: pt_norm}
        "packed_mask":        packed_mask,
        "N_batch":            N_batch,
        "G":                  G,
        "bucket_size":        bucket_size,
    }

    server_pack = {
        "enc_coords":         enc_coords,   # List[Ciphertext] CPU
        "enc_selection_mask": enc_mask,     # Ciphertext       CPU
        "selection_mask_pt":  packed_mask,  # 평문 마스크 (연산 편의용)
    }

    active = sum(1 for pts in grid_to_pairs.values() if pts)
    valid  = int(sum(packed_mask))
    print(f"[DO {owner_id}] {len(owner_points_norm)}개 포인트 → "
          f"{active}/{G} 격자 활성, {valid}/{N_batch} 유효 슬롯")

    return client_pack, server_pack
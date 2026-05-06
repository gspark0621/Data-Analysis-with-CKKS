# core/client/FinalClient.py

from typing import Dict, List, Optional, Tuple


def assign_global_indices(
    owner_client_packs: List[dict],
) -> Tuple[List[dict], int]:
    """
    각 DO 의 client_pack 을 순회하여 유효 슬롯에 전역 인덱스 부여.

    반환:
      all_points     : 유효 포인트 메타데이터 리스트
                       각 원소: {global_idx, owner_id, owner_local_idx,
                                 slot_idx, grid_idx, k_within_bucket}
                       ★ point_norm 미포함 ─ 정보 유출 방지
      global_counter : 전체 유효 포인트 수
    """
    global_counter = 0
    all_points: List[dict] = []

    for pack in owner_client_packs:
        owner_id    = pack["owner_id"]
        G           = pack["G"]
        bucket_size = pack["bucket_size"]
        N_batch     = pack["N_batch"]
        packed_mask = pack["packed_mask"]
        slot_to_ref = pack["slot_to_ref"]   # {(i, g): ref_dict}

        for i in range(bucket_size):
            for g in range(G):
                slot_idx = i * G + g
                if slot_idx >= N_batch:
                    continue
                if packed_mask[slot_idx] != 1.0:
                    continue
                ref = slot_to_ref.get((i, g))
                if ref is None:
                    continue

                all_points.append({
                    "global_idx":      global_counter,
                    "owner_id":        owner_id,
                    "owner_local_idx": ref["owner_local_idx"],
                    "slot_idx":        slot_idx,  # CT 내 실제 위치
                    "grid_idx":        g,
                    "k_within_bucket": i,
                })
                global_counter += 1

    return all_points, global_counter


def build_owner_coord_map(
    owner_client_packs: List[dict],
) -> Dict[Tuple[int, int], list]:
    """
    슬롯 → 정규화 좌표 매핑 구성.

    ★ FinalClient 로컬 전용 ─ 서버에 절대 전달하지 않음 ★

    반환:
      {(owner_id, slot_idx): point_norm_list}
    """
    coord_map: Dict[Tuple[int, int], list] = {}
    for pack in owner_client_packs:
        owner_id = pack["owner_id"]
        for slot_idx, pt_norm in pack.get("slot_to_point_norm", {}).items():
            coord_map[(owner_id, slot_idx)] = pt_norm
    return coord_map


def reconstruct_results(
    engine,
    secret_key,
    final_ct,
    plain_all_points:  List[dict],
    owner_coord_map:   Optional[Dict[Tuple[int, int], list]] = None,
    global_min:        Optional[float] = None,
    scale_factor:      Optional[float] = None,
) -> List[dict]:
    """
    FHE 복호화 후 라벨 + (옵션) 좌표 복원.

    owner_coord_map / global_min / scale_factor 모두 제공 시 좌표 복원.
    미제공 시 라벨만 반환 (진정한 multiparty 시나리오).

    ★ 핵심 변경점:
      기존 코드는 global_idx 로 CT 슬롯을 읽었으나,
      컬럼-메이저 패킹에서는 slot_idx ≠ global_idx 이므로
      반드시 item["slot_idx"] 로 읽어야 함.
    """
    decrypted_raw = engine.decrypt(final_ct, secret_key)
    results: List[dict] = []

    for item in plain_all_points:
        slot_idx = item["slot_idx"]          # CT 내 실제 위치
        label    = int(round(decrypted_raw[slot_idx]))

        result = {
            "global_idx":      item["global_idx"],
            "owner_id":        item["owner_id"],
            "owner_local_idx": item["owner_local_idx"],
            "grid_idx":        item["grid_idx"],
            "label":           label,
        }

        # 좌표 복원 (FinalClient 로컬 coord_map 이용)
        if (owner_coord_map is not None
                and global_min is not None
                and scale_factor is not None):
            key     = (item["owner_id"], slot_idx)
            pt_norm = owner_coord_map.get(key)
            result["point_reconstructed"] = (
                [round(v * scale_factor + global_min, 6) for v in pt_norm]
                if pt_norm is not None else None
            )

        results.append(result)

    return results
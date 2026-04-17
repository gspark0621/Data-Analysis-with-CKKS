# core/client/FinalClient.py


def assign_global_indices(owner_client_blocks_list):
    """
    여러 DO의 client_blocks를 순회하며 실제 포인트에 전역 인덱스 부여.

    [수정] point_norm을 all_points에서 제거.
      이유: all_points가 다른 경로로 전달될 경우 실제 좌표 노출 위험.
      대신: build_owner_coord_map()으로 (owner_id, owner_local_idx) → point_norm
            매핑을 FinalClient 로컬에서만 별도 관리.

    입력:
      owner_client_blocks_list : prepare_and_encrypt_owner_blocks()가 반환한
                                  client_blocks의 리스트 (DO별 리스트의 리스트)
    반환:
      all_points    : 전역 인덱스 + 라우팅 정보만 포함 (좌표 없음)
      global_counter: 총 실제 포인트 수
    """
    global_counter = 0
    all_points = []

    for owner_blocks in owner_client_blocks_list:
        for blk in owner_blocks:
            for i, ref in enumerate(blk["point_refs"]):
                if ref is not None and blk["selection_mask"][i] == 1.0:
                    ref["global_idx"] = global_counter
                    all_points.append({
                        "global_idx":      global_counter,
                        "owner_id":        ref["owner_id"],
                        "owner_local_idx": ref["owner_local_idx"],
                        # [수정] point_norm 제거
                        #        좌표는 build_owner_coord_map에서 별도 관리
                        "grid_idx":        blk["grid_idx"],
                        "block_idx":       blk["block_idx"],
                    })
                    global_counter += 1

    return all_points, global_counter


def build_owner_coord_map(owner_client_blocks_list):
    """
    [신규] (owner_id, owner_local_idx) → point_norm 매핑 생성.

    assign_global_indices에서 point_norm을 제거한 대신,
    reconstruct_results에서 좌표를 역정규화할 때 이 맵을 사용.
    FinalClient 로컬에서만 사용되며 서버에 전달되지 않음.

    반환:
      dict: { (owner_id, owner_local_idx): point_norm }
    """
    coord_map = {}
    for owner_blocks in owner_client_blocks_list:
        for blk in owner_blocks:
            for i, ref in enumerate(blk["point_refs"]):
                if ref is not None and blk["selection_mask"][i] == 1.0:
                    key = (ref["owner_id"], ref["owner_local_idx"])
                    coord_map[key] = blk["points_norm"][i]
    return coord_map


def reconstruct_results(engine, secret_key, final_ct,
                        plain_all_points, owner_coord_map,
                        global_min, scale_factor):
    """
    FHE 라벨 암호문 복호화 후 원본 포인트와 매핑하여 결과 반환.

    [수정] point_norm 조회 방식 변경.
      기존: plain_all_points[i]["point_norm"] 직접 참조
      변경: owner_coord_map[(owner_id, owner_local_idx)] 조회
      이유: all_points에서 point_norm 제거 후 coord_map으로 분리 관리.

    입력:
      plain_all_points : assign_global_indices()의 반환값
      owner_coord_map  : build_owner_coord_map()의 반환값
      global_min       : 역정규화 기준 최솟값
      scale_factor     : global_max - global_min
    """
    decrypted_labels_raw = engine.decrypt(final_ct, secret_key)

    results = []
    for item in plain_all_points:
        gidx  = item["global_idx"]
        label = int(round(decrypted_labels_raw[gidx]))

        # [수정] coord_map에서 좌표 조회
        coord_key  = (item["owner_id"], item["owner_local_idx"])
        point_norm = owner_coord_map.get(coord_key)

        if point_norm is not None:
            point_reconstructed = [
                round((v * scale_factor) + global_min, 6)
                for v in point_norm
            ]
        else:
            point_reconstructed = None

        results.append({
            "global_idx":          gidx,
            "owner_id":            item["owner_id"],
            "owner_local_idx":     item["owner_local_idx"],
            "point_reconstructed": point_reconstructed,
            "grid_idx":            item["grid_idx"],
            "block_idx":           item["block_idx"],
            "label":               label,
        })

    return results
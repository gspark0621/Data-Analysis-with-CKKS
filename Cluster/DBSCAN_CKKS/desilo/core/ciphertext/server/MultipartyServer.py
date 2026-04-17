# core/ciphertext/server/MultipartyServer.py
import os
import glob
import tempfile
import gc

from core.ciphertext.server.Normalize import check_neighbor_closed_interval
from core.ciphertext.server.Core import identify_core_points_fhe_converted
from core.ciphertext.server.LabelPropagation import fhe_max_propagation_fhe, _refresh

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False


def _gpu_cleanup():
    gc.collect()
    if _has_torch:
        torch.cuda.empty_cache()


# ── 블록 CPU ↔ GPU 전송 헬퍼 ─────────────────────────────────────

# ── 블록 CPU ↔ GPU 전송 헬퍼 (in-place 버전) ──────────────────────

def _block_to_gpu(engine, blk: dict) -> None:
    """블록 내 모든 CT를 GPU로 이동 (in-place). 반환값 없음."""
    for c in blk["enc_coords"]:
        engine.to_cuda(c)
    engine.to_cuda(blk["enc_selection_mask"])


def _block_to_cpu(engine, blk: dict) -> None:
    """블록 내 모든 CT를 CPU로 이동 (in-place). 반환값 없음."""
    for c in blk["enc_coords"]:
        engine.to_cpu(c)
    engine.to_cpu(blk["enc_selection_mask"])


def _free_gpu_block(blk_gpu: dict) -> None:
    """GPU 블록 CT 즉시 해제. 비교 완료 직후 호출."""
    del blk_gpu["enc_coords"]
    del blk_gpu["enc_selection_mask"]


# ── adj_ct 스트리밍 (write_ciphertext / read_ciphertext) ─────────

class _AdjacencyStreamWriter:
    def __init__(self, engine, dir_path: str = None):
        self._engine  = engine
        self._dir     = dir_path or tempfile.mkdtemp(prefix="adj_cts_")
        self._created = (dir_path is None)
        self._index   = 0

    def write(self, ct) -> None:
        path = os.path.join(self._dir, f"{self._index}.ct")
        self._engine.write_ciphertext(ct, path)
        self._index += 1
        del ct

    def total_count(self) -> int:
        return self._index

    def path(self) -> str:
        return self._dir

    def close(self) -> None:
        pass

    def cleanup(self) -> None:
        if self._created:
            import shutil
            shutil.rmtree(self._dir, ignore_errors=True)


def _stream_load_chunks(engine, dir_path: str, chunk_size: int):
    files = sorted(
        glob.glob(os.path.join(dir_path, "*.ct")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    chunk = []
    for fpath in files:
        chunk.append(engine.read_ciphertext(fpath))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ── 블록 쌍 비교 ─────────────────────────────────────────────────

def compare_two_blocks(engine, keypack,
                       left_coords, left_mask_ct,
                       right_coords, right_mask_ct,
                       bucket_size, eps, dimension):
    """두 GPU 블록 간 거리 비교. 중간 CT 즉시 해제."""
    relin_key    = keypack.relinearization_key
    masked_left  = [engine.multiply(c, left_mask_ct,  relin_key) for c in left_coords]
    masked_right = [engine.multiply(c, right_mask_ct, relin_key) for c in right_coords]

    neighbor_ct_list         = []
    total_neighbor_from_pair = None

    for k in range(bucket_size):
        dist_sq_k = None
        for d in range(dimension):
            rotated   = engine.rotate(masked_right[d], keypack.rotation_key, k)
            diff      = engine.subtract(masked_left[d], rotated)
            sq        = engine.square(diff, relin_key)
            del rotated, diff
            dist_sq_k = sq if dist_sq_k is None else engine.add(dist_sq_k, sq)
            del sq

        adj_k = check_neighbor_closed_interval(
            engine, dist_sq_k, eps ** 2, keypack, dimension
        )
        del dist_sq_k

        neighbor_ct_list.append(adj_k)
        total_neighbor_from_pair = (
            adj_k if total_neighbor_from_pair is None
            else engine.add(total_neighbor_from_pair, adj_k)
        )

    del masked_left, masked_right
    return total_neighbor_from_pair, neighbor_ct_list


# ── 메인 클러스터링 ───────────────────────────────────────────────

def run_multiparty_point_dbscan(engine, keypack,
                                encrypted_server_blocks_list,
                                grid_centers_norm,
                                query_epsilon_norm, base_epsilon_norm,
                                min_pts, bucket_size, max_blocks_per_grid,
                                total_points_upper_bound,
                                adj_chunk_size: int = 2000,
                                adj_stream_dir: str = None):
    """
    GPU 메모리 관리 전략:
      - 모든 블록 CT: CPU RAM 상주 (to_cpu 완료 상태로 전달받음)
      - 비교 시: left 블록 to_cuda → 전체 right 순회 → left to_gpu 해제
                right 블록: 매 쌍마다 to_cuda → 비교 후 즉시 해제
      - 동시 GPU 점유: left(1) + right(1) + 중간 CT = 최소화
      - running_sum: GPU 상주 (1개만 유지)
      - adj_ct: write_ciphertext 즉시 디스크 → GPU 해제
    """
    N         = total_points_upper_bound
    dimension = len(grid_centers_norm[0])

    # 모든 owner의 CPU 블록을 flat list로 통합
    all_blocks_cpu = []
    for server_blocks in encrypted_server_blocks_list:
        all_blocks_cpu.extend(server_blocks)

    total_pairs = len(all_blocks_cpu) ** 2
    running_sum = None
    writer      = _AdjacencyStreamWriter(engine, dir_path=adj_stream_dir)

    try:
        pair_count = 0

        for i, left_blk in enumerate(all_blocks_cpu):

            # left: CPU → GPU (in-place, 원본 dict 수정됨)
            _block_to_gpu(engine, left_blk)

            for right_blk in all_blocks_cpu:

                # right: CPU → GPU (in-place)
                _block_to_gpu(engine, right_blk)

                pair_sum_ct, pair_adj_list = compare_two_blocks(
                    engine=engine, keypack=keypack,
                    left_coords=left_blk["enc_coords"],
                    left_mask_ct=left_blk["enc_selection_mask"],
                    right_coords=right_blk["enc_coords"],
                    right_mask_ct=right_blk["enc_selection_mask"],
                    bucket_size=bucket_size,
                    eps=query_epsilon_norm,
                    dimension=dimension,
                )

                # right: GPU → CPU (in-place, 즉시 반환)
                _block_to_cpu(engine, right_blk)

                if running_sum is None:
                    running_sum = pair_sum_ct
                else:
                    running_sum = engine.add(running_sum, pair_sum_ct)
                    del pair_sum_ct

                for adj_ct in pair_adj_list:
                    writer.write(adj_ct)
                del pair_adj_list

                pair_count += 1
                if pair_count % 5 == 0:
                    _gpu_cleanup()
                    print(f"  └─ [{pair_count}/{total_pairs}] 쌍 처리 완료")

            # left: GPU → CPU (in-place, 이 행 끝나면 반환)
            _block_to_cpu(engine, left_blk)
            _gpu_cleanup()

        writer.close()
        adj_path = writer.path()

        # neighbor count 완성 (+1: 자기 자신 포함)
        ones_pt            = engine.encode([1.0] * N)
        total_neighbors_ct = engine.add(running_sum, ones_pt)
        del running_sum

        core_ct = identify_core_points_fhe_converted(
            engine, total_neighbors_ct, min_pts, N, keypack
        )

        cluster_id_pt = [(i + 1) / float(N + 1) for i in range(N)]
        final_norm_ct = None

        for chunk in _stream_load_chunks(engine, adj_path, adj_chunk_size):
            if final_norm_ct is None:
                final_norm_ct = fhe_max_propagation_fhe(
                    engine, keypack, chunk, core_ct,
                    cluster_id_pt, N, max_iter=5
                )
            else:
                final_norm_ct = fhe_max_propagation_fhe(
                    engine, keypack, chunk, core_ct,
                    cluster_id_pt, N, max_iter=5,
                    init_label_ct=final_norm_ct
                )
            del chunk

    finally:
        writer.cleanup()

    scale_back_pt = engine.encode([float(N + 1)] * N)
    final_ct      = engine.multiply(final_norm_ct, scale_back_pt)
    return _refresh(engine, final_ct, keypack)
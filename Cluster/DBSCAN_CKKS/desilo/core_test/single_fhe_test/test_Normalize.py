import os
import numpy as np
from time import time
from desilofhe import Engine
from util.keypack import KeyPack

from core.EncryptModule import DimensionalEncryptor
from core.Label_Propagation import fhe_circular_shift
from core.Normalize import (
    check_neighbor_closed_interval,
    check_neighbor_closed_interval_heaviside9,
)

DATASET_PATH = "/home/junhyung/study/Data_Analysis_with_CKKS/Cluster/DBSCAN_CKKS/desilo/dataset/Other_cluster/hepta.arff"


def load_arff_to_pts_with_labels(filepath: str):
    pts = []
    true_labels = []
    data_section = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@data'):
                data_section = True
                continue
            if data_section:
                line = line.replace('\t', ' ').replace(',', ' ')
                values = line.split()
                if len(values) < 2:
                    continue
                row = [float(v) for v in values[:-1]]
                label = int(float(values[-1]))
                pts.append(row)
                true_labels.append(label)

    if not pts:
        raise ValueError("데이터를 찾을 수 없습니다. 파일 포맷을 확인해주세요.")
    return np.array(pts, dtype=np.float64), np.array(true_labels, dtype=int)


def plaintext_neighbor_mask(normalized_pts, normalized_eps, k, margin_val=0.05):
    pts = np.array(normalized_pts, dtype=np.float64)
    rotated = np.roll(pts, -k, axis=0)
    dist_sq = np.sum((pts - rotated) ** 2, axis=1)
    threshold = normalized_eps ** 2 + margin_val
    raw = dist_sq - threshold
    gt = (raw <= 0).astype(np.float64)
    return dist_sq, raw, gt


def build_dist_sq_ct(engine, keypack, encrypted_columns, N, dimension, k):
    dist_sq_k = None
    for d in range(dimension):
        base_col = encrypted_columns[d]
        rotated_col = fhe_circular_shift(engine, base_col, k, N, keypack)
        diff_ct = engine.subtract(base_col, rotated_col)
        sq_ct = engine.square(diff_ct, keypack.relinearization_key)

        if dist_sq_k is None:
            dist_sq_k = sq_ct
        else:
            dist_sq_k = engine.add(dist_sq_k, sq_ct)
    return dist_sq_k


def summarize_errors(name, dec, gt):
    err = np.abs(dec - gt)
    dec_bin = (dec >= 0.5).astype(np.float64)
    acc = (dec_bin == gt).mean()

    print(f"[{name}]")
    print(f"  mean abs err : {err.mean():.6f}")
    print(f"  max  abs err : {err.max():.6f}")
    print(f"  binary acc   : {acc * 100:.2f}%")
    return err, acc


def main():
    print("==================================================")
    print("      Hepta Normalize: old vs heaviside9 test     ")
    print("==================================================\n")

    pts, true_labels = load_arff_to_pts_with_labels(DATASET_PATH)
    N = len(pts)
    dimension = pts.shape[1]

    print(f"▶ 데이터셋 로드 완료: N={N}, dimension={dimension}")

    eps_val = float(input("eps 값을 입력하세요 (예: 0.5) > "))

    global_min = np.min(pts)
    global_max = np.max(pts)
    scale_factor = global_max - global_min if (global_max - global_min) != 0.0 else 1.0

    normalized_pts = ((pts - global_min) / scale_factor).tolist()
    normalized_eps = eps_val / scale_factor

    print(f"▶ 정규화 완료: normalized_eps = {normalized_eps:.6f}\n")

    print("FHE 엔진 및 키 생성 중...")
    engine = Engine(use_bootstrap=True, mode="gpu")
    secret_key = engine.create_secret_key()
    keypack = KeyPack(
        public_key=engine.create_public_key(secret_key),
        rotation_key=engine.create_rotation_key(secret_key),
        relinearization_key=engine.create_relinearization_key(secret_key),
        conjugation_key=engine.create_conjugation_key(secret_key),
        bootstrap_key=engine.create_bootstrap_key(secret_key),
    )

    print("데이터 암호화 중...")
    encryptor = DimensionalEncryptor(engine, keypack)
    encrypted_columns = encryptor.encrypt_data(normalized_pts, dimension)

    test_ks = [1, 2, 3, N - 1] if N > 4 else list(range(1, N))

    total_old_err = []
    total_new_err = []

    bound_shrink_list = [1.0, 0.75, 0.5, 0.35, 0.25]

    for k in test_ks:
        print("\n--------------------------------------------------")
        print(f"▶ k = {k}")
        print("--------------------------------------------------")

        dist_sq_ct = build_dist_sq_ct(engine, keypack, encrypted_columns, N, dimension, k)

        old_ct = check_neighbor_closed_interval(
            engine=engine,
            dist_sq_ct=dist_sq_ct,
            eps_sq=normalized_eps ** 2,
            keypack=keypack,
            dimension=dimension,
        )
        old_dec = np.array(engine.decrypt(old_ct, secret_key)[:N], dtype=np.float64)

        dist_sq_plain, raw_plain, gt = plaintext_neighbor_mask(
            normalized_pts=normalized_pts,
            normalized_eps=normalized_eps,
            k=k,
            margin_val=0.05,
        )

        print("[old baseline]")
        summarize_errors("old", old_dec, gt)

        for shrink in bound_shrink_list:
            new_ct = check_neighbor_closed_interval_heaviside9(
                engine=engine,
                dist_sq_ct=dist_sq_ct,
                eps_sq=normalized_eps ** 2,
                keypack=keypack,
                dimension=dimension,
                bound_shrink=shrink,
            )
            new_dec = np.array(engine.decrypt(new_ct, secret_key)[:N], dtype=np.float64)

            print(f"\n[new_heaviside9 | bound_shrink={shrink}]")
            err, _ = summarize_errors(f"new_heaviside9_{shrink}", new_dec, gt)

            print("idx | raw(dist^2-th) | gt | new_dec | err_new")
            for i in range(min(5, N)):
                print(
                    f"{i:3d} | {raw_plain[i]: .6f} | {gt[i]:.0f} | "
                    f"{new_dec[i]: .4f} | {err[i]: .4f}"
                )


        total_old_err = np.concatenate(total_old_err)
        total_new_err = np.concatenate(total_new_err)

        print("\n==================================================")
        print("                    최종 요약                      ")
        print("==================================================")
        print(f"[old] mean abs err = {total_old_err.mean():.6f}, max abs err = {total_old_err.max():.6f}")
        print(f"[new] mean abs err = {total_new_err.mean():.6f}, max abs err = {total_new_err.max():.6f}")


if __name__ == '__main__':
    main()

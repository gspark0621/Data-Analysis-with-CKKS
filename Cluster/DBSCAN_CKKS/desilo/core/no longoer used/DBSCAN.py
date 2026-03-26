# -*- coding: utf-8 -*-
import numpy as np
from desilofhe import engine
from .Euclidean_dist import Euclidean_Distance_ct
from .Comp import Comp

UNCLASSIFIED = False
NOISE = None

class EncryptedDBSCANProcessor:
    def __init__(self, context, eps, min_points):
        self.ctx = context
        self.eps = eps
        self.min_points = min_points

    def _eps_neighborhood(self, p_enc, q_enc):
        dist = self.Euclidean_Distance_ct(p_enc, q_enc)
        comp_result = engine.subtract(dist, self.eps)
        return Comp(engine, comp_result, 0, 5, 5, 6, 2, self.ctx.keypack.relinearization_key, self.ctx.keypack.conjugation_key, self.ctx.keypack.bootstrap_key)

    def _region_query(self, encrypted_points, point_id):
        seeds = []
        #len 함수는 임시완료(desilofhe 팁.ipynb 참고)
        n_points = len(encrypted_points*engine.slot_count)

        for i in range(n_points):
            # 자기 자신도 포함
            x = self._eps_neighborhood(encrypted_points[point_id], encrypted_points[i])
            x and seeds.append(i)
        return seeds

    def _expand_cluster(self, encrypted_points, classifications, point_id, cluster_id):
        seeds = self._region_query(encrypted_points, point_id)
        if len(seeds) < self.min_points:
            classifications[point_id] = 'noise'
            return False
        else:
            classifications[point_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id

            while len(seeds) > 0:
                current_point = seeds[0]
                results = self._region_query(encrypted_points, current_point)
                if len(results) >= self.min_points:
                    for result_point in results:
                        if classifications[result_point] in ['unclassified', 'noise']:
                            if classifications[result_point] == 'unclassified':
                                seeds.append(result_point)
                            classifications[result_point] = cluster_id
                seeds = seeds[1:]
            return True

    def run_dbscan(self, encrypted_points):
        cluster_id = 1
        n_points = len(encrypted_points)
        classifications = ['unclassified'] * n_points

        for point_id in range(n_points):
            if classifications[point_id] == 'unclassified':
                if self._expand_cluster(encrypted_points, classifications, point_id, cluster_id):
                    cluster_id += 1
        return classifications, cluster_id

    def get_plaintext_results(self, encrypted_points, labels):
        results = []
        for idx, label in enumerate(labels):
            lat, lon = self._decrypt_point(encrypted_points[idx])
            results.append((lat, lon, label))
        return results



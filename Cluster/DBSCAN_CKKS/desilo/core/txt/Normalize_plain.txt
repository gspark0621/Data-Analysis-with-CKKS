# core/plain/Normalize_plain.py
def is_neighbor_plain(p, q, eps):
    dist_sq = sum((a - b) ** 2 for a, b in zip(p, q))
    return 1 if dist_sq <= eps ** 2 else 0
# core/plain/Core_plain.py
def identify_core_points_plain(neighbor_counts, min_pts):
    core_mask = []
    for c in neighbor_counts:
        core_mask.append(1 if c >= min_pts else 0)
    return core_mask
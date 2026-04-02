# core/plain/Label_Propagation_plain.py
from collections import deque


def build_core_components_plain(adjacency_list, core_mask, num_points):
    labels = [-1] * num_points
    current_label = 0

    for i in range(num_points):
        if core_mask[i] != 1 or labels[i] != -1:
            continue

        current_label += 1
        q = deque([i])
        labels[i] = current_label

        while q:
            u = q.popleft()
            for v in adjacency_list[u]:
                if core_mask[v] == 1 and labels[v] == -1:
                    labels[v] = current_label
                    q.append(v)

    return labels


def assign_border_points_plain(adjacency_list, core_mask, labels):
    final_labels = labels[:]
    num_points = len(labels)

    for i in range(num_points):
        if core_mask[i] == 1:
            continue

        candidate_labels = []
        for j in adjacency_list[i]:
            if core_mask[j] == 1 and labels[j] != -1:
                candidate_labels.append(labels[j])

        if candidate_labels:
            final_labels[i] = min(candidate_labels)

    return final_labels


def run_label_propagation_plain(adjacency_list, core_mask, num_points):
    core_labels = build_core_components_plain(adjacency_list, core_mask, num_points)
    final_labels = assign_border_points_plain(adjacency_list, core_mask, core_labels)
    return final_labels
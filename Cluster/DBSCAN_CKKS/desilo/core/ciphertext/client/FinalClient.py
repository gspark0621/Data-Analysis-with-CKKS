# core/client/FinalClient.py
def decrypt_cluster_labels(engine, secret_key, encrypted_cluster_result, total_points_upper_bound, original_points_flat):
    decrypted_labels = engine.decrypt(encrypted_cluster_result, secret_key)

    cluster_labels = []
    for x in decrypted_labels[:len(original_points_flat)]:
        r = round(x)
        if r <= 0:
            cluster_labels.append(-1)
        elif r > total_points_upper_bound:
            cluster_labels.append(total_points_upper_bound)
        else:
            cluster_labels.append(r)

    result_pts = []
    for i in range(len(original_points_flat)):
        result_pts.append(list(original_points_flat[i]) + [cluster_labels[i]])

    return result_pts, cluster_labels
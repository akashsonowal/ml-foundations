import math

def predict_label(examples, features, k, label_key="is_intrusive"):
    k_nearest_neighbors = find_k_nearest_neighbors(examples, features, k)
    k_nearest_neighbors_labels = [examples[pid][label_key] for pid in k_nearest_neighbors]
    return round(sum(k_nearest_neighbors_labels) / k)

def find_k_nearest_neighbors(examples, features, k):
    distances = {}
    for pid, features_label_map in examples.items():
        distance = get_euclidean_distance(features, features_label_map["features"])
        distances[pid] = distance
    return sorted(distances, key=distances.get)[:k]

def get_euclidean_distance(features, other_features):
    squared_differences = []
    for i in range(len(features)):
        squared_differences.append((other_features[i] - features[i]) ** 2)
    return math.sqrt(sum(squared_differences))

import torch

def compute_recall(query_features, gallery_features, k_list=["Recall@1", "Recall@2", "Recall@4", "Recall@8"]):
    """
    Example implementation: Computes Recall@K (assumes query and gallery are the same).
    :param query_features: Tensor of shape (N, D)
    :param gallery_features: Tensor of shape (N, D)
    :param k_list: List of recall metrics to compute
    :return: Dictionary mapping each k to the recall percentage
    """
    # Compute pairwise Euclidean distance matrix
    distances = torch.cdist(query_features, gallery_features, p=2)
    # Set the diagonal (self-distance) to max value to ignore it
    for i in range(distances.size(0)):
        distances[i, i] = distances.max()
    
    recall = {}
    for k_str in k_list:
        k = int(k_str.split('@')[1])
        _, indices = distances.topk(k, largest=False)
        correct = 0
        for i in range(distances.size(0)):
            if i in indices[i]:
                correct += 1
        recall[k_str] = 100.0 * correct / distances.size(0)
    return recall

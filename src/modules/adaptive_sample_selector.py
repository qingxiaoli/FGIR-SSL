import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSampleSelector(nn.Module):
    def __init__(self, feature_extractor, sigma=16, Tmax=800, device='cuda'):
        """
        :param feature_extractor: A lightweight network for feature extraction (e.g., ViT-S/16)
        :param sigma: Difficulty adjustment hyperparameter
        :param Tmax: Maximum number of iterations
        :param device: Device to use
        """
        super(AdaptiveSampleSelector, self).__init__()
        self.feature_extractor = feature_extractor
        self.sigma = sigma
        self.Tmax = Tmax
        self.current_iteration = 0
        self.device = device

    def update_gamma(self):
        # Update the difficulty factor gamma based on the current iteration
        gamma = self.sigma ** (self.Tmax / (self.current_iteration + 1))
        return gamma

    def forward(self, anchor_images, candidate_images):
        """
        :param anchor_images: Anchor images (N, C, H, W)
        :param candidate_images: Candidate images (K, C, H, W)
        :return: Returns the original anchors and sampled negative samples
        """
        anchor_features = self.feature_extractor(anchor_images)  # (N, feature_dim)
        candidate_features = self.feature_extractor(candidate_images)  # (K, feature_dim)
        
        # Construct similarity matrix (N, K)
        similarity = torch.matmul(anchor_features, candidate_features.t())
        gamma = self.update_gamma()
        
        # Compute sampling weights via row-wise softmax
        weights = F.softmax(gamma * similarity, dim=1)
        # Sample one negative for each anchor
        selected_indices = torch.multinomial(weights, num_samples=anchor_images.size(0), replacement=True)
        negative_samples = candidate_images[selected_indices]
        
        self.current_iteration += 1
        return anchor_images, negative_samples

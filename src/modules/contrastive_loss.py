import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(F.relu(pos_distance - neg_distance + self.margin))
        return loss

class IntraImageContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(IntraImageContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, shared_repr, auxiliary_repr):
        logits = torch.mm(shared_repr, auxiliary_repr.t()) / self.temperature
        labels = torch.arange(shared_repr.size(0)).to(shared_repr.device)
        loss = F.cross_entropy(logits, labels)
        return loss

import torch
from src.modules.contrastive_loss import TripletContrastiveLoss, IntraImageContrastiveLoss
from src.modules.generative_loss import ReconstructionLoss

def test_triplet_loss():
    loss_fn = TripletContrastiveLoss(margin=1.0)
    anchor = torch.randn(4, 128)
    positive = torch.randn(4, 128)
    negative = torch.randn(4, 128)
    loss = loss_fn(anchor, positive, negative)
    assert loss.item() >= 0

def test_reconstruction_loss():
    loss_fn = ReconstructionLoss()
    original = torch.randn(4, 3, 224, 224)
    reconstructed = torch.randn(4, 3, 224, 224)
    mask = torch.ones_like(original)
    loss = loss_fn(original, reconstructed, mask)
    assert loss.item() >= 0

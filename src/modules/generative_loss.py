import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, original, reconstructed, mask):
        # Compute MSE only on masked regions
        loss = self.criterion(reconstructed * mask, original * mask)
        return loss

import torch
import torch.nn as nn
import timm  # Use timm to load ViT models

class SelfSupervisedModel(nn.Module):
    def __init__(self, backbone="vit", pretrained=True):
        super(SelfSupervisedModel, self).__init__()
        if backbone == "vit":
            # Load ViT-B/16 model
            self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            self.encoder.head = nn.Identity()  # Remove classification head
            feature_dim = self.encoder.embed_dim
        elif backbone == "resnet":
            # Use ResNet18 as an example
            from torchvision import models
            self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError("Unsupported backbone!")
        
        # A simple decoder mapping features back (example implementation)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
        )
    
    def forward(self, x):
        shared_repr = self.encoder(x)
        reconstructed = self.decoder(shared_repr)
        return shared_repr, reconstructed

import torch
from src.model import SelfSupervisedModel

def test_model_forward():
    model = SelfSupervisedModel(backbone="resnet", pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    shared_repr, reconstructed = model(dummy_input)
    assert shared_repr.shape[0] == 4
    assert reconstructed.shape[0] == 4

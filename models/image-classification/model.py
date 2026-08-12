import torch
from torchvision.models import resnet18, ResNet18_Weights


def create_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model

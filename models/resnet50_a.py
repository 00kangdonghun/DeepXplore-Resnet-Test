from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def get_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, 10)  # For CIFAR-10
    return model
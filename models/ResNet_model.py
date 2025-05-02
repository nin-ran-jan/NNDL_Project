import torch
import torchvision.models as models

def get_resnet_model():
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*list(resnet18.children())[:-1])

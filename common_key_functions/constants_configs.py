import torch
from torchvision import models


MODEL_CONFIGS = {
    "regnet_x_3_2": {
        "batch_size": 16,
        "builder": models.regnet_x_3_2gf,
        "weights": models.RegNet_X_3_2GF_Weights.DEFAULT,
        "feature_attr": "trunk_output"
    },
    "regnet_x_16": {
        "batch_size": 4,
        "builder": models.regnet_x_16gf,
        "weights": models.RegNet_X_16GF_Weights.DEFAULT,
        "feature_attr": "trunk_output"
    },
    "regnet_x_32": {
        "batch_size": 2,
        "builder": models.regnet_x_32gf,
        "weights": models.RegNet_X_32GF_Weights.DEFAULT,
        "feature_attr": "trunk_output"
    },
    "resnet50": {
        "batch_size": 12,
        "builder": models.resnet50,
        "weights": models.ResNet50_Weights.DEFAULT,
        "feature_attr": "layer4"
    },
    "resnet18": {
        "batch_size": 24,
        "builder": models.resnet18,
        "weights": models.ResNet18_Weights.DEFAULT,
        "feature_attr": "layer4"
    }
}

MEAN_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_CONSTANTS = {
    "mean": MEAN_IMAGENET,
    "std":  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
    "mean_awb": MEAN_IMAGENET.mean(dim=1, keepdim=True),
    "size_resize": 256,
    "size_center_crop": 224
}



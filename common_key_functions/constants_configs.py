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
    },
    "wide_resnet101_2": {
        "batch_size": 3,
        "builder": models.wide_resnet101_2,
        "weights": models.Wide_ResNet101_2_Weights.IMAGENET1K_V1,
        "feature_attr": "layer4"
    },
    "efficientnet_v2_l": {
        "batch_size": 3,
        "builder": models.efficientnet_v2_l,
        "weights": models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        "feature_attr": "features"
    },
    "convnext_l": {
        "batch_size": 1,
        "builder": models.convnext_large,
        "weights": models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
        "feature_attr": "features"
    },
    "swin_v2_b": {
        "batch_size": 3,
        "builder": models.swin_v2_b,
        "weights": models.Swin_V2_B_Weights.IMAGENET1K_V1,
        "feature_attr": "features"
    },
    "vit_l_16": {
        "batch_size": 1,
        "builder": models.vit_l_16,
        "weights": models.ViT_L_16_Weights.IMAGENET1K_V1,
        "feature_attr": "encoder"       # output: (B, n_tokens, hidden_dim)
    },
}


MEAN_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_CONSTANTS = {
    "mean": MEAN_IMAGENET,
    "std":  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
    "mean_awb": MEAN_IMAGENET.mean(dim=1, keepdim=True),
    "size_resize": 256,
    "size_center_crop": 224
}



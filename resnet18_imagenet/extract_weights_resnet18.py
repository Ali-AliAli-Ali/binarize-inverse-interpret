from math import sqrt
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms, utils


def fuse_avgpool_linear(
    model,
    input_size=(1, 3, 224, 224),  # ImageNet default input size
    classifier_attr='fc',         # torchvision uses 'fc' for the final Linear
    device='cpu'
):
    """
    Fuse an AvgPool2d followed by a Linear into one Linear layer.

    Args:
        model:           PyTorch model instance (ResNet18 ImageNet from torchvision.models).
        input_size:      Input shape (default ImageNet size: 224x224).
        classifier_attr: Attribute name of final Linear ('fc' in torchvision).
        device:          Device to perform computations on ('cuda' if is available else 'cpu')

    Returns:
        W_fused: (C_out, C_in * H * W) fused weight tensor.
        b_fused: (C_out,) fused bias tensor.
    """
    # Extract classifier (fc layer)
    classifier = getattr(model, classifier_attr)
    avgpool = model.avgpool

    # Determine divisor H*W for pooling
    if isinstance(avgpool, torch.nn.AdaptiveAvgPool2d):
        # Forward pass through the model up to avgpool to get spatial dims
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(input_size).to(device)
            feats = model.conv1(dummy)
            feats = model.bn1(feats)
            feats = model.relu(feats)
            feats = model.maxpool(feats)
            
            feats = model.layer1(feats)
            feats = model.layer2(feats)
            feats = model.layer3(feats)
            feats = model.layer4(feats)  # Get features right before avgpool
            
            feats = avgpool(feats)
            H, W = feats.shape[-2:] if feats.dim() == 4 else (1, 1)
        
        divisor = H * W
    else:
        raise ValueError(f"Unsupported pooling layer: {avgpool}")

    # Get original Linear parameters
    W = classifier.weight.data       # shape (1000, 512) for ResNet18
    b = classifier.bias.data if classifier.bias is not None else torch.zeros(W.shape[0])

    # Build fused weight: repeat each channel weight over spatial positions and normalize
    W_fused = W.repeat(1, divisor) / divisor
    b_fused = b.clone()

    return W_fused, b_fused

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 1. Load ResNet18 pretrained on ImageNet from torchvision.models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()

    # 2. Fuse linear layers Save to CSV
    W_fused, b_fused = fuse_avgpool_linear(model)

    # Convert to homogeneous form by appending bias as an extra column
    # Resulting shape: (num_classes, channels*H*W + 1)
    W_hom = torch.cat([W_fused, b_fused.unsqueeze(1)], dim=1)

    # Save homogeneous weight matrix to CSV
    np.savetxt('fused_weight_homogeneous.csv', W_hom.cpu().numpy(), delimiter=',')

    print('Fused weights saved to fused_weight_homogeneous.csv')


if __name__ == '__main__':
    main()

from math import sqrt
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from pytorchcv.model_provider import get_model


def fuse_avgpool_linear(
    model,
    input_size=(1, 3, 32, 32),
    trunk_attr='features',
    pool_idx=-1,
    classifier_attr='output',
):
    """
    Fuse an AvgPool2d followed by a Linear into one Linear layer.

    Args:
        model:           PyTorch model instance (ResNet20 CIFAR10 from pytorchcv).
        input_size:      Dummy input shape to infer adaptive pooling output.
        trunk_attr:      Attribute name of the feature extractor (nn.Sequential).
        pool_idx:        Index of the AvgPool2d layer in the trunk sequence.
        classifier_attr: Attribute name of the final Linear classifier.

    Returns:
        W_fused: (C_out, C_in * H * W) fused weight tensor.
        b_fused: (C_out,) fused bias tensor.
    """
    # Extract feature trunk and classifier
    trunk = getattr(model, trunk_attr)
    classifier = getattr(model, classifier_attr)

    # Locate the pooling layer
    avgpool = trunk[pool_idx]

    # Determine the divisor H*W for pooling
    if isinstance(avgpool, torch.nn.AdaptiveAvgPool2d):
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(input_size)
            feats = trunk(dummy)
            H, W = feats.shape[-2:]
        divisor = H * W
    elif isinstance(avgpool, torch.nn.AvgPool2d):
        k = avgpool.kernel_size
        if isinstance(k, tuple):
            divisor = k[0] * k[1]
        else:
            divisor = k * k
    else:
        raise ValueError(f"Unsupported pooling layer: {avgpool}")

    # Get original Linear parameters
    W = classifier.weight.data       # shape (num_classes, channels)
    b = classifier.bias.data if classifier.bias is not None else torch.zeros(W.shape[0])

    # Build fused weight: repeat each channel weight over spatial positions and normalize
    W_fused = W.repeat(1, divisor) / divisor
    b_fused = b.clone()

    return W_fused, b_fused

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load ResNet20 pretrained on CIFAR-100 from torchcv
    model = get_model("resnet20_cifar100", pretrained=True)

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

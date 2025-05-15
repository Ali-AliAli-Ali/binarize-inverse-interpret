import argparse
from math import sqrt
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms, utils


def replace_relu_with_softplus(model: nn.Module, beta: float = 1.0, threshold: float = 20.0):
    """
    Recursively replace all nn.ReLU in `module` (and its children)
    with `nn.Softplus(beta=beta, threshold=threshold)`.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            # replace in-place
            setattr(model, name, nn.Softplus(beta=beta, threshold=threshold))
        else:
            # recurse into child modules
            replace_relu_with_softplus(child, beta=beta, threshold=threshold)

def tv_loss(f):
    """
    Calculate sum of neighbouring pixels absolute difference over height and width of picture (`L_1` gradient norm)
    """
    # f: (B, C, H, W)
    dh = f[:, :, 1:, :] - f[:, :, :-1, :]   # shape (B,C,H-1,W)
    dw = f[:, :, :, 1:] - f[:, :, :, :-1]   # shape (B,C,H,W-1)
    return dh.abs().sum() + dw.abs().sum()

def tv_feature_loss(f):
    """
    Calculate sum of neighbouring pixels absolute difference over height and width of picture (`L_1` gradient norm)
    """
    # f: (B, C, H, W)
    # whiten
    f = (f - f.mean(dim=[0,2,3], keepdim=True)) / (f.std(dim=[0,2,3], keepdim=True) + 1e-5)
    # compute spatial diffs
    dh = f[:, :, 1:, :-1] - f[:, :, :-1, :-1]  # → (B, C, H-1, W-1)
    dw = f[:, :, :-1, 1:] - f[:, :, :-1, :-1]  # → (B, C, H-1, W-1)
    # squared magnitude of the C-dimensional gradient vector
    grad2 = dh.pow(2).sum(dim=1) + dw.pow(2).sum(dim=1)  # (B, H-1, W-1)
    # isotropic vector‐TV
    return torch.sqrt(grad2 + 1e-12).sum()

def normalize_batch_to_unit_range(batch: torch.Tensor) -> torch.Tensor:
    """
    Normalize each image in a batch from its [min, max] range to [0, 1].

    Args:
        batch: tensor of shape (B, C, H, W)

    Returns:
        tensor of same shape with values in [0, 1]
    """
    # Compute per‐image min and max over C, H, W
    mins = batch.quantile(0.01)  # shape (B,1,1,1)
    maxs = batch.quantile(0.99)  # shape (B,1,1,1)
    denom = (maxs - mins).clamp(min=1e-6)          # avoid divide‐by‐zero
    return (batch - mins) / denom



def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('class_id', help='ImageNet class id', type=int, default=0)
    # args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    class_id = 0 # args.class_id
    batch_size = 256
    num_steps = 1000  # 2000-5000
    sigma = 1e-2      # 1e-3
    lr = 0.1
    beta = 10.
    l_tv = 1e-5
    l_tv_f = 1e-6
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load ResNet18 pretrained on ImageNet from torchvision.models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    layers = {
        'target': model,          # fc output
        'layer1': model.layer1,   # activations after layer1
        'layer2': model.layer2,   # activations after layer2
        'layer3': model.layer3,   # activations after layer3
        'layer4': model.layer4,   # activations after layer4
    }

    # before
    print("ReLUs:", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    # swap them all, setting beta=2.0 (for instance)
    replace_relu_with_softplus(model, beta=beta)

    # after
    print("ReLUs:", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    model = model.to(device).eval()

    # 1b. Register hooks on desired layers
    activation = {}

    for name, layer in layers.items():
        layer.register_forward_hook(
            lambda module, inp, out, name=name: activation.__setitem__(name, out)
        )

    # 2. Load ImageNet dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    imagenet = datasets.ImageFolder(root='./imagenet', train=True, transform=transform)
    # imagenet = datasets.ImageNet(root='./imagenet', train=True, transform=transform)

    # Filter dataset for chosen class_id
    indices = [i for i, (_, y) in enumerate(imagenet) if y == class_id]
    if not len(indices):
        raise ValueError(f"No images found for class_id={class_id}!")
    subset = Subset(imagenet, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # 3. Take first batch from the dataset
    img_raw_batch, _ = next(iter(loader))
    img_raw_batch = img_raw_batch.to(device)  # shape: [B, 3, 224, 224]

    utils.save_image(img_raw_batch, f'{output_dir}/orig_{class_id:04d}.png', nrow=int(sqrt(batch_size)))

    # 4. Normalize using ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_norm = (img_raw_batch - mean) / std

    # 5. Compute and fix the target activation via hook
    #with torch.no_grad():
    #    _ = model(img_norm)
    #    target_act = activation['target'].detach()
    # 5b. Prepare constant target_act as one-hot true-label logits
    num_classes = model.fc.out_features 
    target_act = torch.zeros(batch_size, num_classes, device=device)
    target_act[:, class_id] = 1.

    # 6. Invert the network for the whole batch via SGD + regularization + jitter
    #x0 = torch.randint_like(img_norm, 2).mul_(2.).sub_(1.)
    x0 = torch.randn_like(img_norm)
    x = (sigma * x0).requires_grad_()

    optimizer = optim.Adam([x], lr=lr)

    for step in range(0, num_steps + 1):
        optimizer.zero_grad()

        # Forward through full model to populate hook
        _ = model(x)
        f1 = activation['layer1']
        f2 = activation['layer2']
        f3 = activation['layer3']
        f4 = activation['layer4']
        act = activation['target']

        # Compute losses
        loss = F.mse_loss(act, target_act)
        total_loss = loss + \
            l_tv * tv_loss(x) + \
            l_tv_f * (tv_feature_loss(f1) + tv_feature_loss(f2) + tv_feature_loss(f3) + tv_feature_loss(f4))

        # Backpropagation and update
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}/{num_steps}, loss: {loss.item():.4f}")
            # 7. Denormalize and save reconstructed batch as a grid
            x_denorm = normalize_batch_to_unit_range(x)
            utils.save_image(x_denorm, f'{output_dir}/batch_iter_{step:04d}.png', nrow=int(sqrt(batch_size)))

    print(f"Optimization complete. Results saved in {output_dir}/")

if __name__ == '__main__':
    main()

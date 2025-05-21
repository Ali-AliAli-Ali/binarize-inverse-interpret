import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.decomposition import TruncatedSVD


def make_val_loader(val_dir: str, batch_size: int = 256, num_workers: int = 4):
    """
    Create ImageNet validation DataLoader with required transforms.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_ds = datasets.ImageFolder(os.path.join(val_dir, 'val'), transform=val_transforms)
    return DataLoader(val_ds, 
                      batch_size=batch_size, 
                      shuffle=False,
                      num_workers=num_workers, 
                      pin_memory=True)

@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """
    Runs validation and returns top-1 and top-5 accuracy percentages.
    """
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        # get top-5 predictions
        _, preds = logits.topk(5, dim=1, largest=True, sorted=True)

        # compare with targets
        # preds: [batch,5]  targets: [batch]
        correct = preds.eq(targets.view(-1, 1))
        top1_correct += correct[:, :1].sum().item()
        top5_correct += correct.sum().item()
        total += targets.size(0)

    top1 = top1_correct / total * 100.0
    top5 = top5_correct / total * 100.0
    return top1, top5

def binarize_hook(module, input):
    """
    Forward-pre-hook for avgpool: replace input tensor by its sign (−1,0,+1).
    """
    x, = input
    #return (2**(x+1).log2().round()-1,) #-0.1%
    return (2**((x+1).log2().mul(2).round().mul(0.5))-1,) #-0.01%

def val_top1_top5(model, val_loader, device="cpu", message="Validating model"):
    print(message)
    top1, top5 = validate(model, val_loader, device)
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")

def fuse_avgpool_linear(model):
    W = model.model.fc.weight.data  # [1000, 368]
    b = model.model.fc.bias.data if model.model.fc.bias is not None else torch.zeros(W.shape[0])
    
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        features = model.model.stem(dummy)
        features = model.model.trunk_output[:-1](features)
        features = model.model.trunk_output[-1](features)
        features = model.model.avgpool(features)
        H, W = features.shape[-2:]
    
    divisor = H * W
    W_fused = W.repeat(1, divisor) / divisor
    
    return W_fused.cpu().numpy(), b.cpu().numpy()


def pipeline(model, device, val_loader):
    # 3. validate & print 
    val_top1_top5(model, val_loader, device, "Validating model")

    # 2. binarize features
    model.avgpool.register_forward_pre_hook(binarize_hook)

    # 3. validate & print
    val_top1_top5(model, val_loader, device, "Validating binarized model")

    # 4. avgpool and linear fusing
    W_fused, b_fused = fuse_avgpool_linear(model)
    print(f"Fused weight matrix shape: {W_fused.shape}")  # [1000, 368*H*W]

    # 5. SVD
    svd = TruncatedSVD(n_components=50)
    svd.fit(W_fused)
    print(f"Explained variance ratio: {np.sum(svd.explained_variance_ratio_):.2f}")

    # 6. binary features extraction
    components = svd.components_  # [50, 368*H*W]
    binary_features = (components > 0).astype(int)

    # 7. sufficient features count
    significant_features = np.sum(np.abs(svd.components_) > 0.1)
    print(f"Number of significant binary features: {significant_features}")


def main():
    parser = argparse.ArgumentParser(description="Validate RegNetX_3.2GF on ImageNet val")
    parser.add_argument("--data_dir", default="data/imagenet")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. instantiate RegNetX_3.2GF with ImageNet1K weights
    print("Instantiating model")
    regnetx_3_2 = models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.DEFAULT).to(device)
    regnetx_16 = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.DEFAULT).to(device)

    print("Creating dataset...")
    val_loader = make_val_loader(args.data_dir, args.batch_size, args.num_workers)
    print("Dataset created")

    pipeline(regnetx_3_2, device, val_loader)
    pipeline(regnetx_16, device, val_loader)


if __name__ == "__main__":
    main()

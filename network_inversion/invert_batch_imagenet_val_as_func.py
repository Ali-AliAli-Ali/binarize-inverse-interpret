#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invert ImageNet-1K samples from regnet_x_3_2gf trunk-output features via gradient-based optimization.
Uses only conv/ReLU/skip/pool/linear layers. Saves original and reconstructed batches.
"""
import os
import sys
import time 
import random
from math import sqrt, pi, erf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, utils
from torchvision.datasets import ImageNet

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import IMAGENET_CONSTANTS   # noqa: E402
from helpful_funcs import get_model_and_features                                    # noqa: E402
from image_processing import tv_loss, normalize_contrast_saturation, gray_edge_l1   # noqa: E402


def replace_relu_with_softplus(module: nn.Module, 
                               beta: float = 1.0, 
                               threshold: float = 20.0) -> None:
    """
    Recursively replace ReLU with Softplus inplace
    
    Args:
        module:    PyTorch module whose children will be traversed
        beta:      Beta parameter for Softplus (controls smoothness)
        threshold: Threshold for Softplus (beyond this value, Softplus approximates linear)
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.Softplus(beta=beta, threshold=threshold))
        else:
            replace_relu_with_softplus(child, beta=beta, threshold=threshold)


def sym_kl_div(x, y, per_sample: bool = False):
    """
    Symmetric Kullback–Leibler divergence between two logit tensors.

    Both inputs are assumed to be logits (will be converted to log-probabilities
    using log_softmax along dimension 1). The symmetric divergence is defined as
        
    $ 0.5 * (KL(x||y) + KL(y||x))

    Args:
        x:          First logit tensor, shape (B, C, H, W).
        y:          Second logit tensor, shape (B, C, H, W).
        per_sample: If True,  returns per-sample divergences (B,);
                    otherwise returns the batch-mean divergence (scalar).

    Returns:
        Symmetric KL divergence.
    """
    # Mean along spatial dimensions (2, 3) to emulate avgpool2d, then softmax along channels (dim=1)
    #x = F.log_softmax(x.mean(dim=[2, 3], keepdim=True), dim=1)  # (B, C, 1, 1)
    #y = F.log_softmax(y.mean(dim=[2, 3], keepdim=True), dim=1)  # (B, C, 1, 1)
    x = F.log_softmax(x, dim=1)
    y = F.log_softmax(y, dim=1)
    if per_sample:
        kl1 = F.kl_div(x, y, log_target=True, reduction='none').sum(dim=[1,2,3])  # (B, C, H, W) -> (B,)
        kl2 = F.kl_div(y, x, log_target=True, reduction='none').sum(dim=[1,2,3])  # (B, C, H, W) -> (B,)
    else:
        kl1 = F.kl_div(x, y, log_target=True, reduction='batchmean')
        kl2 = F.kl_div(y, x, log_target=True, reduction='batchmean')

    return 0.5 * (kl1 + kl2)
    

def compute_all_losses(feat, target_feat, x, tv_weight, l2_weight=0.0, l1_weight=0.0, tv_loss_orig=None, per_sample=False):
    """
    Compute all loss components for feature matching and image regularization.

    The loss includes:
        - Symmetric KL divergence between feat and target_feat.
        - L2 (MSE) loss on the channel-wise mean (spatial average) of feat and target_feat.
        - Total variation loss on x, optionally with a threshold relative to tv_loss_orig.
        - L1 loss (via gray_edge_l1, assumed defined elsewhere).
        - Centering and border losses from centering_losses.

    Args:
        feat:         Predicted feature map (B, C, H, W).
        target_feat:  Target feature map (B, C, H, W).
        x:            Reconstructed image (B, 3, H, W).
        tv_weight:    Weight for the total variation term.
        l2_weight:    Weight for the MSE term.
        l1_weight:    Weight for the L1 term.
        tv_loss_orig: Reference TV loss (e.g., from original image) used to clip TV loss.
        per_sample:   If True, returns per-sample loss components; else scalar averages.

    Returns:
        Tuple(loss_kl, loss_mse, loss_tv, loss_l1, centering_loss, border_loss, total_loss).
    """
    loss_kl = sym_kl_div(feat, target_feat, per_sample=per_sample)
    loss_mse = l2_weight * F.mse_loss(
        feat.mean(dim=[1,2,3], keepdim=True), 
        target_feat.mean(dim=[1,2,3], keepdim=True), 
        reduction='none'
    ).sum(dim=[1,2,3]) \
        if per_sample else \
               l2_weight * F.mse_loss(
        feat.mean(dim=[1,2,3], keepdim=True), 
        target_feat.mean(dim=[1,2,3], keepdim=True)
    )
    
    tv_x = tv_loss(x, per_sample=per_sample) # TV loss with threshold: max(tv_loss(x) - 0.1*tv_loss_orig, 0)
    if tv_loss_orig is not None:
        tv_x = torch.maximum(tv_x - 0.4 * tv_loss_orig, torch.tensor(0.0, device=x.device, dtype=tv_x.dtype))
    loss_tv = tv_weight * tv_x
    
    loss_l1 = l1_weight * gray_edge_l1(x)

    centering_loss, border_loss = centering_losses(x)

    total_loss = loss_kl + loss_mse + loss_tv + loss_l1
    return loss_kl, loss_mse, loss_tv, loss_l1, centering_loss, border_loss, total_loss


def denormalize_and_process(x, mean, std):
    """
    Denormalize a tensor and apply contrast/saturation normalization.

    Args:
        x:    Input tensor (usually in the range of the denormalization transform).
        mean: Mean used for normalization (shape (3,) or (1,3,1,1)).
        std:  Standard deviation used for normalization.

    Returns:
        Denormalised tensor clamped to [0,1] and processed by `normalize_contrast_saturation`.
    """
    return normalize_contrast_saturation( (x * std + mean).clamp_(0., 1.) )


def forward_and_get_feat(model, x, activation):
    """
    Perform a forward pass through the model and extract a stored feature.

    Args:
        model:      PyTorch model that writes its intermediate feature into `activation['feat']`.
        x:          Input tensor to the model.
        activation: Dictionary that must contain the key 'feat' after the forward pass.

    Returns:
        The feature tensor stored in `activation['feat']`.
    """
    _ = model(x)
    return activation['feat']


def save_reconstructed_images(x, mean, std, output_path, nrow=None):
    """
    Denormalise, process and save a grid of reconstructed images.

    Args:
        x:           Reconstructed image tensor (B, C, H, W).
        mean:        Denormalization mean.
        std:         Denormalization standard deviation.
        output_path: Path where the image grid will be saved.
        nrow:        Number of images per row in the grid. If None, uses sqrt(batch_size).
    """
    x_denorm = denormalize_and_process(x, mean, std)
    if not nrow:
        nrow = int(sqrt(len(x)))
    utils.save_image(x_denorm, output_path, nrow=nrow)


def save_best_images(x_best, imgs_best, mean, std, out_dir, class_id_str, nrow=None):
    """
    Save interleaved original/reconstruction grids and separate grids for best examples.

    Creates three PNG files:
        - `best_orig_vs_recon_{class_id_str}.png`
        - `best_recon_{class_id_str}.png`
        - `best_orig_{class_id_str}.png`

    Args:
        x_best: Reconstructed images (B, C, H, W).
        imgs_best: Original images (B, C, H, W).
        mean: Denormalization mean.
        std: Denormalization standard deviation.
        out_dir: Output directory.
        class_id_str: String identifier (e.g., class name or index).
        nrow: Number of images per row (default sqrt(batch_size)).
    """
    recon_denorm = denormalize_and_process(x_best, mean, std)
    if not nrow:
        nrow = int(sqrt(len(x_best)))
    # Interleaved: orig, recon, orig, recon, ...
    comparison = torch.stack([imgs_best, recon_denorm], dim=1).reshape(-1, *imgs_best.shape[1:])
    utils.save_image(
        comparison,
        os.path.join(out_dir, f'best_orig_vs_recon_{class_id_str}.png'),
        nrow=nrow,
    )
    # Separate grids
    utils.save_image(
        recon_denorm,
        os.path.join(out_dir, f'best_recon_{class_id_str}.png'),
        nrow=nrow,
    )
    utils.save_image(
        imgs_best,
        os.path.join(out_dir, f'best_orig_{class_id_str}.png'),
        nrow=nrow,
    )


def tv_feature_loss(f):
    """
    Compute isotropic total variation loss on a whitened feature map.

    Steps:
        1. Whiten the feature map per channel (subtract mean, divide by std).
        2. Compute forward differences (horizontal and vertical) on the (H-1,W-1) grid.
        3. Sum the absolute values of all differences.

    Args:
        f: Feature tensor of shape (B, C, H, W).

    Returns:
        Scalar TV loss (sum over batch, channels, and spatial positions).
    """
    # f: (B, C, H, W)
    # whiten
    f = (f - f.mean(dim=[0,2,3], keepdim=True)) / (f.std(dim=[0,2,3], keepdim=True) + 1e-5)
    # compute spatial diffs
    dh = f[:, :, 1:, :-1] - f[:, :, :-1, :-1]  # -> (B, C, H-1, W-1)
    dw = f[:, :, :-1, 1:] - f[:, :, :-1, :-1]  # -> (B, C, H-1, W-1)
    # squared magnitude of the C-dimensional gradient vector
    grad2 = dh.abs().sum() + dw.abs().sum()  # (B, H-1, W-1)
    # isotropic vector-TV
    return grad2


def gradient_edginess(x: torch.Tensor, 
                      eps: float = 1e-8) -> torch.Tensor:
    """
    Compute edge strength (gradient magnitude) per pixel.

    Uses simple forward differences along height and width, then sums over channels
    and takes square root. The result is a 2D map per sample.

    Args:
        x:   Input tensor of shape (B, C, H, W), in float32, normalized as in inversion.
        eps: Small constant for numerical stability.

    Returns:
        Edge strength tensor of shape (B, H, W) >= 0.
    """
    # simple forward differences (cheap + stable)
    du = x[..., 1:, :] - x[..., :-1, :]
    dv = x[..., :, 1:] - x[..., :, :-1]

    # pad back to (H,W)
    du = F.pad(du, (0, 0, 0, 1))  # pad last row
    dv = F.pad(dv, (0, 1, 0, 0))  # pad last col

    s = torch.sqrt((du * du + dv * dv).sum(dim=1) + eps)  # sum over channels
    return s


def centering_losses(x: torch.Tensor, 
                     eps: float = 1e-8):
    """
    Compute centering and border losses based on the edge strength map.

    The centering loss encourages the centre of mass of edges to be close to the image centre.
    The border loss penalises edges near the image border.

    Args:
        x:   Input image tensor of shape (B, 3, H, W).
        eps: Small constant for numerical stability.

    Returns:
        A tuple (L_ctr, L_bord) where both are scalar tensors (mean over batch).
    """
    B, C, H, W = x.shape
    s = gradient_edginess(x, eps=eps)  # (B,H,W)

    # coordinates
    ii = torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1).expand(B, H, W)
    jj = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, W).expand(B, H, W)

    denom = s.sum(dim=(1,2)) + eps
    mu_i = (s * ii).sum(dim=(1,2)) / denom
    mu_j = (s * jj).sum(dim=(1,2)) / denom

    c_i = (H - 1) / 2.0
    c_j = (W - 1) / 2.0

    L_ctr = ((mu_i - c_i) ** 2 + (mu_j - c_j) ** 2).mean()

    # soft border penalty (radius^2)
    di = (ii - c_i) / float(H)
    dj = (jj - c_j) / float(W)
    d2 = di * di + dj * dj
    L_bord = (s * d2).sum(dim=(1,2)).mean() / (s.sum(dim=(1,2)).mean() + eps)

    return L_ctr, L_bord


def filter_and_sort_by_confidence(dataset, indices, model, device, batch_size, num_workers, mean, std, subset_size=5000):
    """
    Filter correctly classified images and sort by logit confidence difference.
    
    Args:
        dataset: ImageNet dataset
        indices: List of dataset indices to process
        model: Classification model (should be in eval mode)
        device: Device to run inference on
        batch_size: Batch size for processing
        mean: Mean tensor for normalization
        std: Std tensor for normalization
        subset_size: Number of samples to take from indices before filtering (default: 5000)
        
    Returns:
        List of sorted indices (descending by top-1 vs top-2 logit difference)
    """
    # Fix random seed for reproducibility
    random.seed(2147483648)
    print("Random seed:", random.getstate()[1][0])
    
    original_len = len(indices)
    # Shuffle indices and limit to subset_size if specified
    indices_list = list(indices)
    random.shuffle(indices_list)
    if subset_size > 0 and original_len > subset_size:
        indices_list = indices_list[:subset_size]
        print(f"Processing shuffled subset of {subset_size} images from {original_len} total indices")
    
    print("Filtering correctly classified images and computing logit differences...")
    
    subset_temp = Subset(dataset, indices_list)
    loader_temp = DataLoader(subset_temp, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    correct_indices = []
    logit_diffs = []
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader_temp):
            imgs = imgs.to(device)
            imgs_norm = (imgs - mean) / std
            logits = model(imgs_norm)
            
            # Get top-2 logits
            top2_logits, top2_indices = torch.topk(logits, k=2, dim=1)
            top1_logits = top2_logits[:, 0]
            top2_logits = top2_logits[:, 1]
            diff = top1_logits - top2_logits  # difference between top-1 and top-2 logits
            
            # Check which are correctly classified
            preds = top2_indices[:, 0]
            correct_mask = (preds == targets.to(device))
            
            # Store indices and differences for correctly classified images
            batch_start_idx = batch_idx * batch_size
            for i in range(len(targets)):
                if correct_mask[i]:
                    global_idx = indices_list[batch_start_idx + i]
                    correct_indices.append(global_idx)
                    logit_diffs.append(diff[i].item())
    
    # Sort by logit difference (descending)
    sorted_pairs = sorted(zip(correct_indices, logit_diffs), key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in sorted_pairs]
    
    print(f"Found {len(sorted_indices)} correctly classified images out of {len(indices_list)} total")
    
    return sorted_indices


def balance_classes(indices, 
                    targets, 
                    class_ids, 
                    max_count=None):
    """
    Balance indices across multiple classes by interleaving.
    
    Args:
        indices: List of dataset indices
        targets: List of class labels for each index (e.g., imgset.targets)
        class_ids: List of class IDs to balance across
        max_count: Maximum number of indices to return (None for all)
        
    Returns:
        Balanced list of indices interleaved across classes
    """
    if len(class_ids) <= 1 or (len(class_ids) == 1 and class_ids[0] == -1):
        # No balancing needed for single class or all classes
        result = indices[:max_count] if max_count else indices
        return result
    
    # Group indices by class
    indices_by_class = {cid: [] for cid in class_ids}
    for idx in indices:
        class_label = targets[idx]
        if class_label in indices_by_class:
            indices_by_class[class_label].append(idx)
    
    # Interleave indices to balance classes
    balanced_indices = []
    max_per_class = max(len(indices_by_class[cid]) for cid in class_ids)
    for i in range(max_per_class):
        for cid in sorted(class_ids):
            if i < len(indices_by_class[cid]):
                balanced_indices.append(indices_by_class[cid][i])
    
    # Limit to max_count if specified
    if max_count and len(balanced_indices) > max_count:
        balanced_indices = balanced_indices[:max_count]
    
    return balanced_indices


def select_best_per_class(batch_indices, loss_per_sample, targets, class_ids, select_n):
    """
    Select best samples per class, then balance across classes.
    
    Args:
        batch_indices: List of batch indices (0 to batch_size-1)
        loss_per_sample: Tensor of per-sample losses (shape: [batch_size])
        targets: List of class labels for each batch index (from dataset)
        class_ids: List of class IDs to balance across
        select_n: Total number of samples to select
        
    Returns:
        List of balanced batch indices
    """
    if len(class_ids) <= 1 or (len(class_ids) == 1 and class_ids[0] == -1):
        # No balancing needed, just select top N
        if select_n < len(batch_indices):
            _, best = torch.topk(-loss_per_sample, k=select_n)
            return best.cpu().tolist()
        return list(range(len(batch_indices)))
    
    # Group batch indices by class
    batch_indices_by_class = {cid: [] for cid in class_ids}
    for batch_idx in batch_indices:
        class_label = targets[batch_idx]
        if class_label in batch_indices_by_class:
            batch_indices_by_class[class_label].append(batch_idx)
    
    # Select best samples per class
    samples_per_class = select_n // len(class_ids)
    remainder = select_n % len(class_ids)
    
    selected_indices = []
    for i, cid in enumerate(sorted(class_ids)):
        class_indices = batch_indices_by_class[cid]
        if len(class_indices) == 0:
            continue
        
        # Get losses for this class
        class_losses = loss_per_sample[class_indices]
        # Select best samples from this class
        n_select = samples_per_class + (1 if i < remainder else 0)
        n_select = min(n_select, len(class_indices))
        
        if n_select > 0:
            _, best_class = torch.topk(-class_losses, k=n_select)
            selected_indices.extend([class_indices[idx] for idx in best_class.cpu().tolist()])
    
    # Interleave to balance (in case we have more than needed)
    if len(selected_indices) > select_n:
        # Group by class again and interleave
        selected_by_class = {cid: [] for cid in class_ids}
        for idx in selected_indices:
            class_label = targets[idx]
            if class_label in selected_by_class:
                selected_by_class[class_label].append(idx)
        
        balanced = []
        max_per_class = max(len(selected_by_class[cid]) for cid in class_ids)
        for i in range(max_per_class):
            for cid in sorted(class_ids):
                if i < len(selected_by_class[cid]):
                    balanced.append(selected_by_class[cid][i])
                    if len(balanced) >= select_n:
                        break
            if len(balanced) >= select_n:
                break
        selected_indices = balanced[:select_n]
    
    return selected_indices


def Phi(x: float) -> float:
    # Standard normal CDF
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def relu_normal_mean_std():
    # v = max(0,s), s~N(0,1)
    mu = 1.0 / sqrt(2.0 * pi)
    var = 0.5 - mu * mu
    return mu, sqrt(var)


def ab_for_threshold(t: float):
    """
    Compute parameters `a`, `b` and probability `p` for a 2-level binariser:
    
    $ v_hat = b + a * 1{v > t}
    
    where `v = max(0,s)` and `s ~ N(0,1)`.
    The parameters are chosen so that `E[v_hat] = E[v]` and `Std[v_hat] = Std[v]`.

    Args:
        t: Threshold (must be >= 0). If `t == 0`, `p = 0.5`. For `t<0`, `P(v>t)=1` degenerates (can't match nonzero std)

    Returns:
        a: Scaling factor
        b: Shift
        p: Probability P(v > t) = 1 - Phi(t) for t>0, else 0.5

    Raises:
        ValueError: If t < 0 or if p is not in (0,1) (e.g., t too large leading to p=0).
    """
    mu, sigma = relu_normal_mean_std()

    if t < 0:
        raise ValueError("t < 0 => P(v>t)=1 (degenerate). Use t >= 0.")

    p = 0.5 if not t else 1.0 - Phi(t)   # P(v>t) = P(s>t) for t>0

    if not (0.0 < p < 1.0):
        raise ValueError(f"Need 0<p<1, got p={p}. Pick t>0 (or t=0 gives p=0.5).")

    a = sigma / sqrt(p * (1.0 - p))
    b = mu - a * p
    return a, b, p


def main_pipeline(args: dict[str : int|float|str]):
    """
    Run feature inversion for ImageNet classes using specified hyperparameters.

    Args
    ----------
    args : Dictionary containing configuration parameters (see keys below).

    Keys in ``args``:
    ----------------
    data_dir : str, default './data/imagenet'
        Path to the ImageNet validation set root folder.
    class_ids : str, default '-1'
        ImageNet class index(es). Can be:
            - a single integer as string, e.g. '42'
            - comma-separated list, e.g. '0,1,2'
            - '-1' to process all classes.
    network : str, default 'regnet_x_3_2'
        Name of the neural network architecture to invert.
    batch_size : int, default 25
        Number of samples per batch during optimisation.
    steps : int, default 4000
        Number of optimisation iterations (gradient steps) per image.
    lr : float, default 0.1
        Learning rate for the optimiser.
    sigma : float, default 0.01
        Standard deviation of Gaussian noise used for initial image guess.
    beta : float, default 4.0
        Beta parameter of the Softplus activation (replaces ReLU).
    tv_weight : float, default 5e-5
        Weight of the total variation (TV) regularization term.
    l2_weight : float, default 10.0
        Weight of the L2 (MSE) loss between the target and predicted features.
    l1_weight : float, default 0.0
        Weight of the L1 (Lasso) regularization term on the reconstructed image.
    subset_size : int, default 5000
        Number of samples to load from the dataset before filtering/sorting.
    threshold : float, default None
        If not None, binarises the target feature map: values >= threshold become 1,
        others become 0. This is applied before computing losses.
    select_best_n : int, default 9
        Number of best samples (with lowest total loss) to save at the end.
        If set to 0, saves all samples.
    nrow : int, default None
        Number of images per row in the output grid. If None or 0, automatically
        determined as sqrt(batch_size).
    out : str, default 'output'
        Directory where all results (reconstructed images, logs, etc.) will be saved.
    """
    
    print(f"\nStart inversion of {args["network"]} network on ImageNet dataset...\n")

    # Parse comma-separated class IDs
    try:
        class_ids = [ int(cid.strip()) for cid in args["class_ids"].split(',') ]
    except ValueError:
        raise ValueError(f"Invalid class id format: '{args["class_ids"]}'. Expected comma-separated integers or '-1' for all classes.")


    os.makedirs(args["out_dir"], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    IMAGENET_CONSTANTS["mean"] = IMAGENET_CONSTANTS["mean"].to(device)
    IMAGENET_CONSTANTS["std"] = IMAGENET_CONSTANTS["std"].to(device)
    IMAGENET_CONSTANTS["mean_awb"] = IMAGENET_CONSTANTS["mean_awb"].to(device)


    # 1. Load network 
    
    model, features = get_model_and_features(args["network"])

    # replacing ReLU with SoftPlus  
    print("ReLUs:   ", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    replace_relu_with_softplus(model, beta=args["beta"])

    print("ReLUs:   ", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    model = model.to(device).eval()

    # 1b. Register hooks: target, features, and all BatchNorm outputs
    
    activation = {}
    hooks = {}

    # Hook trunk_output (pre-pool) features
    def hook_fn(module, input, output):
        activation['feat'] = output
    hooks['feat'] = features.register_forward_hook(hook_fn)


    # 2. Prepare ImageNet validation subset for given class

    # standard ImageNet val preprocessing
    transform_raw = transforms.Compose([
        transforms.Resize(IMAGENET_CONSTANTS["size_resize"]),
        transforms.CenterCrop(IMAGENET_CONSTANTS["size_center_crop"]), 
        transforms.ToTensor(),    # [0..1] float tensor
    ])
    
    imgset = ImageNet(root=args["data_dir"], split='val', transform=transform_raw)
    # Create string representation for file naming
    # imgset.targets is a plain Python list of length N with the class idx for each sample
    if len(class_ids) == 1 and class_ids[0] == -1:
        class_id_str = 'all'
        indices = list(range(len(imgset.targets)))
    else:
        class_id_str = '_'.join(f'{cid:04d}' for cid in sorted(class_ids))
        indices = [i for i, t in enumerate(imgset.targets) if t in class_ids]
    
    # DEBUG: print class distribution
    class_counts = {}
    for idx in indices:
        class_label = imgset.targets[idx]
        class_counts[class_label] = class_counts.get(class_label, 0) + 1
    print(f"Found {len(indices)} images from classes {sorted(class_ids)}")
    for cid in sorted(class_ids):
        print(f"  Class {cid}: {class_counts.get(cid, 0)} images")

    
    # 2a. Filter correctly classified images and sort by softmax confidence
    # Load a clean model for classification (before Softplus replacement)
    model_cls, _ = get_model_and_features(args["network"])
    model_cls.to(device).eval()
    sorted_indices = filter_and_sort_by_confidence(
        imgset, 
        indices, 
        model_cls, 
        device, 
        args["batch_size"], 
        args["num_workers"],
        IMAGENET_CONSTANTS["mean"], 
        IMAGENET_CONSTANTS["std"], 
        args["subset_size"]
    )
    
    # DEBUG: print class distribution after filtering
    if len(class_ids) > 1 or (len(class_ids) == 1 and class_ids[0] != -1):
        class_counts_after = {}
        for idx in sorted_indices:
            class_label = imgset.targets[idx]
            class_counts_after[class_label] = class_counts_after.get(class_label, 0) + 1
        print(f"After filtering: {len(sorted_indices)} correctly classified images")
        for cid in sorted(class_ids):
            print(f"  Class {cid}: {class_counts_after.get(cid, 0)} correctly classified images")

    # Balance sampling across classes if multiple classes are requested
    if len(class_ids) > 1 and class_ids[0] != -1:
        sorted_indices = balance_classes(sorted_indices, imgset.targets, class_ids, max_count=args["batch_size"])
        print(f"Balanced sampling: selected {len(sorted_indices)} images across classes {sorted(class_ids)}")
    
    # Create new loader from sorted images without shuffle
    subset = Subset(imgset, sorted_indices)
    loader = DataLoader(subset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])

    imgs, _ = next(iter(loader))  # (B,3,224,224)
    imgs = imgs.to(device)
    utils.save_image(imgs, os.path.join(args["out_dir"], f'orig_{class_id_str}.png'), nrow=int(sqrt(len(imgs))))

    
    # 3. Normalize for model input
    
    imgs_norm = (imgs - IMAGENET_CONSTANTS["mean"]) / IMAGENET_CONSTANTS["std"]  # batch normalization with global constants
    
    
    # 4. Extract and fix target features
    
    tv_loss_orig = tv_loss(imgs_norm)  # TV loss of original images for threshold
    with torch.no_grad():
        _ = model(imgs_norm)
        target_feat = activation['feat'].detach()
        if args["threshold"] is not None:
            a, b, _ = ab_for_threshold(args["threshold"])
            target_feat = b + a * (target_feat >= args["threshold"]).float()

    
    # 5. Initialize reconstruction
    
    noise = torch.randn_like(imgs_norm) * args["sigma"]
    x = noise.requires_grad_(True)

    opt = optim.Adam([x], lr=args["lr"], betas=(0.9, 0.999))
    #opt = optim.SGD([x], lr=args["lr"], momentum=0.9)
    #scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1 - step / float(args["steps"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args["steps"])

    
    # 6. Reconstruction loop
    
    recon_start = time.perf_counter()
    n_steps_log = 100
    
    for step in range(args["steps"]+1):
        recon_step_start = time.perf_counter()

        l2_weight = args["l2_weight"]# * scheduler.get_last_lr()[0] / args["lr"]
        opt.zero_grad()
        
        feat = forward_and_get_feat(model, x, activation)
        loss_kl, loss_mse, loss_tv, loss_l1, centering_loss, border_loss, total_loss = compute_all_losses(
            feat, 
            target_feat, 
            x, 
            args["tv_weight"], 
            l2_weight=l2_weight, 
            l1_weight=args["l1_weight"], 
            tv_loss_orig=tv_loss_orig
        )
        
        recon_step_time = time.perf_counter() - recon_step_start
        
        if not step % n_steps_log:
            current_lr = opt.param_groups[0]['lr']
            print(
                f"Step {step:<5}/{args["steps"]} | "
                f"MSE: {loss_mse.item():<12.6f} "
                f"KL: {loss_kl.item():<12.6f} "
                f"TV: {loss_tv.item():<12.6f} "
                f"L1: {loss_l1.item():<12.6f} "
                f"Centering: {centering_loss.item():<12.6f} "
                f"Border: {border_loss.item():<12.6f} "
                f"Total: {total_loss.item():<12.6f} "
                f"LR: {current_lr:<10.6f} "
                f"Step time: {recon_step_time:<10.3f} s"
            )
            save_reconstructed_images(x, IMAGENET_CONSTANTS["mean_awb"], IMAGENET_CONSTANTS["std"], os.path.join(args["out_dir"], f'recon_{step:04d}.png'), nrow=args["nrow"])

        total_loss.backward()
        opt.step()
        scheduler.step()
        
    recon_time = time.perf_counter() - recon_start
    print(f"Reconstruction done in {recon_time:.3f} s. Average time for step: {recon_time / args["steps"]:.3f} s")

    # Check classification of all reconstructed images with original model
    with torch.no_grad():
        feat = forward_and_get_feat(model, x, activation)
        
        # Get true class labels for all samples
        all_true_labels = torch.tensor([imgset.targets[sorted_indices[i]] for i in range(len(x))], device=device)
        
        # x is already in normalized space, classify directly
        logits_recon = model_cls(x)
        preds_recon = logits_recon.argmax(dim=1)
        correct_mask = (preds_recon == all_true_labels)
        correct_recon = correct_mask.sum().item()
        # correct_indices = torch.where(correct_mask)[0].cpu().tolist()
        
    
    # Print classification statistics
    print("\nClassification accuracy (all samples):")
    print(f"  Reconstructed images: {correct_recon}/{len(all_true_labels)} ({100*correct_recon/len(all_true_labels):.1f}%)")
    
    # Print per-class accuracy if multiple classes
    if len(class_ids) > 1 and class_ids[0] != -1:
        print("\nPer-class accuracy (reconstructed, all samples):")
        for cid in sorted(class_ids):
            class_mask = (all_true_labels == cid)
            if class_mask.sum() > 0:
                class_correct = (preds_recon[class_mask] == all_true_labels[class_mask]).sum().item()
                class_total = class_mask.sum().item()
                print(f"  Class {cid}: {class_correct}/{class_total} ({100*class_correct/class_total:.1f}%)")

    
    # 7. Select best samples and save (only from correctly classified)
    
    with torch.no_grad():
        feat = forward_and_get_feat(model, x, activation)
        
        # Compute per-sample losses
        # Calculate per-sample TV loss for originals
        tv_loss_orig_per_sample = tv_loss(imgs_norm, per_sample=True)
        loss_kl_per_sample, _, _, _, _, _, _ = compute_all_losses(
            feat, target_feat, x, args["tv_weight"], l2_weight=args["l2_weight"], l1_weight=args["l1_weight"], tv_loss_orig=tv_loss_orig_per_sample, per_sample=True
        )
        
        loss_per_sample = loss_kl_per_sample
        
        # Use all samples (do not drop misclassified)
        loss_per_sample_correct = loss_per_sample
        num_correct = loss_per_sample_correct.size(0)
        select_n = min(args["select_best_n"], num_correct) if args["select_best_n"] is not None else num_correct
        
        # Get class labels for all batch indices
        batch_targets_correct = [imgset.targets[sorted_indices[i]] for i in range(num_correct)]
        
        # Select best samples per class if multiple classes requested
        # Use position indices (0 to num_correct-1) for select_best_per_class, then map back to batch indices
        if len(class_ids) > 1 and class_ids[0] != -1:
            position_indices = list(range(num_correct))
            best_position_indices = select_best_per_class(
                position_indices, loss_per_sample_correct, batch_targets_correct, class_ids, select_n
            )
            # Map back to actual batch indices
            best_indices = best_position_indices
            print(f"Selected {len(best_indices)} balanced samples across classes {sorted(class_ids)}")
        else:
            if select_n < num_correct:
                # Use full sort (ascending) for deterministic top-k selection
                _, sorted_idx = torch.sort(loss_per_sample_correct)
                best_indices = sorted_idx[:select_n].cpu().tolist()
                print(f"Selected {select_n} samples with minimal loss (out of {num_correct})")
            else:
                _, sorted_idx = torch.sort(loss_kl_per_sample)
                best_indices = sorted_idx.cpu().tolist()
                print(f"Saving all {num_correct} samples (sorted by KL loss)")
        
        # Select best samples
        x_best = x[best_indices]
        imgs_best = imgs[best_indices]
        #imgs_mean_best = imgs_mean[best_indices]
        
        # Save reconstructed and original images for the best subset
        save_best_images(x_best, imgs_best, IMAGENET_CONSTANTS["mean_awb"], IMAGENET_CONSTANTS["std"], args["out_dir"], class_id_str, nrow=args["nrow"])
        
        # Print loss statistics
        print(f"\nBest samples total_loss range: [{loss_per_sample[best_indices].min().item():<12.6f}, {loss_per_sample[best_indices].max().item():<12.6f}]")
        print(f"Mean total_loss for best samples: {loss_per_sample[best_indices].mean().item():<12.6f}")

    for h in hooks.values():
        h.remove()

    print("\nDone!\n")

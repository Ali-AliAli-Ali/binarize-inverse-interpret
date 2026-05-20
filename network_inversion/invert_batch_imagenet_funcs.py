#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invert ImageNet-1K samples from chosen network trunk-output features via gradient-based optimization.
Uses only conv/ReLU/skip/pool/linear layers. Saves original and reconstructed batches.
"""
import os
import sys
import time 
import random
from math import sqrt, pi, erf
from PIL import Image
from typing import Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, utils
from torchvision.datasets import ImageNet

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import IMAGENET_CONSTANTS, MODELS_CONFIGS   # noqa: E402
from helpful_funcs import get_model_and_features, format_classes_ids_str            # noqa: E402
from image_processing import normalize_contrast_saturation, gray_edge_l1   # noqa: E402
from training_logger import TrainingLogger    # noqa: E402


# INVERSION TRAINING


# Operations with network & features


def replace_relu_with_softplus(module: nn.Module, 
                               beta: float = 1.0, 
                               threshold: float = 20.0):
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


def forward_and_get_feat(model, 
                         x: torch.Tensor, 
                         activation: dict) -> torch.Tensor:
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


def filter_and_sort_by_confidence(dataset: ImageNet, 
                                  indices: list, 
                                  model, 
                                  device: str, 
                                  batch_size: int, 
                                  mean: torch.Tensor,
                                  std: torch.Tensor, 
                                  subset_size: int | None = 5000,
                                  num_workers: int | None = 8, 
                                  random_seed: int = 42) -> list:
    """
    Filter correctly classified images and sort by logit confidence difference.
    
    Args:
        dataset:     ImageNet dataset.
        indices:     List of dataset indices to process.
        model:       Classification model.
        device:      Device to run model inference on.
        batch_size:  Batch size for processing.
        mean:        Mean tensor for normalization.
        std:         Standart deviation tensor for normalization.
        subset_size: Number of samples to take from indices before filtering (default: 5000).
        num_workers: Number of subprocesses for data loading.
        
    Returns:
        List of sorted indices (descending by top-1 vs top-2 logit difference)
    """
    # Fix random seed for reproducibility
    random.seed(random_seed)
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
    
    model = model.to(device)
    model.eval()
    
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
            batch_start_i = batch_idx * batch_size
            for i in range(len(targets)):
                if correct_mask[i]:
                    global_idx = indices_list[batch_start_i + i]
                    correct_indices.append(global_idx)
                    logit_diffs.append(diff[i].item())
    
    # Sort by logit difference (descending)
    sorted_pairs = sorted(zip(correct_indices, logit_diffs), key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in sorted_pairs]
    
    print(f"Found {len(sorted_indices)} correctly classified images out of {len(indices_list)} total")
    
    return sorted_indices


def balance_classes(indices: list, 
                    targets: list, 
                    class_ids: list, 
                    max_count: int | None = None):
    """
    Balance indices across multiple classes by interleaving.
    
    Args:
        indices:   List of dataset indices
        targets:   List of class labels for each index (e.g., imgset.targets)
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
    indices_by_class = {class_id: [] for class_id in class_ids}
    for idx in indices:
        class_label = targets[idx]
        if class_label in indices_by_class:
            indices_by_class[class_label].append(idx)
    
    # Interleave indices to balance classes
    balanced_indices = []
    max_per_class = max(len(indices_by_class[class_id]) for class_id in class_ids)
    for i in range(max_per_class):
        for class_id in sorted(class_ids):
            if i < len(indices_by_class[class_id]):
                balanced_indices.append(indices_by_class[class_id][i])
    
    # Limit to max_count if specified
    if max_count and len(balanced_indices) > max_count:
        balanced_indices = balanced_indices[:max_count]
    
    return balanced_indices


def select_best_per_class(batch_indices: list[int], 
                          loss_per_sample: torch.Tensor, 
                          targets: list, 
                          class_ids: list, 
                          select_n: int):
    """
    Select best samples per class, then balance across classes.
    
    Args:
        batch_indices:   List of batch indices (0 to batch_size-1)
        loss_per_sample: Tensor of per-sample losses (shape: [batch_size])
        targets:         List of class labels for each batch index (from dataset)
        class_ids:       List of class IDs to balance across
        select_n:        Total number of samples to select
        
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
    batch_indices_by_class = {class_id: [] for class_id in class_ids}
    for batch_idx in batch_indices:
        class_label = targets[batch_idx]
        if class_label in batch_indices_by_class:
            batch_indices_by_class[class_label].append(batch_idx)
    
    # Select best samples per class
    samples_per_class = select_n // len(class_ids)
    remainder = select_n % len(class_ids)
    
    selected_indices = []
    for i, class_id in enumerate(sorted(class_ids)):
        class_indices = batch_indices_by_class[class_id]
        if not len(class_indices):
            continue
        
        # Get losses for this class
        class_losses = loss_per_sample[class_indices]
        # Select best samples from this class
        n_select = min(
            samples_per_class + int(i < remainder), 
            len(class_indices)
        )
        
        if n_select:
            _, best_class = torch.topk(-class_losses, k=n_select)
            selected_indices.extend([class_indices[idx] for idx in best_class.cpu().tolist()])
    
    # Interleave to balance (in case we have more than needed)
    if len(selected_indices) > select_n:
        # Group by class again and interleave
        selected_by_class = {class_id: [] for class_id in class_ids}
        for idx in selected_indices:
            class_label = targets[idx]
            if class_label in selected_by_class:
                selected_by_class[class_label].append(idx)
        
        balanced = []
        max_per_class = max(len(selected_by_class[class_id]) for class_id in class_ids)
        for i in range(max_per_class):
            for class_id in sorted(class_ids):
                if i < len(selected_by_class[class_id]):
                    balanced.append(selected_by_class[class_id][i])
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
        t: Threshold (must be >= 0). If `t == 0`, `p = 0.5`. 
           For `t<0`, `P(v>t)=1` degenerates (can't match nonzero std)

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


# Losses calculations


def sym_kl_div_feature_loss(x: torch.Tensor, 
                            y: torch.Tensor, 
                            per_sample: bool = False) -> float:
    """
    Symmetric Kullback-Leibler divergence between two logit tensors.

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


def mse_feature_loss(x: torch.Tensor,
                     y: torch.Tensor,
                     per_sample: bool | None = False):
    """
    Mean Squared Error between the channel-wise spatial average of two feature maps.

    When `per_sample=False`, the result is averaged over the batch for batch size
    independency. When `per_sample=True`, per-sample MSE are returned with shape (B,).

    Args:
        x:          First feature tensor, shape (B, C, H, W).
        y:          Second feature tensor, shape (B, C, H, W).
        per_sample: If `True`, return per-sample losses (B,);
                    otherwise return the batch-mean loss (scalar).

    Returns:
        MSE loss.
    """
    x_mean = x.mean(dim=[1,2,3], keepdim=True)
    y_mean = y.mean(dim=[1,2,3], keepdim=True)

    return F.mse_loss(x_mean, y_mean, reduction="none").sum(dim=[1,2,3]) \
        if per_sample else \
            F.mse_loss(x_mean, y_mean, reduction="mean")


def tv_feature_loss(x: torch.Tensor, 
                    per_sample: bool | None = False) -> torch.Tensor:
    """
    Total variation (TV) loss with separate channel differences for RGB images.
    Computes horizontal and vertical differences, then adds cross-channel terms
    (R-G, B-G) for both directions.

    Args:
        x: Input tensor of shape (B, 3, H, W) in arbitrary range.
        per_sample: If True, returns a 1D tensor of shape (B,) with per-sample losses;
                    otherwise returns a scalar (sum over batch).

    Returns:
        TV loss value (scalar or per-sample vector).
    """
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    dhr = dh[:, 0, :, :] - dh[:, 1, :, :]
    dhb = dh[:, 2, :, :] - dh[:, 1, :, :]
    dwr = dw[:, 0, :, :] - dw[:, 1, :, :]
    dwb = dw[:, 2, :, :] - dw[:, 1, :, :]
    
    return dh.abs().sum(dim=[1,2,3]) + dw.abs().sum(dim=[1,2,3]) + dhr.abs().sum(dim=[1,2]) + dhb.abs().sum(dim=[1,2]) + dwr.abs().sum(dim=[1,2]) + dwb.abs().sum(dim=[1,2]) \
           if per_sample else \
           dh.abs().sum() + dw.abs().sum() + dhr.abs().sum() + dhb.abs().sum() + dwr.abs().sum() + dwb.abs().sum()


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

    return torch.sqrt((du * du + dv * dv).sum(dim=1) + eps)  # sum over channels


def centering_feature_losses(x: torch.Tensor, 
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
  

AllLosses = tuple[float, float, float, float, float, float, float]
def compute_all_losses(feat: torch.Tensor, 
                       target_feat: torch.Tensor, 
                       x: torch.Tensor, 
                       tv_weight: float | None = 0.0, 
                       l2_weight: float | None = 0.0, 
                       l1_weight: float | None = 0.0, 
                       tv_loss_orig: float | None = None, 
                       per_sample: bool | None = False) -> AllLosses:
    """
    Compute all loss components for feature matching and image regularization:
        - Symmetric KL divergence between `feat` and `target_feat`.
        - Total variation loss on `x`, optionally with a threshold relative to `tv_loss_orig`.
        - L2 (MSE) loss on the channel-wise mean of `feat` and `target_feat`.
        - L1 loss on the channel-wise mean of `feat` and `target_feat`.
        - Centering and border losses.

    Args:
        feat:         Predicted feature map.
        target_feat:  Target feature map.
        x:            Reconstructed image.
        tv_weight:    Weight for the total variation term.
        l2_weight:    Weight for the MSE term.
        l1_weight:    Weight for the L1 term.
        tv_loss_orig: Reference TV loss used to clip TV loss.
        per_sample:   If `True`, returns per-sample loss components; else scalar averages.

    Returns:
        tuple(loss_kl, loss_mse, loss_tv, loss_l1, centering_loss, border_loss, total_loss), where
        losses are not weighted and `total_loss` is a weighted sum.
    """
    loss_kl = sym_kl_div_feature_loss(feat, target_feat, per_sample=per_sample)
    loss_mse =  mse_feature_loss(feat, target_feat, per_sample=per_sample)
    loss_l1 = gray_edge_l1(x)
    loss_centering, loss_border = centering_feature_losses(x)
    
    loss_tv = tv_feature_loss(x, per_sample=per_sample) # TV loss with threshold: max(tv_feature_loss(x) - 0.1*tv_feature_loss_orig, 0)
    if tv_loss_orig is not None:
        loss_tv = torch.maximum(
            loss_tv - 0.4 * tv_loss_orig, 
            torch.tensor(0.0, device=x.device, dtype=loss_tv.dtype)
        )
    
    return loss_kl, loss_mse, loss_tv, loss_l1, loss_centering, loss_border, \
           loss_kl + l2_weight * loss_mse + tv_weight * loss_tv + l1_weight * loss_l1


# Images processing


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
    


def main_pipeline(data_dir: str | None = './data/imagenet',
                  class_ids_str: str | None = "-1",
                  network_name: str | None = "regnet_x_3_2",
                  batch_size: int | None = 25,
                  num_workers: int | None = 8,
                  steps: int | None = 4000,
                  lr: float | None = 0.1,
                  sigma: float | None = 0.01,
                  beta: float | None = 4.0,
                  tv_weight: float | None = 5e-5,
                  l2_weight: float | None = 10.0,
                  l1_weight: float | None = 0.0,
                  subset_size: int | None = 5000,
                  threshold: float | None = None,
                  select_best_n: int | None = None,
                  nrow: int | None = None,
                  out_dir: str | None = "./output",
                  run_mode: Literal["debug", "run"] | None = "run"):
    """
    Run feature inversion for ImageNet classes using specified hyperparameters.

    Args:
        data_dir:      Path to the ImageNet validation set root folder.
        class_ids:     ImageNet class index(es). Can be:
                           - a single integer as string, e.g. '42'
                           - comma-separated list, e.g. '0,1,2'
                           - '-1' to process all classes.
        network_name:  Name of the neural network architecture to invert.
        batch_size:    Number of samples per batch during optimization.
        num_workers:   Number of subprocesses for data loading.
        steps:         Number of optimization iterations (gradient steps) per image.
        lr:            Learning rate for the optimiser.
        sigma:         Standard deviation of Gaussian noise used for initial image guess.
        beta:          Beta parameter of the Softplus activation (replaces ReLU).
        tv_weight:     Weight of the total variation (TV) regularization term.
        l2_weight:     Weight of the L2 (MSE) loss between the target and predicted features.
        l1_weight:     Weight of the L1 (Lasso) regularization term on the reconstructed image.
        subset_size:   Number of samples to load from the dataset before filtering/sorting.
        threshold:     If not `None`, binarises the target feature map: values >= threshold become 1,
                       others become 0. This is applied before computing losses.
        select_best_n: Number of best samples (with lowest total loss) to save at the end.
                       If None or 0, saves all samples.
        nrow:          Number of images per row in the output grid. If `None` or `0`, automatically
                       determined as `sqrt(batch_size)`.
        out_dir:       Directory to save all results (reconstructed images, logs, etc.).
        run_mode:      Mode to run inversion training. Mode `"debug"` adds debugging print statements and
                       saves intermediate inversion images
    """
    
    print(f"\nStart inversion of {network_name} network on ImageNet dataset...\n")

    # Parse comma-separated class IDs
    try:
        class_ids = [ int(class_id.strip()) for class_id in class_ids_str.split(',') ]
    except ValueError:
        raise ValueError(f"Invalid class id format: '{class_ids_str}'. Expected comma-separated integers or '-1' for all classes.")


    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    IMAGENET_CONSTANTS["mean"] = IMAGENET_CONSTANTS["mean"].to(device)
    IMAGENET_CONSTANTS["std"] = IMAGENET_CONSTANTS["std"].to(device)
    IMAGENET_CONSTANTS["mean_awb"] = IMAGENET_CONSTANTS["mean_awb"].to(device)


    # 1. Load network 
    
    model, features = get_model_and_features(network_name)

    print("ReLUs:   ", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    replace_relu_with_softplus(model, beta=beta)

    print("ReLUs:   ", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    model = model.to(device).eval()

    # 1b. Register hooks: target, features, and all BatchNorm outputs
    
    activation = {}
    hooks = {}


    # hook getting trunk_output (pre-pool) features
    def hook_fn(module, input, output):
        activation['feat'] = output
        
    # hook reshaping ViT encoder output to 4D feature map
    def vit_encoder_hook(module, input, output):
        # Remove CLS token (assumed to be the first token)
        patch_tokens = output[:, 1:, :]                     # (B, num_patches, D)
        B, N, D = patch_tokens.shape
        
        # Calculate grid size (square root of num_patches)
        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(
                f"Number of ViT patches {N} is not a perfect square, " \
                f"check input image size and patch size."
            )
        
        # Reshape: (B, grid, grid, D) -> (B, D, grid, grid)
        activation['feat'] = patch_tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)    
        
    if "vit_" in network_name:
        hooks['feat'] = model.encoder.register_forward_hook(vit_encoder_hook)
    else:
        hooks['feat'] = features.register_forward_hook(hook_fn)    


    # 2. Prepare ImageNet validation subset for given class

    # standard ImageNet val preprocessing
    transform_raw = transforms.Compose([
        transforms.Resize(IMAGENET_CONSTANTS["size_resize"]),
        transforms.CenterCrop(IMAGENET_CONSTANTS["size_center_crop"]), 
        transforms.ToTensor(),    # [0..1] float tensor
    ])
    
    imgset = ImageNet(root=data_dir, split='val', transform=transform_raw)
    # Create string representation for file naming
    # imgset.targets is a plain Python list of length N with the class idx for each sample
    if len(class_ids) == 1 and class_ids[0] == -1:
        class_id_str = 'all'
        indices = list(range(len(imgset.targets)))
    else:
        class_id_str = format_classes_ids_str(class_ids)
        indices = [i for i, t in enumerate(imgset.targets) if t in class_ids]
    
    # DEBUG: print class distribution
    if run_mode == "debug":
        print("    [debug] Class distribution before filtering: ")
        class_counts = {}
        for idx in indices:
            class_label = imgset.targets[idx]
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        print(f"    Found {len(indices)} images from classes {sorted(class_ids)}")
        for class_id in sorted(class_ids):
            print(f"        Class {class_id}: {class_counts.get(class_id, 0)} images")

    # 2a. Filter correctly classified images and sort by softmax confidence
    # Load a clean model for classification (before Softplus replacement)
    model_cls, _ = get_model_and_features(network_name)
    model_cls.to(device).eval()
    sorted_indices = filter_and_sort_by_confidence(
        imgset, 
        indices, 
        model_cls, 
        device, 
        batch_size, 
        IMAGENET_CONSTANTS["mean"], 
        IMAGENET_CONSTANTS["std"], 
        subset_size,
        num_workers
    )
    
    # DEBUG: print class distribution after filtering
    if run_mode == "debug":
        print("    [debug] Class distribution before filtering: ")
        if len(class_ids) > 1 or (len(class_ids) == 1 and class_ids[0] != -1):
            class_counts_after = {}
            for idx in sorted_indices:
                class_label = imgset.targets[idx]
                class_counts_after[class_label] = class_counts_after.get(class_label, 0) + 1
            print(f"    After filtering: {len(sorted_indices)} correctly classified images")
            for class_id in sorted(class_ids):
                print(f"        Class {class_id}: {class_counts_after.get(class_id, 0)} correctly classified images")

    # Balance sampling across classes if multiple classes are requested
    if len(class_ids) > 1 and class_ids[0] != -1:
        sorted_indices = balance_classes(sorted_indices, imgset.targets, class_ids, max_count=batch_size)
        print(f"Balanced sampling: selected {len(sorted_indices)} images across classes {sorted(class_ids)}")
    
    # Create new loader from sorted images without shuffle
    subset = Subset(imgset, sorted_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    imgs, _ = next(iter(loader))  # (B,3,224,224)
    imgs = imgs.to(device)
    utils.save_image(imgs, os.path.join(out_dir, f'orig_{class_id_str}.png'), nrow=int(sqrt(len(imgs))))

    
    # 3. Normalize images for model input
    
    imgs_norm = (imgs - IMAGENET_CONSTANTS["mean"]) / IMAGENET_CONSTANTS["std"]  # batch normalization with global constants
    
    
    # 4. Extract and fix target features
    
    tv_loss_orig = tv_feature_loss(imgs_norm)  # TV loss of original images for threshold
    with torch.no_grad():
        _ = model(imgs_norm)
        target_feat = activation['feat'].detach()
        if threshold is not None:
            a, b, _ = ab_for_threshold(threshold)
            target_feat = b + a * (target_feat >= threshold).float()

    
    # 5. Initialize reconstruction
    
    noise = torch.randn_like(imgs_norm) * sigma
    x = noise.requires_grad_(True)

    #optimizer = optim.SGD([x], lr=lr, momentum=0.9)
    optimizer = optim.Adam([x], lr=lr, betas=(0.9, 0.999))
    #scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1 - step / float(steps))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    
    # 6. Reconstruction loop
    
    logger = TrainingLogger(
        metric_names=["kl_loss", "mse_loss", "tv_loss", "l1_loss", "centering_loss", "border_loss", "total_loss"],
        log_file=os.path.join(out_dir, "training_log.npz"),
        max_steps=steps
    )
    
    recon_start = time.perf_counter()
    n_steps_log = 100
    
    for step in range(steps+1):
        recon_step_start = time.perf_counter()

        optimizer.zero_grad()
        
        feat = forward_and_get_feat(model, x, activation)
        loss_kl, loss_mse, loss_tv, loss_l1, centering_loss, border_loss, total_loss = compute_all_losses(
            feat, 
            target_feat, 
            x, 
            tv_weight, 
            l2_weight=l2_weight, 
            l1_weight=l1_weight, 
            tv_loss_orig=tv_loss_orig
        )
        
        recon_step_time = time.perf_counter() - recon_step_start
        
        logger.log(step, loss_kl, loss_mse, loss_tv, loss_l1, centering_loss, border_loss, total_loss)
        
        if not step % n_steps_log:
            print(
                f"Step {step:<5}/{steps} | "
                f"KL: {loss_kl.item():<12.6f} "
                f"MSE: {loss_mse.item():<12.6f} "
                f"TV: {loss_tv.item():<12.6f} "
                f"L1: {loss_l1.item():<12.6f} "
                f"Centering: {centering_loss.item():<12.6f} "
                f"Border: {border_loss.item():<12.6f} "
                f"Total: {total_loss.item():<12.6f} "
                f"LR: {optimizer.param_groups[0]['lr']:<10.6f} "
                f"Step time: {recon_step_time:<6.3f} s"
            )
            if run_mode == "debug":
                save_reconstructed_images(
                    x, 
                    IMAGENET_CONSTANTS["mean_awb"], 
                    IMAGENET_CONSTANTS["std"], 
                    os.path.join(out_dir, f"recon_{step:04d}.png"), 
                    nrow=nrow
                )

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
    recon_time = time.perf_counter() - recon_start
    print(f"Reconstruction done in {recon_time:.3f} s. Average time for step: {recon_time / steps:.3f} s")

    logger.close()

    # Check classification of all reconstructed images with original model
    with torch.no_grad():
        feat = forward_and_get_feat(model, x, activation)
        
        # Get true class labels for all samples
        all_labels_true = torch.tensor([imgset.targets[sorted_indices[i]] for i in range(len(x))], device=device)
        
        # x is already in normalized space, classify directly
        logits_recon = model_cls(x)
        preds_recon = logits_recon.argmax(dim=1)
        correct_mask = (preds_recon == all_labels_true)
        correct_recon = correct_mask.sum().item()
        # correct_indices = torch.where(correct_mask)[0].cpu().tolist()
    
    # Print classification statistics
    print("\nClassification accuracy (all samples):")
    print(f"    Reconstructed images: {correct_recon}/{len(all_labels_true)} ({100*correct_recon/len(all_labels_true):.1f}%)")
    
    # Print per-class accuracy if multiple classes
    if len(class_ids) > 1 and class_ids[0] != -1:
        print("\nPer-class accuracy (reconstructed, all samples):")
        for class_id in sorted(class_ids):
            class_mask = (all_labels_true == class_id)
            if class_mask.sum() > 0:
                class_correct = (preds_recon[class_mask] == all_labels_true[class_mask]).sum().item()
                class_total = class_mask.sum().item()
                print(f"  Class {class_id}: {class_correct}/{class_total} ({100*class_correct/class_total:.1f}%)")

    
    # 7. Select best samples and save (only from correctly classified)
    
    with torch.no_grad():
        feat = forward_and_get_feat(model, x, activation)
        
        # Compute per-sample losses
        loss_kl_per_sample, _, _, _, _, _, _ = compute_all_losses(
            feat, 
            target_feat, 
            x,
            tv_weight, 
            l2_weight=l2_weight, 
            l1_weight=l1_weight, 
            tv_loss_orig=tv_feature_loss(imgs_norm, per_sample=True), 
            per_sample=True
        )
        
        loss_per_sample = loss_kl_per_sample
        
        # Use all samples (do not drop misclassified)
        loss_per_sample_correct = loss_per_sample
        num_correct = loss_per_sample_correct.size(0)
        select_n = min(select_best_n, num_correct) if select_best_n else num_correct
        
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
        save_best_images(x_best, imgs_best, IMAGENET_CONSTANTS["mean_awb"], IMAGENET_CONSTANTS["std"], out_dir, class_id_str, nrow=nrow)
        
        # Print loss statistics
        print(f"\nBest samples total_loss range: [{loss_per_sample[best_indices].min().item():<12.6f}, {loss_per_sample[best_indices].max().item():<12.6f}]")
        print(f"Mean total_loss for best samples: {loss_per_sample[best_indices].mean().item():<12.6f}")

    
    # 8. Clean model from hooks
    
    for h in hooks.values():
        h.remove()

    print("\nDone!\n")


# INVERSION VALIDATION


class ImageFolderDataset(Dataset):
    """
    Dataset for loading images from a folder, extracting labels from file names or provided list.
    """
    def __init__(self, 
                 images_paths: list[str], 
                 labels: list[int], 
                 transform: None):
        """
        Args:
            images_paths: List of paths (absolute or relative) to images.
            labels:      Ground truth class indices corresponding to each image.
            transform:   `torchvision transform` to apply to each image.
        """
        self.images_paths = images_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img = Image.open(self.images_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label



def get_grid_images_paths(networks_names: list[str],
                          val_images_dir: str,
                          classes_ids_list: list[int],
                          batch_sizes: dict[int] | None = None,
                          grid_images_prefix: str | None = "best_orig_vs_recon_",
                          grid_images_ext: str | None = "png") -> dict:
    """
    Return a dictionary mapping each network name to a list of grid image paths.

    If the network's batch size is at least the number of classes, a single path
    constructed with the formatted class IDs is returned. Otherwise, all files
    with the given extension in the network's subdirectory are returned.
    
    Args:
        networks_names:     List of network names.
        val_images_dir:     Path to directory with grid image(s).
        classes_ids_list:   List of integer class IDs.
        batch_sizes:        Dict `{ network_name : batch_size }` for every network name in `networks_names`.
                            If `None`, default batch sizes from config are taken.
        grid_images_prefix: Prefix for the grid image filename when a single path is generated.
        grid_images_ext:    File extension for the grid images (without dot).

    Returns:
        Dictionary mapping network name to a list of absolute file paths to grid images.
    """
    batch_sizes = batch_sizes or { 
        network_name : MODELS_CONFIGS[network_name]["batch_size"]
        for network_name in networks_names
    }
    
    return {
        network_name :  [ 
                            os.path.join(
                                val_images_dir, 
                                network_name, 
                                f"{grid_images_prefix}{format_classes_ids_str(classes_ids_list)}.{grid_images_ext}"
                            ) 
                        ]
            if batch_sizes[network_name] >= len(classes_ids_list) else
                        [
                           os.path.join(val_images_dir, network_name, image_name)
                           for image_name in os.listdir( os.path.join(val_images_dir, network_name) ) 
                           if image_name.endswith(f".{grid_images_ext}")
                        ]
        for network_name in networks_names
    }


def split_comparison_val_image(grid_path: str,
                               output_dir: str,
                               img_size: tuple[int, int],
                               batch_size: int,
                               select_best_n: int | None = None,
                               n_images_per_row: int | None = 2,
                               origs_dirname: str | None = "origs",
                               recons_dirname: str | None = "recons"):
    """
    Split a comparison grid image (original / reconstructed interleaved) into individual samples.

    The grid is assumed to contain `2 * select_best_n` sub-images of size 
    `img_size x img_size`, arranged in rows with `n_images_per_row` images per row. 
    The sub-images are ordered as: `orig_0, recon_0, orig_1, recon_1, ..., orig_N-1, recon_N-1`

    Output files are saved in folders `output_dir/origs_dirname`, `output_dir/recons_dirname` accordingly.

    Args:
        grid_path:        Path to the composite PNG file.
        output_dir:       Directory where the individual images will be saved.
        img_size:         Width and height of a single sub-image.
        select_best_n:    Number of best sample pairs (original + reconstructed) in the grid.
        n_images_per_row: Number of images per row in the grid. If `None`, computed as
                          `total_width // img_size`.
    """
    origs_dir = os.path.join(output_dir, origs_dirname)
    recons_dir = os.path.join(output_dir, recons_dirname)
    os.makedirs(origs_dir, exist_ok=True)
    os.makedirs(recons_dir, exist_ok=True)

    grid_img = Image.open(grid_path)
    grid_filename = os.path.splitext(os.path.basename(grid_path))[0]

    for i in range(min(select_best_n, batch_size) if select_best_n else batch_size):
        # Position of the original image (even index in the flat list)
        flat_orig_i = 2 * i
        row_orig = flat_orig_i // n_images_per_row
        col_orig = flat_orig_i % n_images_per_row
        x_orig = col_orig * img_size
        y_orig = row_orig * img_size

        # Crop original
        orig_crop = grid_img.crop((x_orig, y_orig, x_orig + img_size, y_orig + img_size))
        orig_crop.save(
            os.path.join(origs_dir, f"{grid_filename}_{i:04d}.png")
        )

        # Position of the reconstructed image (odd index)
        flat_recon_i = 2 * i + 1
        row_recon = flat_recon_i // n_images_per_row
        col_recon = flat_recon_i % n_images_per_row
        x_recon = col_recon * img_size
        y_recon = row_recon * img_size

        # Crop reconstructed
        recon_crop = grid_img.crop((x_recon, y_recon, x_recon + img_size, y_recon + img_size))
        recon_crop.save(
            os.path.join(recons_dir, f"{grid_filename}_{i:04d}.png")
        )


def get_labels_for_val_from_batch(classes_ids_list: list,
                                  batch_size: int,
                                  select_best_n: int | None = None) -> list:
    """
    Generate ground truth labels for validation based on the balanced selection logic 
    of `main_pipeline`.

    Args:
        classes_ids_list: List of class IDs to balance.
        batch_size:       Total number of images in the batch (capped by `select_best_n`).
        select_best_n:    Number of best sample pairs selected.

    Returns:
        List of integer class labels in the order they appear in the split grid.
    """
    labels = {}
    select_best_n = min(batch_size, select_best_n) if select_best_n else batch_size
    n_samples_base = select_best_n // len(classes_ids_list)
    remainder = select_best_n % len(classes_ids_list)
    
    for class_id_i, class_id in enumerate(classes_ids_list):        
        n_class_samples = n_samples_base + int(class_id_i < remainder)
        labels.extend([class_id] * n_class_samples)
        
    return labels


def get_labels_for_val_from_names(images_path: list,
                                  extract_label_func: Callable | None = None,
                                  **extract_label_args) -> dict:
    """
    Generate integer class labels for validation images by extracting them from file names.

    Args:
        images_path:          Path to a directory containing validation images.
        extract_label_func:   Callable receiving the file name (and any additional keyword 
                              arguments) and returns an integer class label. 
                              If `None`, image filename is considered having a scheme like
                              `<key arg>_<value>_<class_label>_....<extension>` and parsed
                              accordingly.
        **extract_label_args: Additional keyword arguments passed to `extract_label_func`.

    Returns:
        List of integer class labels corresponding to each image in the directory.
    """
    labels = {}
    for image_path in os.listdir(images_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        labels[image_name] = extract_label_func(image_name, **extract_label_args) \
            if extract_label_func else \
                             image_path.split("_")[2]
    return labels


def sort_labels_by_images(labels_true_networks: list[dict],
                          orig_images_dirs: list[str]) -> list[list]:
    return  [
                [
                    network_labels[
                        orig_image_name.rsplit('.', 1)[0] 
                        if ('.' in orig_image_name) else 
                        orig_image_name
                    ] 
                    for orig_image_name in os.listdir(network_orig_images_dir)
                ] 
                for network_labels, network_orig_images_dir in zip(labels_true_networks, orig_images_dirs)
            ]
    


def val_model_on_files(network_name: str, 
                       images_paths: str | list[str], 
                       val_by_paths: bool,
                       labels_true: list[str], 
                       batch_size: int | None = 32,
                       num_workers: int | None = 8, 
                       top_k_preds: int | None = 1,
                       get_top2_gap: bool | None = False,
                       print_details: bool | None = True) -> tuple[float, int, int]:
    """
    Load saved images from disk and evaluate classification accuracy

    Args:
        network_name:  Name of PyTorch model to be evaluated.
        images_paths:  Directory containing images as files or exact paths to images files.
        val_by_paths:  If `False`, validates network on all files in directory given
                       (PLEASE ENSURE `images_paths` IS A PATH TO DIRECTORY, NOT FILE!)
                       Otherwise iterates over exact paths
        labels_true:   List or array of true class indices, aligned with the sorted image file list.
        batch_size:    Batch size for inference.
        num_workers:   Number of subprocesses for data loading.
        top_k_preds:   Number of top predictions to display when `print_details=True`.
        get_top2_gap:  If `True`, calculate difference between top-2 predictions probabilities.
        print_details: If `True`, print file name, true class, and `top_k` predictions with probabilities
                       for every image.

    Returns:
        Tuple(accuracy, correct samples, total samples, probability gap between top2 predictions)
    """ 
    if val_by_paths:
        images_paths_sorted = images_paths
        labels_true_sorted = labels_true
    else:
        images_paths_listed = os.listdir(images_paths)
    
        sorted_images_ids = sorted(
            range(len(images_paths_listed)), 
            key=lambda i: images_paths_listed[i]
        )
        images_paths_sorted = [ images_paths_listed[i] for i in sorted_images_ids ]
        images_paths_sorted = [ os.path.join(images_paths, image_path) for image_path in images_paths_sorted ]
        labels_true_sorted =  [ labels_true[i]  for i in sorted_images_ids ]    
        
         
    transform = transforms.Compose([
        transforms.Resize(IMAGENET_CONSTANTS["size_resize"]),
        transforms.CenterCrop(IMAGENET_CONSTANTS["size_center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=IMAGENET_CONSTANTS["mean"].tolist(),
        #                      std=IMAGENET_CONSTANTS["std"].tolist())  # TODO: CHECK IF Normalize IS NEEDED!
    ])
    dataset = ImageFolderDataset(images_paths_sorted, labels_true_sorted, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = get_model_and_features(network_name)
    model = model.to(device)
    model.eval()

    correct, total, batch_start_i = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, k=top_k_preds, dim=1)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if print_details:
                for i in range(len(imgs)):
                    global_idx = batch_start_i + i
                    image_name = os.path.basename(images_paths_sorted[global_idx])
                    print(
                        f"File: {image_name}\n" 
                        f"    True class: {labels_true_sorted[global_idx]}\n"
                        f"    Top-{top_k_preds} predictions: {[ \
                            f'class {idx} ({prob:.8f})' for idx, prob in zip(topk_indices[i], topk_probs[i]) \
                        ]}"
                    )
            batch_start_i += len(imgs)

    accuracy = correct / total
    print(f"Classification accuracy: {correct}/{total} ({accuracy:.2%})")
    top2_prob_gap = topk_probs[0] - topk_probs[1] if get_top2_gap else None

    return accuracy, correct, total, top2_prob_gap


def val_model_orig_recon(network_name: str,
                         orig_paths: str | list[str],
                         recon_paths: str | list[str],
                         per_subset: bool,
                         labels_true: list[str], 
                         class_info: str | None = "",
                         num_workers: int | None = 8, 
                         top_k_preds: int | None = 1,
                         get_top2_gap: bool | None = False,
                         print_details: bool | None = True) -> dict:
    """
    Validate a specific subset of images.
    
    Args:
        network_name:  Name of PyTorch model to be evaluated.
        orig_paths:    Directory containing original images as files or exact paths to images files.
        recon_paths:   Directory containing reconstructed images as files or exact paths to images files.
        per_subset:    If `False`, validates network on all files in directory given
                       (PLEASE ENSURE `images_paths` IS A PATH TO DIRECTORY, NOT FILE!)
                       Otherwise iterates over exact paths
        labels_true:   List or array of true class indices, aligned with the sorted image file list.
        class_info:    String identifying the subset (e.g., `"bs=4, class=100"`).
        num_workers:   Number of subprocesses for data loading.
        top_k_preds:   Number of top predictions to display when `print_details=True`.
        print_details: If `True`, print file name, true class, and `top_k` predictions with probabilities
                       for every image.

    Returns:
        Classification accuracy & number of correctly classified images for the network
    """
    subset_comment = f" on subset: {class_info}" if per_subset else ""
    print(f"\nValidating {network_name}{subset_comment}...")
    
    classes_ids_n = len(set(labels_true))

    print("\n    Original images validation:")
    acc_orig, correct_orig, total_orig, top2_gap_orig = val_model_on_files(
        network_name, 
        orig_paths, 
        per_subset,
        labels_true, 
        max(MODELS_CONFIGS[network_name]["batch_size"], classes_ids_n),
        num_workers, 
        top_k_preds,
        get_top2_gap,
        print_details,
    )
    print("\n    Reconstructed images validation:")
    acc_recon, correct_recon, total_recon, top2_gap_recon = val_model_on_files(
        network_name, 
        recon_paths, 
        per_subset,
        labels_true, 
        max(MODELS_CONFIGS[network_name]["batch_size"], classes_ids_n),
        num_workers, 
        top_k_preds,
        get_top2_gap,
        print_details
    )
        
    val_results = {
        "orig_accuracy":  acc_orig * 100,
        "orig_correct":   correct_orig,
        "recon_accuracy": acc_recon * 100,
        "recon_correct":  correct_recon,
        "total_images":   total_orig,
    }
    if get_top2_gap:
        val_results["orig_top2_gap"] = top2_gap_orig
        val_results["recon_top2_gap"] = top2_gap_recon
    return val_results

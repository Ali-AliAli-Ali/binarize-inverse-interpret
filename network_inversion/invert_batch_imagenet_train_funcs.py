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
import numpy as np
from math import sqrt, pi, erf
from typing import Literal

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
           (dh.abs().sum() + dw.abs().sum() + dhr.abs().sum() + dhb.abs().sum() + dwr.abs().sum() + dwb.abs().sum()) / x.shape[0]


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


def denormalize_and_process(x: torch.Tensor, 
                            mean: torch.Tensor,
                            std: torch.Tensor, 
                            with_hsv_normalize: bool | None = True):
    """
    Denormalize a tensor and apply contrast/saturation normalization.

    Args:
        x:    Input tensor (usually in the range of the denormalization transform).
        mean: Mean used for normalization (shape (3,) or (1,3,1,1)).
        std:  Standard deviation used for normalization.

    Returns:
        Denormalised tensor clamped to [0,1] and processed by `normalize_contrast_saturation`.
    """
    return normalize_contrast_saturation( (x * std + mean).clamp_(0., 1.) ) \
        if with_hsv_normalize else \
           (x * std + mean).clamp_(0., 1.) \


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
    recon_denorm = denormalize_and_process(x_best, mean, std, False)
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
                  run_mode: Literal["debug", "run"] | None = "run",
                  seed: int = 42):
    """
    Run feature inversion for ImageNet classes using specified hyperparameters.

    Args:
        data_dir:      Path to the ImageNet validation set root folder.
        class_ids_str: ImageNet class index(es). Can be:
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"\nStart inversion of {network_name} network on ImageNet dataset...\n")

    # Parse comma-separated class IDs
    try:
        class_ids = [ int(class_id.strip()) for class_id in class_ids_str.split(',') ]
    except ValueError:
        raise ValueError(f"Invalid class id format: '{class_ids_str}'. Expected comma-separated integers or '-1' for all classes.")

    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mean = IMAGENET_CONSTANTS["mean"].to(device)
    std = IMAGENET_CONSTANTS["std"].to(device)
    mean_awb = IMAGENET_CONSTANTS["mean_awb"].to(device)


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
        # # Remove CLS token (assumed to be the first token)
        # patch_tokens = output[:, 1:, :]                     # (B, num_patches, D)
        # B, N, D = patch_tokens.shape
        # grid_size = int(N ** 0.5)  # Calculate grid size (square root of num_patches)
        #                            # Reshape: (B, grid, grid, D) -> (B, D, grid, grid)
        # activation['feat'] = patch_tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)    
        activation['feat'] = output[:, 0, :] .unsqueeze(-1).unsqueeze(-1)  # cls_token -> (B, D, 1, 1)
        
    if "vit_" in network_name:
        # model.encoder.pos_embedding.data.zero_()          # corrects reconstr colors
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
    
    imgset = ImageNet(root=data_dir, split="val", transform=transform_raw)
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
        mean, 
        std, 
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
    
    imgs_norm = (imgs - mean) / std  # batch normalization with global constants
    
    
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
    
    # logger = TrainingLogger(
    #     metric_names=["kl_loss", "mse_loss", "tv_loss", "l1_loss", "centering_loss", "border_loss", "total_loss"],
    #     log_file=os.path.join(out_dir, "training_log.npz"),
    #     max_steps=steps
    # )
    
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
        
        # logger.log(step, loss_kl, loss_mse, loss_tv, loss_l1, centering_loss, border_loss, total_loss)
        
        if (run_mode == "debug") and (not step % n_steps_log):
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
                    mean_awb, 
                    std, 
                    os.path.join(out_dir, f"recon_{step:04d}.png"), 
                    nrow=nrow
                )

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
    recon_time = time.perf_counter() - recon_start
    print(f"Reconstruction done in {recon_time:.3f} s. Average time for step: {recon_time / steps:.3f} s")

    # logger.close()

    # Check classification of all reconstructed images with original model
    with torch.no_grad():
        feat = forward_and_get_feat(model, x, activation)
        
        # Get true class labels for all samples
        all_labels_true = torch.tensor([imgset.targets[sorted_indices[i]] for i in range(len(x))], device=device)
        
        # x is already in normalized space, classify directly
        logits_recon = model_cls(x)
        probs_recon = F.softmax(logits_recon, dim=1)
        
        # Get top-2 logits & probs
        top2_logits_recon, top2_indices_recon = torch.topk(logits_recon, k=2, dim=1)
        top1_logits = top2_logits_recon[:, 0]
        top2_logits = top2_logits_recon[:, 1]
        top1_probs = probs_recon.gather(1, top2_indices_recon[:, :1]).squeeze()
        top2_probs = probs_recon.gather(1, top2_indices_recon[:, 1:]).squeeze()
        
        preds_recon = top2_indices_recon[:, 0]  # logits_recon.argmax(dim=1)
        correct_mask = (preds_recon == all_labels_true)
        correct_recon = correct_mask.sum().item()
    
    # Print classification statistics
    print("\nClassification accuracy (all samples):"
              f"    Reconstructed images: {correct_recon}/{len(all_labels_true)} ({100*correct_recon/len(all_labels_true):.1f}%)\n")
    # Separate mean statistics for correct and incorrect predictions
    if correct_recon:
        print("    Mean statistics for CORRECT reconstructions:\n"
              f"        Top-1 logit:       {top1_logits[correct_mask].mean().item():.4f}\n"
              f"        Top-1 probability: {top1_probs[correct_mask].mean().item():.4f}\n"
              f"        Top-2 logit:       {top2_logits[correct_mask].mean().item():.4f}\n"
              f"        Top-2 probability: {top2_probs[correct_mask].mean().item():.4f}\n"
              f"        Logit difference:  {(top1_logits[correct_mask] - top2_logits[correct_mask]).mean().item():.4f}")
    if correct_recon < len(all_labels_true):
        print("    Mean statistics for INCORRECT reconstructions:")
        incorrect_mask = ~correct_mask
        print(f"        Top-1 logit:      {top1_logits[incorrect_mask].mean().item():.4f}\n"
              f"        Top-1 probability: {top1_probs[incorrect_mask].mean().item():.4f}\n"
              f"        Top-2 logit:      {top2_logits[incorrect_mask].mean().item():.4f}\n"
              f"        Top-2 probability: {top2_probs[incorrect_mask].mean().item():.4f}\n"
              f"        Logit difference:  {(top1_logits[incorrect_mask] - top2_logits[incorrect_mask]).mean().item():.4f}")
    
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
        save_best_images(x_best, imgs_best, mean_awb, std, out_dir, class_id_str, nrow=nrow)
        
        # Print loss statistics
        print(f"\nBest samples total_loss range: [{loss_per_sample[best_indices].min().item():<12.6f}, {loss_per_sample[best_indices].max().item():<12.6f}]")
        print(f"Mean total_loss for best samples: {loss_per_sample[best_indices].mean().item():<12.6f}")

    
    # 8. Clean model from hooks
    
    for h in hooks.values():
        h.remove()

    print("\nDone!\n")


def main_pipeline_fast(data_dir: str | None = './data/imagenet',
                       class_ids_fixed: str | None = "-1",
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
                       seed: int | None = 42):
    """
    Faster version of feature inversion for ImageNet classes using specified hyperparameters
    without logging time, metrics or metrics except total.

    Args:
        data_dir:        Path to the ImageNet validation set root folder.
        class_ids_fixed: ImageNet class index(es). Can be:
                           - a single integer as string, e.g. '42'
                           - comma-separated list, e.g. '0,1,2'
                           - '-1' to process all classes.
        network_name:    Name of the neural network architecture to invert.
        batch_size:      Number of samples per batch during optimization.
        num_workers:     Number of subprocesses for data loading.
        steps:           Number of optimization iterations (gradient steps) per image.
        lr:              Learning rate for the optimiser.
        sigma:         Standard deviation of Gaussian noise used for initial image guess.
        beta:          Beta parameter of the Softplus activation (replaces ReLU).
        tv_weight:     Weight of the total variation (TV) regularization term.
        l2_weight:     Weight of the L2 (MSE) loss between the target and predicted features.
        l1_weight:     Weight of the L1 (Lasso) regularization term on the reconstructed image.
        seed:          Random seed to fix state
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"\nStart inversion of {network_name} network on fixed subset...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = IMAGENET_CONSTANTS["mean"].to(device)
    std = IMAGENET_CONSTANTS["std"].to(device)

    model, features = get_model_and_features(network_name)
    replace_relu_with_softplus(model, beta=beta)
    model = model.to(device).eval()

    activation = {}
    if "vit_" in network_name:
        def vit_hook(module, input, output):
            activation['feat'] = output[:, 0, :].unsqueeze(-1).unsqueeze(-1)
        hook = model.encoder.register_forward_hook(vit_hook)
    else:
        def hook_fn(module, input, output):
            activation['feat'] = output
        hook = features.register_forward_hook(hook_fn)

    transform = transforms.Compose([
        transforms.Resize(IMAGENET_CONSTANTS["size_resize"]),
        transforms.CenterCrop(IMAGENET_CONSTANTS["size_center_crop"]),
        transforms.ToTensor()
    ])
    dataset = ImageNet(root=data_dir, split="val", transform=transform)
    subset = Subset(dataset, class_ids_fixed)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)
    labels = labels.to(device)

    imgs_norm = (imgs - mean) / std

    with torch.no_grad():
        _ = model(imgs_norm)
        target_feat = activation['feat'].detach()

    noise = torch.randn_like(imgs_norm) * sigma
    x = noise.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    for step in range(steps):
        optimizer.zero_grad()
        _ = model(x)
        feat = activation['feat']
        loss = compute_all_losses(
            feat, 
            target_feat, 
            x, 
            tv_weight=tv_weight, 
            l2_weight=l2_weight, 
            l1_weight=l1_weight
        )[-1]
        loss.backward()
        optimizer.step()
        scheduler.step()

    model_cls, _ = get_model_and_features(network_name)
    model_cls.to(device).eval()
    with torch.no_grad():
        logits = model_cls(x)
        top2_logits, top2_idx = torch.topk(logits, 2, dim=1)
        top1_logits = top2_logits[:, 0]
        top2_logits = top2_logits[:, 1]
        
        probs = torch.softmax(logits, dim=1)
        
        preds = top2_idx[:, 0]
        preds_correct = (preds == labels)

        acc = preds_correct.float().mean().item()
        if preds_correct.any():
            gap_correct = (top1_logits[preds_correct] - top2_logits[preds_correct]).mean().item()
            mean_prob_correct = probs[preds_correct, preds[preds_correct]].mean().item()
        else:
            gap_correct = float("nan")
            mean_prob_correct = float("nan")

    hook.remove()
    return {
        "accuracy": acc,
        "mean_logit_gap_correct": gap_correct,
        "mean_prob_correct": mean_prob_correct,
        "top1_logits": top1_logits.cpu().numpy(),
        "correct_mask": preds_correct.cpu().numpy()
    }


def sample_random_indices(dataset, 
                          subset_size: int | None = 10,
                          class_ids: list | None = None,
                          seed: int | None = 42) -> list:

    rng = np.random.RandomState(seed)
    if class_ids is not None and class_ids != [-1]:
        valid_indices = [i for i, (_, label) in enumerate(dataset) if label in class_ids]
    else:
        valid_indices = list(range(len(dataset)))
    chosen = rng.choice(valid_indices, size=min(subset_size, len(valid_indices)), replace=False)
    return chosen.tolist()


def evaluate_inversion_on_random_subset(data_dir: str | None = './data/imagenet',
                                        network_name: str | None = "regnet_x_3_2",
                                        subset_size: int | None = 10,
                                        num_repeats: int | None = 5,
                                        steps: int | None = 4000,
                                        class_ids: int | None = None,  
                                        lr: float | None = 0.1,
                                        sigma: float | None = 0.01,
                                        beta: float | None = 4,
                                        tv_weight: float | None = 5e-5,
                                        l2_weight: float | None = 10.0,
                                        l1_weight: float | None = 0.0,
                                        seed: int | None = 42) -> list:

    transform = transforms.Compose([
        transforms.Resize(IMAGENET_CONSTANTS["size_resize"]),
        transforms.CenterCrop(IMAGENET_CONSTANTS["size_center_crop"]),
        transforms.ToTensor()
    ])
    dataset = ImageNet(root=data_dir, split="val", transform=transform)
    fixed_indices = sample_random_indices(dataset, subset_size, class_ids=class_ids, seed=seed)
    print(f"Fixed random subset of length {len(fixed_indices)} images with indices:\n"
          f"    {fixed_indices}")
    
    batch_size = MODELS_CONFIGS[network_name]["batch_size"]
    num_batches = len(fixed_indices) // batch_size
    if not num_batches:
        num_batches = 1
    used_indices = fixed_indices[:num_batches * batch_size]
    print(f"Using {len(used_indices)} images ({num_batches} batches of {batch_size})")

    batch_indices = [ used_indices[i*batch_size : (i+1)*batch_size] for i in range(num_batches) ]

    results_per_repeat = []
    for rep in range(num_repeats):
        seed_rep = seed + rep * 1000
        
        batch_metrics = []
        for batch_idx, batch_inds in enumerate(batch_indices):
            seed_batch = seed_rep + batch_idx
            
            batch_metrics.append(
                main_pipeline_fast(
                    data_dir=data_dir,
                    class_ids_fixed=batch_inds,
                    network_name=network_name,
                    batch_size=batch_size,
                    steps=steps,
                    lr=lr,
                    sigma=sigma,
                    beta=beta,
                    tv_weight=tv_weight,
                    l2_weight=l2_weight,
                    l1_weight=l1_weight,
                    seed=seed_batch
            ))

        avg_acc = np.mean([m["accuracy"] for m in batch_metrics])

        gaps = [m["mean_logit_gap_correct"] for m in batch_metrics if not np.isnan(m["mean_logit_gap_correct"])]
        avg_gap = np.mean(gaps) if gaps else float('nan')
        probs = [m["mean_prob_correct"] for m in batch_metrics if not np.isnan(m["mean_prob_correct"])]
        avg_prob = np.mean(probs) if probs else float('nan')

        aggregated = {
            "accuracy": avg_acc,
            "mean_logit_gap_correct": avg_gap,
            "mean_prob_correct": avg_prob,
        }
        results_per_repeat.append(aggregated)
        print(f"Repeat {rep+1}/{num_repeats} | Acc={avg_acc:.3f} | Logit gap={avg_gap:.4f}")

    # Итоговая статистика по повторам
    accs = [r["accuracy"] for r in results_per_repeat]
    gaps = [r["mean_logit_gap_correct"] for r in results_per_repeat if not np.isnan(r["mean_logit_gap_correct"])]

    print(f"Accuracy:  {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    if gaps:
        print(f"Logit gap: {np.mean(gaps):.4f} +- {np.std(gaps):.4f}")
    else:
        print("No correct predictions to compute logit gap.")
    return results_per_repeat

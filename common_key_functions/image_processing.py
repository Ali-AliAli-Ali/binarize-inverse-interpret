import torch


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to HSV.
    
    Args:
        rgb: (B, 3, H, W) tensor in [0, 1]
        
    Returns:
        hsv: (B, 3, H, W) tensor with H in [0, 1], S in [0, 1], V in [0, 1]
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    max_val, max_idx = torch.max(rgb, dim=1)
    min_val = torch.min(rgb, dim=1)[0]
    delta = max_val - min_val
    
    v = max_val                                                                   # value (brightness)
    s = torch.where(max_val > 1e-6, delta / max_val, torch.zeros_like(max_val))   # saturation
    h = torch.zeros_like(max_val)                                                 # hue
    
    mask_r = (max_idx == 0) & (delta > 1e-6)
    mask_g = (max_idx == 1) & (delta > 1e-6)
    mask_b = (max_idx == 2) & (delta > 1e-6)
    
    h[mask_r] = (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6) / 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2) / 6.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4) / 6.0
    
    return torch.stack([h, s, v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV tensor to RGB.
    
    Args:
        hsv: (B, 3, H, W) tensor with H in [0, 1], S in [0, 1], V in [0, 1]
        
    Returns:
        rgb: (B, 3, H, W) tensor in [0, 1] range
    """
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c
    
    h6 = (h * 6) % 6
    sector = h6.floor().long()
    
    # Initialize RGB channels
    r = torch.zeros_like(c)
    g = torch.zeros_like(c)
    b = torch.zeros_like(c)
    
    # Map sectors to RGB values
    mask0 = (sector == 0)
    mask1 = (sector == 1)
    mask2 = (sector == 2)
    mask3 = (sector == 3)
    mask4 = (sector == 4)
    mask5 = (sector == 5)
    
    r[mask0] = c[mask0]
    g[mask0] = x[mask0]
    b[mask0] = 0
    
    r[mask1] = x[mask1]
    g[mask1] = c[mask1]
    b[mask1] = 0
    
    r[mask2] = 0
    g[mask2] = c[mask2]
    b[mask2] = x[mask2]
    
    r[mask3] = 0
    g[mask3] = x[mask3]
    b[mask3] = c[mask3]
    
    r[mask4] = x[mask4]
    g[mask4] = 0
    b[mask4] = c[mask4]
    
    r[mask5] = c[mask5]
    g[mask5] = 0
    b[mask5] = x[mask5]
    
    rgb = torch.stack([r + m, g + m, b + m], dim=1)
    return rgb.clamp_(0., 1.)


def grayworld_white_balance(batch: torch.Tensor) -> torch.Tensor:
    """
    Apply gray world white balance to each image in batch.
    The gray world assumption states that the average color of an image should be gray.
    This neutralizes color casts by scaling RGB channels so their means are equal.
    
    Args:
        batch: (B, 3, H, W) tensor in [0, 1] range
        
    Returns:
        White-balanced batch with neutralized color casts
    """
    # Compute mean of each RGB channel for each image: (B, 3)
    r_mean = batch[:, 0, :, :].mean(dim=[1, 2])  # (B,)
    g_mean = batch[:, 1, :, :].mean(dim=[1, 2])  # (B,)
    b_mean = batch[:, 2, :, :].mean(dim=[1, 2])  # (B,)
    
    # Compute overall mean (average of R, G, B means) for each image: (B,)
    overall_mean = (r_mean + g_mean + b_mean) / 3.0
    
    # Compute scaling factors to make each channel mean equal to overall mean
    # Avoid division by zero
    r_scale = (overall_mean / r_mean.clamp(min=1e-6)).view(batch.size(0), 1, 1)
    g_scale = (overall_mean / g_mean.clamp(min=1e-6)).view(batch.size(0), 1, 1)
    b_scale = (overall_mean / b_mean.clamp(min=1e-6)).view(batch.size(0), 1, 1)
    
    # Apply scaling to each channel (broadcasting works with (B, 1, 1) * (B, H, W))
    balanced = torch.stack([
        batch[:, 0, :, :] * r_scale,
        batch[:, 1, :, :] * g_scale,
        batch[:, 2, :, :] * b_scale
    ], dim=1)
    
    return balanced.clamp_(0., 1.)


def normalize_contrast_saturation(batch: torch.Tensor) -> torch.Tensor:
    """
    Normalize each image in batch by saturation (S) and value (V) in HSV space.
    Converts RGB to HSV, normalizes S and V using 0.01 and 0.99 quantiles (keeping original hue),
    then converts back to RGB.
    
    Args:
        batch: (B, 3, H, W) tensor in [0, 1] range
        
    Returns:
        Normalized batch with maximized contrast and saturation per image
    """
    # Convert RGB to HSV
    hsv = rgb_to_hsv(batch)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    
    # Helper function to normalize a channel using quantiles
    def normalize_channel(channel, l, h):
        channel_flat = channel.view(batch.size(0), -1)
        q01 = torch.quantile(channel_flat, l, dim=1, keepdim=True).view(batch.size(0), 1, 1)
        q99 = torch.quantile(channel_flat, h, dim=1, keepdim=True).view(batch.size(0), 1, 1)
        channel_range = (q99 - q01).clamp(min=5e-2)
        return ((channel - q01) / channel_range).clamp_(0., 1.)
    
    # Normalize S (saturation) and V (value/brightness) for each image
    # Keep original H (hue) - hue shifting causes color cast issues
    s_normalized = normalize_channel(s, 0.0, 1.0)
    v_normalized = normalize_channel(v, 0.05, 0.95).clamp(min=1e-6)

    # Apply gamma 
    mean_v_n = v_normalized.mean(dim=[1, 2], keepdim=True)
    mean_v = v.mean(dim=[1, 2], keepdim=True).clamp(min=1e-6)
    gamma = torch.log(mean_v) / torch.log(mean_v_n)
    gamma = gamma.clamp(0.1, 5.0)  # avoid extremes
    v_adjusted = v_normalized.pow(gamma).clamp_(0., 1.)
    
    # Reconstruct HSV with original H, normalized S and V
    hsv_normalized = torch.stack([h, s_normalized, v_adjusted], dim=1)
    
    # Convert back to RGB
    normalized = hsv_to_rgb(hsv_normalized)
    
    return normalized.clamp_(0., 1.)


def tv_loss(x: torch.Tensor, 
            per_sample: bool = False) -> torch.Tensor:
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


def gray_edge_l1(rgb):  # rgb: (B,3,H,W) already ImageNet-normalized
    """
    Compute L1 loss on the deviation of colour channels from their mean gradient.

    Args:
        rgb: Input tensor of shape (B, 3, H, W), ImageNet‑normalised

    Returns:
        Average L1 deviation of channel gradients from their channel‑wise mean.
    """
    
    dx = rgb[:,:,:,1:] - rgb[:,:,:,:-1]   # (B,3,H,W-1)
    dy = rgb[:,:,1:,:] - rgb[:,:,:-1,:]   # (B,3,H-1,W)

    def dev_from_mean(d):
        return (d - d.mean(dim=1, keepdim=True)).abs().sum(dim=1)      # L1 over channels -> (B,*,*)

    return dev_from_mean(dx).mean() + dev_from_mean(dy).mean()


from time import time
import torch
from torch.utils.data import DataLoader

from constants_configs import MODEL_CONFIGS


seed = 42
torch.manual_seed(seed)


def get_model_and_features(model_name: str) -> tuple:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported network name: {model_name}. Available networks: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model = config["builder"](weights=config["weights"])
    return model, getattr(model, config["feature_attr"])


@torch.inference_mode()
def validate_top1_top5_time(model, 
                            loader: DataLoader):
    device = next(model.parameters()).device

    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    start_time = time()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        _, preds = logits.topk(5, dim=1, largest=True, sorted=True)

        correct = preds.eq(targets.view(-1, 1))
        top1_correct += correct[:, :1].sum().item()
        top5_correct += correct.sum().item()
        total += targets.size(0)
        
    val_time = time() - start_time

    return (top1_correct / total * 100,
            top5_correct / total * 100,
            val_time)


def pretty_print_top1_top5_time(model, 
                                loader: DataLoader, 
                                top1_orig: float | None = 0,
                                top5_orig: float | None = 0):    
    top1_new, top5_new, val_time = validate_top1_top5_time(model, loader)
    
    if top1_orig:
        drop1 = top1_orig - top1_new
        drop1_report = f", drop: {drop1:.4f}"
    else:
        drop1, drop1_report = None, ""
    if top5_orig:
        drop5 = top5_orig - top5_new
        drop5_report = f", drop: {drop5:.4f}"
    else:
        drop5, drop5_report = None, ""
        
    print(f"    Top-1 Accuracy:  {top1_new:.4f}{drop1_report}")
    print(f"    Top-5 Accuracy:  {top5_new:.4f}{drop5_report}")
    print(f"    Validation time: {val_time:.4f} s")
    return top1_new, top5_new, val_time, drop1, drop5
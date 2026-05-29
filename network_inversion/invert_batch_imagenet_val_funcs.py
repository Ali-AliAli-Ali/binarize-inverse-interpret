import os
import sys
from PIL import Image
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import IMAGENET_CONSTANTS, MODELS_CONFIGS   # noqa: E402
from helpful_funcs import get_model_and_features, format_classes_ids_str            # noqa: E402



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
                          network_prefix: str | None = "",
                          network_postfix: str | None = "",
                          grid_images_prefix: str | None = "best_orig_vs_recon_",
                          grid_images_ext: str | None = "png") -> dict:
    """
    Return a dictionary mapping each network name to a list of grid image paths.

    If the network's batch size is at least the number of classes, a single path
    constructed with the formatted class IDs is returned. Otherwise, all files
    with the given extension in the network's subdirectory are returned.
    
    Args:
        networks_names:     List of network names.
        val_images_dir:     Path to directory with grid image(s). Is considered to contain directories for each
                            network named like `f"{network_prefix}{network_name}{network_postfix}"`
        classes_ids_list:   List of integer class IDs.
        batch_sizes:        Dict `{ network_name : batch_size }` for every network name in `networks_names`.
                            If `None`, default batch sizes from config are taken.
        network_prefix:     Prefix  to network name defining it's results directory (if present)
        network_postfix:    Postfix to network name defining it's results directory (if present)
        grid_images_prefix: Prefix for the grid image filename when a single path is generated.
        grid_images_ext:    File extension for the grid images (without dot).

    Returns:
        Dictionary mapping network name to a list of absolute file paths to grid images.
    """
    batch_sizes = batch_sizes or { 
        network_name : MODELS_CONFIGS[network_name]["batch_size"]
        for network_name in networks_names
    }
    
    grid_images_paths = {}
    for network_name in networks_names:
        network_dirname = network_prefix + network_name + network_postfix
        grid_images_paths[network_name] = [ 
                            os.path.join(
                                val_images_dir, 
                                network_dirname, 
                                f"{grid_images_prefix}{format_classes_ids_str(classes_ids_list)}.{grid_images_ext}"
                            ) 
                        ] \
            if batch_sizes[network_name] >= len(classes_ids_list) else \
                        [
                           os.path.join(
                               val_images_dir, 
                               network_dirname, 
                               image_name
                            )
                           for image_name in os.listdir( os.path.join(val_images_dir, network_dirname) ) 
                           if image_name.endswith(f".{grid_images_ext}")
                        ]
    return grid_images_paths


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
            if extract_label_func is not None else \
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
                       get_top2_logits_gap: bool | None = False,
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
        get_top2_logits_gap:  If `True`, calculate mean difference between top-2 predictions logits over batch.
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
        
        paired = sorted(zip(images_paths_listed, labels_true), key=lambda x: x[0])
        images_paths_sorted = [ os.path.join(images_paths, image_path) for image_path, _ in paired ]
        labels_true_sorted = [label for _, label in paired]
    
        # sorted_images_ids = sorted(
        #     range(len(images_paths_listed)), 
        #     key=lambda i: images_paths_listed[i]
        # )
        # images_paths_sorted = [ images_paths_listed[i] for i in sorted_images_ids ]
        # images_paths_sorted = [ os.path.join(images_paths, image_path) for image_path in images_paths_sorted ]
        # labels_true_sorted =  [ labels_true[i]  for i in sorted_images_ids ]    
        
         
    transform = transforms.Compose([
        transforms.Resize(IMAGENET_CONSTANTS["size_resize"]),
        transforms.CenterCrop(IMAGENET_CONSTANTS["size_center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=IMAGENET_CONSTANTS["mean"].tolist(),
        #                      std=IMAGENET_CONSTANTS["std"].tolist())
    ])
    dataset = ImageFolderDataset(images_paths_sorted, labels_true_sorted, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = get_model_and_features(network_name)
    model = model.to(device)
    model.eval()

    correct, total, batch_start_i = 0, 0, 0
    top2_logits_gaps = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            logits = model(imgs)
            if get_top2_logits_gap:
                top2_logits, _ = torch.topk(logits, k=2, dim=1)
                top2_logits_gaps.extend((top2_logits[:, 0] - top2_logits[:, 1]).to_list())
            
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
    top2_logits_gap = sum(top2_logits_gaps) / len(top2_logits_gaps) if get_top2_logits_gap else None

    return accuracy, correct, total, top2_logits_gap


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

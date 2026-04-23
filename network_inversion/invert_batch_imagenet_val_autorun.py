import os
import sys
from typing import Literal

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import MODEL_CONFIGS, IMAGENET_CONSTANTS         # noqa: E402
from invert_batch_imagenet_val_as_func import main_pipeline, val_model_on_files, \
    format_classes_ids_str, get_grid_images_paths, split_comparison_val_image, get_labels_for_val    # noqa: E402


def train_inversion(networks_names: list[str],
                    dataset_dir: str | None = "./data/imagenet",
                    dataset_name: str | None = "dataset",
                    classes_ids: str | None = "-1",
                    num_workers: int | None = 8,
                    n_training_steps: int | None = 4000, 
                    learning_rate_start: float | None = 0.1,
                    gaussian_noise_sigma: float | None = 0.01,
                    softplus_beta: float | None = 4.0,
                    total_variance_weight: float | None = 5e-5,
                    l2_reg_weight: float | None = 10.0,
                    l1_reg_weight: float | None = 0.0,
                    subset_size: float | None = 5000,
                    threshold: float | None = None,
                    select_best_n: int | None = 10,
                    n_images_per_row: int | None = 2,
                    out_dir: str | None = None,
                    run_mode: Literal["debug", "run"] | None = "run"):
    """
    Run feature inversion training for ImageNet classes on a list of neural networks.

    Iterates over the given network names and calls the main inversion pipeline
    with the provided hyperparameters, automatically constructing the output
    directory based on network name, dataset name, number of steps, and class IDs.
    
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
        run_mode:      Mode to run inversion training: `"debug"` adds debugging print statements
    """
    for network_name in networks_names:
        main_pipeline(
            dataset_dir,
            classes_ids,
            network_name,
            MODEL_CONFIGS[network_name]["batch_size"],
            num_workers,
            n_training_steps,
            learning_rate_start,
            gaussian_noise_sigma,
            softplus_beta,
            total_variance_weight,
            l2_reg_weight,
            l1_reg_weight,
            subset_size,
            threshold,
            select_best_n,
            n_images_per_row,
            out_dir or f"network_inversion/"
                       f"inversion_images_logs/"
                       f"inversion_{network_name}_{dataset_name}/"
                       f"steps_{n_training_steps}_classes_{classes_ids}",
            run_mode
        )


def val_inversion(networks_names: list[str],
                  orig_images_dirs: list[str],
                  recon_images_dirs: list[str],
                  labels_true_all: list[list],
                  num_workers: int | None = 8,
                  print_details: bool | None = True,
                  top_k_preds: int | None = 1):
    """
    Run validation on original and reconstructed images for a list of networks.

    For each network, validates both the original images and the corresponding
    reconstructions, printing classification accuracy and optional top-k predictions.

    Args:
        networks_names:    List of network architecture names.
        orig_images_dirs:  List of directories containing original images (one per network).
        recon_images_dirs: List of directories containing reconstructed images (one per network).
        labels_true_all:   List of ground truth label lists (one per network).
        num_workers:       Number of DataLoader workers.
        print_details:     Whether to print per-image prediction details.
        top_k_preds:       Number of top predictions to display when `print_details` is True.
    """
    
    for network_name, orig_images_dir, recon_images_dir, labels_true in zip(networks_names, 
                                                                            orig_images_dirs, 
                                                                            recon_images_dirs,
                                                                            labels_true_all): 
        classes_ids_n = len(set(labels_true))
        print(f"\n\nNetwork: {network_name}")
        
        print("\n    Original images validation:\n")
        acc_orig, correct_orig, total_orig = val_model_on_files(
            network_name,
            orig_images_dir,
            labels_true,
            max(MODEL_CONFIGS[network_name]["batch_size"], classes_ids_n),
            num_workers,
            print_details,
            top_k_preds
        )
        
        print("\n    Reconstructed images validation:\n")   
        acc_recon, correct_recon, total_recon = val_model_on_files(
            network_name,
            recon_images_dir,
            labels_true,
            max(MODEL_CONFIGS[network_name]["batch_size"], classes_ids_n),
            num_workers,
            print_details,
            top_k_preds
        )


if __name__ == '__main__':
    
    networks_names = ["convnext_l", "swin_v2_b"]# MODEL_CONFIGS.keys()
    
    classes_ids = "1, 10, 100, 999"
    classes_ids_int = [ int(class_id.strip()) for class_id in classes_ids.split(",") ]
    classes_ids_n = len(classes_ids_int)
    classes_ids_str = format_classes_ids_str(classes_ids_int)

    # inversion training
    
    dataset_dir_imagenet = "/media/user/Hitachi/ILSVRC/Data/CLS-LOC"
    dataset_name = "ImageNet"
    n_training_steps = 10_000
    select_best_n = 10
    
    for class_id in classes_ids_int:
        train_inversion(
            networks_names, 
            dataset_dir_imagenet, 
            dataset_name, 
            n_training_steps=n_training_steps,
            classes_ids=str(class_id),
            select_best_n=select_best_n,
            run_mode="debug"
        )
    
    # inversion validation
    
    val_images_dir = "network_inversion/inversion_images_val"
    networks_grid_images_paths = get_grid_images_paths(networks_names, val_images_dir, classes_ids_int)
    
    for network_name, grid_images_paths in networks_grid_images_paths.items():
        for grid_image_i, grid_image_path in enumerate(grid_images_paths):    
            split_comparison_val_image(
                grid_image_path,
                os.path.dirname(grid_image_path),
                IMAGENET_CONSTANTS["size_center_crop"] + 2,     # empirically adjusted padding
                MODEL_CONFIGS[network_name]["batch_size"],
                sample_number_shift=grid_image_i
            )

    val_inversion(
        networks_names,
        [ os.path.join(val_images_dir, network_name, "origs")  for network_name in networks_names ],
        [ os.path.join(val_images_dir, network_name, "recons") for network_name in networks_names ],
        [
            get_labels_for_val(
                classes_ids_int, 
                max(MODEL_CONFIGS[network_name]["batch_size"], classes_ids_n), 
                select_best_n
            )
            for network_name in networks_names
        ],
        top_k_preds=5
    )
    
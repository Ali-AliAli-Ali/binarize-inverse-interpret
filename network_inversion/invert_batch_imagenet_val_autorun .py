import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import MODELS_CONFIGS, IMAGENET_CONSTANTS         # noqa: E402
from network_inversion.invert_batch_imagenet_funcs import val_model_on_files, \
    format_classes_ids_str, get_grid_images_paths, split_comparison_val_image, get_labels_for_val    # noqa: E402


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
            max(MODELS_CONFIGS[network_name]["batch_size"], classes_ids_n),
            num_workers,
            print_details,
            top_k_preds
        )
        
        print("\n    Reconstructed images validation:\n")   
        acc_recon, correct_recon, total_recon = val_model_on_files(
            network_name,
            recon_images_dir,
            labels_true,
            max(MODELS_CONFIGS[network_name]["batch_size"], classes_ids_n),
            num_workers,
            print_details,
            top_k_preds
        )


if __name__ == '__main__':
    
    networks_names = ["convnext_l", "swin_v2_b"]# MODELS_CONFIGS.keys()
    
    classes_ids = "1, 10, 100, 999"
    classes_ids_int = [ int(class_id.strip()) for class_id in classes_ids.split(",") ]
    classes_ids_n = len(classes_ids_int)
    classes_ids_str = format_classes_ids_str(classes_ids_int)

    # inversion training
    
    dataset_dir_imagenet = "/media/user/Hitachi/ILSVRC/Data/CLS-LOC"
    dataset_name = "ImageNet"
    select_best_n = 10
    
    # inversion validation
    
    val_images_dir = "network_inversion/inversion_images_val"
    networks_grid_images_paths = get_grid_images_paths(networks_names, val_images_dir, classes_ids_int)
    
    for network_name, grid_images_paths in networks_grid_images_paths.items():
        for grid_image_i, grid_image_path in enumerate(grid_images_paths):    
            split_comparison_val_image(
                grid_image_path,
                os.path.dirname(grid_image_path),
                IMAGENET_CONSTANTS["size_center_crop"] + 2,     # empirically adjusted padding
                MODELS_CONFIGS[network_name]["batch_size"],
                sample_number_shift=grid_image_i
            )

    val_inversion(
        networks_names,
        [ os.path.join(val_images_dir, network_name, "origs")  for network_name in networks_names ],
        [ os.path.join(val_images_dir, network_name, "recons") for network_name in networks_names ],
        [
            get_labels_for_val(
                classes_ids_int, 
                max(MODELS_CONFIGS[network_name]["batch_size"], classes_ids_n), 
                select_best_n
            )
            for network_name in networks_names
        ],
        top_k_preds=5
    )
    
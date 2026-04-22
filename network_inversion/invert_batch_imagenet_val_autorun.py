import os
import sys
from typing import Literal
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import MODEL_CONFIGS, IMAGENET_CONSTANTS                    # noqa: E402
from invert_batch_imagenet_val_as_func import main_pipeline, val_model_on_files    # noqa: E402


def format_classes_ids_str(classes_ids_list: list) -> str:
    return '_'.join(f'{class_id:04d}' for class_id in classes_ids_list)


def train_inversion(networks_names: list[str],
                    dataset_dir: str | None = './data/imagenet',
                    dataset_name: str |  None = "dataset",
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
    for network_name in networks_names:
        main_pipeline({
            "data_dir":      dataset_dir,
            "class_ids":     classes_ids,
            "network":       network_name,
            "batch_size":    MODEL_CONFIGS[network_name]["batch_size"],
            "num_workers":   num_workers,
            "steps":         n_training_steps,
            "lr":            learning_rate_start,
            "sigma":         gaussian_noise_sigma,
            "beta":          softplus_beta,
            "tv_weight":     total_variance_weight,
            "l2_weight":     l2_reg_weight,
            "l1_weight":     l1_reg_weight,
            "subset_size":   subset_size,
            "threshold":     threshold,
            "select_best_n": select_best_n,
            "nrow":          n_images_per_row,
            "out_dir":       out_dir or f"network_inversion/"
                                        f"inversion_images_logs/"
                                        f"inversion_{network_name}_{dataset_name}/"
                                        f"steps_{n_training_steps}_classes_{classes_ids}",
            "run_mode":      run_mode
        })


def get_grid_images_paths(networks_names: list[str],
                          val_images_dir: str,
                          classes_ids_list: list[int],
                          grid_images_prefix: str | None = "best_orig_vs_recon_",
                          grid_images_ext: str | None = "png"):
    return {
        network_name :  [ 
                            os.path.join(
                                val_images_dir, 
                                network_name, 
                                f"{grid_images_prefix}{format_classes_ids_str(classes_ids_list)}.{grid_images_ext}"
                            ) 
                        ]
            if MODEL_CONFIGS[network_name]["batch_size"] >= len(classes_ids_int) else
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
                               select_best_n: int | None = 10,
                               n_images_per_row: int | None = 2,
                               origs_dirname: str | None = "origs",
                               recons_dirname: str | None = "recons",
                               sample_number_shift: int | None = 0):
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

    for i in range(min(select_best_n, batch_size)):
        # Position of the original image (even index in the flat list)
        flat_orig_i = 2 * i
        row_orig = flat_orig_i // n_images_per_row
        col_orig = flat_orig_i % n_images_per_row
        x_orig = col_orig * img_size
        y_orig = row_orig * img_size

        # Crop original
        orig_crop = grid_img.crop((x_orig, y_orig, x_orig + img_size, y_orig + img_size))
        orig_crop.save(
            os.path.join(origs_dir, f"sample_{i + sample_number_shift:04d}.png")
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
            os.path.join(recons_dir, f"sample_{i + sample_number_shift:04d}.png")
        )


def get_labels_for_val(classes_ids_list: list,
                       batch_size: int,
                       select_best_n: int | None = 10) -> list:
    labels = []
    select_best_n = min(batch_size, select_best_n)
    n_samples_base = select_best_n // len(classes_ids_list)
    remainder = select_best_n % len(classes_ids_list)
    
    for class_id_i, class_id in enumerate(classes_ids_list):
        n_class_samples = min(
            n_samples_base + int(class_id_i < remainder), 
            len(classes_ids_list)
        )
        labels += [class_id] * n_class_samples
        
    return labels


def val_inversion(networks_names: list[str],
                  orig_images_dirs: list[str],
                  recon_images_dirs: list[str],
                  labels_true_all: list[list],
                  num_workers: int | None = 8,
                  print_details: bool | None = True,
                  top_k_preds: int | None = 1):
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
    
    networks_names = MODEL_CONFIGS.keys()
    
    classes_ids = "1, 10, 100, 999"
    classes_ids_int = [ int(class_id.strip()) for class_id in classes_ids.split(",") ]
    classes_ids_n = len(classes_ids_int)
    classes_ids_str = format_classes_ids_str(classes_ids_int)

    # inversion training
    
    dataset_dir_imagenet = "/media/user/Hitachi/ILSVRC/Data/CLS-LOC"
    dataset_name = "ImageNet"
    n_training_steps = 10_000
    select_best_n = 10
    
    # train_inversion(
    #     networks_names, 
    #     dataset_dir_imagenet, 
    #     dataset_name, 
    #     n_training_steps=n_training_steps,
    #     classes_ids=classes_ids,
    #     select_best_n=select_best_n,
    #     run_mode="debug"
    # )
    
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
    
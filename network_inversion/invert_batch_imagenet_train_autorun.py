import os
import sys
from typing import Literal

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import MODELS_CONFIGS            # noqa: E402
from invert_batch_imagenet_funcs import main_pipeline   # noqa: E402
from helpful_funcs import iterate_by_batch              # noqa: E402


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
                    select_best_n: int | None = None,
                    n_images_per_row: int | None = 2,
                    out_dir_all: str | None = None,
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
    out_dir_all = out_dir_all or "network_inversion/inversion_images_logs/"
    classes_ids_list = sorted([ class_id.strip() for class_id in classes_ids.split(',') ])
    
    for network_name in networks_names:
        batch_size = MODELS_CONFIGS[network_name]["batch_size"]
        
        if batch_size >= len(classes_ids_list):
            main_pipeline(
                dataset_dir,
                classes_ids,
                network_name,
                batch_size,
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
                os.path.join(
                    out_dir_all,
                    f"inversion_{network_name}_{dataset_name}/steps_{n_training_steps}_classes_{classes_ids}"
                ),
                run_mode
            )
        else:            
            for classes_ids_batch in iterate_by_batch(classes_ids_list, batch_size):
                classes_ids_batch_str = str(classes_ids_batch)[1:-1].replace("'", "").replace('"', "")
                
                main_pipeline(
                    dataset_dir,
                    classes_ids_batch_str,
                    network_name,
                    batch_size,
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
                    os.path.join(
                        out_dir_all,
                        f"inversion_{network_name}_{dataset_name}/steps_{n_training_steps}_classes_{classes_ids_batch_str}"
                    ),
                    run_mode
                )
                

def train_inversion_over_batch_size(networks_names: list[str],
                                    batch_size_options: dict[str | list[int]],
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
                                    select_best_n: int | None = None,
                                    n_images_per_row: int | None = 2,
                                    out_dir_all: str | None = None,
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
    out_dir_all = out_dir_all or "network_inversion/inversion_images_logs/"
    classes_ids_list = sorted([ class_id.strip() for class_id in classes_ids.split(',') ])
    
    for network_name in networks_names:
        for batch_size in batch_size_options[network_name]:
            if batch_size > MODELS_CONFIGS[network_name]["batch_size"]:
                print(f"Batch size {batch_size} > the maximum one in config; "
                      f"skipping it...")
        
            if batch_size >= len(classes_ids_list):
                main_pipeline(
                    dataset_dir,
                    classes_ids,
                    network_name,
                    batch_size,
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
                    os.path.join(
                        out_dir_all,
                        f"inversion_{network_name}_{dataset_name}/bs_{batch_size}_classes_{classes_ids}"
                    ),
                    run_mode
                )
            else:            
                for classes_ids_batch in iterate_by_batch(classes_ids_list, batch_size):
                    classes_ids_batch_str = str(classes_ids_batch)[1:-1].replace("'", "").replace('"', "")
                    
                    main_pipeline(
                        dataset_dir,
                        classes_ids_batch_str,
                        network_name,
                        batch_size,
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
                        os.path.join(
                            out_dir_all,
                            f"inversion_{network_name}_{dataset_name}/bs_{batch_size}_classes_{classes_ids_batch_str}"
                        ),
                        run_mode
                    )


if __name__ == '__main__':
    
    dataset_dir_imagenet = "/media/user/Hitachi/ILSVRC/Data/CLS-LOC"
    dataset_name = "ImageNet"
    
    classes_ids_test = "1, 10, 100, 999"
    classes_ids_max_gap = "24, 79, 409, 701, 712, 850, 950, 953, 954"
    classes_ids = classes_ids_test  + ", " + classes_ids_max_gap
    
    classes_ids_int = [ int(class_id.strip()) for class_id in classes_ids.split(",") ]
    classes_ids_n = len(classes_ids_int)
    
    networks_names = ["regnet_x_3_2", "regnet_x_16", "resnet50", "vit_b_16"] #, MODELS_CONFIGS.keys()
    # networks_batch_sizes = {
    #     "regnet_x_3_2" : [16, 8, 4, 2],
    #     "regnet_x_16" :  [4, 2],
    #     "resnet50":      [12, 8, 4, 2],
    #     "vit_b_16":      [4, 2]
    # }
    n_training_steps = 10_000
    select_best_n = 10

    for class_id in classes_ids_int:
        print(f"Processing class {class_id}...")
        train_inversion(
            networks_names,
            dataset_dir_imagenet, 
            dataset_name, 
            n_training_steps=n_training_steps,
            classes_ids=str(class_id),
            select_best_n=select_best_n,
            out_dir_all="network_inversion/inversion_images_logs_mse_batch_indept/",
            run_mode="run"
        )
        # train_inversion_over_batch_size(
        #     networks_names,
        #     networks_batch_sizes,
        #     dataset_dir_imagenet, 
        #     dataset_name, 
        #     n_training_steps=n_training_steps,
        #     classes_ids=str(class_id),
        #     select_best_n=select_best_n,
        #     out_dir_all="network_inversion/inversion_images_logs_diff_batch_size_per_class/",
        #     run_mode="run"
        # )

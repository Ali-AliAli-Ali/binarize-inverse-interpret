import os
import sys
from typing import Literal

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import MODEL_CONFIGS                    # noqa: E402
from invert_batch_imagenet_val_as_func import main_pipeline    # noqa: E402


def train_inversion(network_names: list[str],
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
                    select_best_n: float | None = 10,
                    n_rows_in_output: int | None = 2,
                    out_dir: str | None = None,
                    run_mode: Literal["debug", "run"] | None = "run"):
    for network_name in network_names:
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
            "nrow":          n_rows_in_output,
            "out_dir":       out_dir or f"network_inversion/"
                                        f"inversion_images_logs/"
                                        f"inversion_{network_name}_{dataset_name}/"
                                        f"steps_{n_training_steps}_classes_{classes_ids}",
            "run_mode":      run_mode
        })

    
if __name__ == '__main__':
    
    network_names = enumerate(["regnet_x_32"]*2) # MODEL_CONFIGS.keys()
    dataset_dir_imagenet = "/media/user/Hitachi/ILSVRC/Data/CLS-LOC"
    dataset_name = "ImageNet"
    n_training_steps = 10_000
    classes_ids = "100, 999"

    train_inversion(
        network_names, 
        dataset_dir_imagenet, 
        dataset_name, 
        n_training_steps=n_training_steps,
        classes_ids=classes_ids,
        run_mode="debug"
    )
    
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath( os.path.join(script_dir, '..', 'common_key_functions') )
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from constants_configs import MODEL_CONFIGS                    # noqa: E402
from invert_batch_imagenet_val_as_func import main_pipeline    # noqa: E402

dataset_dir_imagenet = "/media/user/Hitachi/ILSVRC/Data/CLS-LOC"
dataset_name = "ImageNet"
n_training_steps = 10_000   # 4000 by default 

network_names = MODEL_CONFIGS.keys()
networks_done = set(["regnet_x_3_2"])

for network_name in network_names:
    if network_name not in networks_done:
        args = {
            "data_dir":      dataset_dir_imagenet,
            "class_ids":      "1, 10, 100, 999",
            "network":       network_name,
            "batch_size":    MODEL_CONFIGS[network_name]["batch_size"],
            "num_workers":   8,
            "steps":         n_training_steps,
            "lr":            0.1,
            "sigma":         0.01,
            "beta":          4.0,
            "tv_weight":     5e-5,
            "l2_weight":     10.0,
            "l1_weight":     0.0,
            "subset_size":   5000,
            "threshold":     None,
            "select_best_n": 10,
            "nrow":          2,
            "out_dir":       f"network_inversion/inversion_{network_name}_{dataset_name}/steps_{n_training_steps}"
        }

        main_pipeline(args)

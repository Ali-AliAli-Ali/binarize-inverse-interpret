import os
import re
from PIL import Image


def process_image(filepath, 
                  n_cut: int = 2,
                  skip_batch_min: int | None = None):
    """
    Processes image: if batch_size > skip_batch_min (if set), crops it to the first `n_cut` rows.
    Replaces the original file.
    """
    basename = os.path.basename(filepath)
    match = re.match(r'bs_(\d+)_\d+\.png$', basename)
    if not match:
        print(f"Skip: filename does not match pattern 'bs_<batch_size>_<class>.png' - {basename}")
        return
    batch_size = int(match.group(1))
    if (skip_batch_min is not None) and (batch_size <= skip_batch_min):
        print(f"Skip: batch_size = {batch_size} (not > {skip_batch_min}) - {basename}")
        return

    try:
        with Image.open(filepath) as img:
            width, height = img.size
            # row_height = 452 # height // min(batch_size, 10)
            # new_height = 2 * row_height
            # if new_height > height:
            #     new_height = height
            height_crop = min(226 * n_cut, height)
            cropped = img.crop((0, 0, width, height_crop))
            cropped.save(filepath)
            print(f"Processed: {basename} (batch_size={batch_size}) -> cropped to {height_crop}px height")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


root_dir = "/home/user/Downloads/"
grids_dirs = [
    "resnet18", "resnet50", "vit_b_16"
]
pattern = re.compile(r'bs_\d+_\d+\.png$')

for grids_dir in grids_dirs:
    print("\nProcessing", grids_dir, "\n")
    for dirpath, _, filenames in os.walk( os.path.join(root_dir, grids_dir)):
        for fname in filenames:
            if pattern.match(fname):
                full_path = os.path.join(dirpath, fname)
                process_image(full_path, 1)
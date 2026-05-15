#!/bin/bash
#
# Copy best_orig_vs_recon_<class_id>.png from the nested logs directory
# into a flat validation directory, renaming them as
# bs_<batch_size>_<class_id>.png
#

SRC_BASE="inversion_images_logs_diff_batch_size_per_class"
DST_BASE="inversion_images_val_diff_batch_size_per_class"

# Create destination base if it doesn't exist
mkdir -p "$DST_BASE"

for net_dir in "$SRC_BASE"/inversion_*_ImageNet; do
    # Skip if no directories match (in case the pattern fails)
    [ -d "$net_dir" ] || continue

    # Extract network name: remove prefix "inversion_" and suffix "_ImageNet"
    dir_name=$(basename "$net_dir")
    network_name=${dir_name#inversion_}         # remove leading "inversion_"
    network_name=${network_name%_ImageNet}      # remove trailing "_ImageNet"

    echo "Processing network: $network_name"

    for class_dir in "$net_dir"/bs_*_classes_*; do
        [ -d "$class_dir" ] || continue

        class_dir_name=$(basename "$class_dir")   # e.g. bs_8_classes_0001

        # Parse batch_size and class_id from the directory name
        # Format: bs_<batch_size>_classes_<class_id>
        tmp=${class_dir_name#bs_}                 # remove leading "bs_"
        batch_size=${tmp%%_classes_*}             # everything before "_classes_"
        class_id=${tmp#*_classes_}                # everything after "_classes_"

        src_file="$class_dir/best_orig_vs_recon_$(printf "%04d" $class_id).png"

        if [ -f "$src_file" ]; then
            dst_dir="$DST_BASE/$network_name"
            mkdir -p "$dst_dir"
            dst_file="$dst_dir/bs_${batch_size}_${class_id}.png"
            cp "$src_file" "$dst_file"
            echo "  Copied $src_file -> $dst_file"
        else
            echo "  [WARNING] Missing file: $src_file"
        fi
    done
done

echo "Done."
"""
General-Purpose Volumetric Preprocessing Script-

This script applies a series of preprocessing steps to all NIfTI files
found within a specified input directory and saves the results to an output
directory. It is designed to be run as a standalone tool from the command line,
allowing it to be flexibly integrated into various research pipelines.

Core operations include robust Z-score intensity normalization based on
foreground percentiles and 2D slice resizing via centered padding or cropping.
"""

import os
import glob
import time
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm

def robust_zscore(
    img: np.ndarray,
    mask: np.ndarray,
    lower_percentile: int = 2,
    upper_percentile: int = 98
) -> np.ndarray:
    
    if mask.sum() == 0:
        return img

    foreground_vals = img[mask]
    lower_bound, upper_bound = np.percentile(
        foreground_vals, [lower_percentile, upper_percentile]
    )
    
    clipped_img = np.clip(img, lower_bound, upper_bound)
    clipped_foreground_vals = clipped_img[mask]

    mean = clipped_foreground_vals.mean()
    std = clipped_foreground_vals.std()
    
    if std < 1e-8:
        return np.zeros_like(clipped_img, dtype=np.float32)

    normalized_img = (clipped_img - mean) / std
    normalized_img[~mask] = normalized_img[mask].min()
    return normalized_img

def resize_volume(
    volume: np.ndarray,
    target_shape: tuple = (208, 208)
) -> np.ndarray:
    
    original_h, original_w, num_slices = volume.shape
    target_h, target_w = target_shape

    delta_h = target_h - original_h
    pad_top = max(0, delta_h // 2)
    pad_bottom = max(0, delta_h - pad_top)
    crop_top = max(0, -delta_h // 2)
    crop_bottom = max(0, -delta_h - crop_top)

    delta_w = target_w - original_w
    pad_left = max(0, delta_w // 2)
    pad_right = max(0, delta_w - pad_left)
    crop_left = max(0, -delta_w // 2)
    crop_right = max(0, -delta_w - crop_left)

    cropped_volume = volume[
        crop_top:original_h - crop_bottom,
        crop_left:original_w - crop_right,
        :
    ]

    padded_volume = np.pad(
        cropped_volume,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        'constant',
        constant_values=0
    )
    return padded_volume

def process_file(file_path: str, args: argparse.Namespace):
    """
    Loads, processes, and saves a single NIfTI file.
    """
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata().astype(np.float32)
        
        if args.no_norm:
            normalized_data = data
        else:
            mask_data = None
            if args.norm_mask_dir:
                
                mask_path = os.path.join(args.norm_mask_dir, os.path.basename(file_path))
                if os.path.exists(mask_path):
                    mask_data = nib.load(mask_path).get_fdata().astype(bool)
                else:
                    print(f"  [Warning] Mask not found for {os.path.basename(file_path)}. Using auto-mask.")
            
            if mask_data is None:
                # If no mask provided or found, generate a simple one
                mask_data = data > (data.min() + 1e-8)
            
            normalized_data = robust_zscore(data, mask_data, args.percentiles[0], args.percentiles[1])

     
        resized_data = resize_volume(normalized_data, tuple(args.target_size))

        out_nii = nib.Nifti1Image(resized_data, nii.affine)
        output_path = os.path.join(args.output_dir, os.path.basename(file_path))
        nib.save(out_nii, output_path)

    except Exception as e:
        print(f"\n[ERROR] Failed to process file {os.path.basename(file_path)}.")
        print(f"  Reason: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="General-Purpose Preprocessing for Volumetric Medical Images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to the directory containing raw NIfTI files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the directory where processed files will be saved."
    )
    parser.add_argument(
        "--target_size", nargs=2, type=int, default=[208, 208],
        help="Target 2D slice dimensions (Height Width)."
    )
    parser.add_argument(
        "--no_norm", action='store_true',
        help="Disable intensity normalization. Use this for processing mask files."
    )
    parser.add_argument(
        "--norm_mask_dir", type=str, default=None,
        help="Optional. Path to a directory with brain masks for normalization."
    )
    parser.add_argument(
        "--percentiles", nargs=2, type=int, default=[2, 98],
        help="Lower and upper percentiles for robust Z-score normalization."
    )
    args = parser.parse_args()
    
    print("\nStarting Preprocessing...")
    print(f"  Input Directory:  {args.input_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print("-" * 30)

    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    file_paths = glob.glob(os.path.join(args.input_dir, '*.nii*'))
    if not file_paths:
        print(f"[ERROR] No NIfTI files ('*.nii' or '*.nii.gz') found in '{args.input_dir}'.")
        return
        
    for file_path in tqdm(file_paths, desc="Processing files"):
        process_file(file_path, args)
            
    end_time = time.time()
    print(f"\nPreprocessing Complete. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

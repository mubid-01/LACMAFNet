import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd

# This script assumes train_utils.py and dataset_wmh.py are in the same directory
from src.utils.utils import post_process_volume
from src.data_loaders.dataset_wmh import find_mask_dir
from medpy import metric

def tune_hyperparameters():
    class Config:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        default_work = os.path.join(repo_root, 'work')
        default_data = os.path.join(repo_root, 'data', 'wmh_split_data')

        base_work_dir = os.getenv('WORK_DIR', os.getenv('BASE_WORK_DIR', default_work))
        base_data_dir = os.getenv('DATA_ROOT', os.getenv('BASE_DATA_DIR', default_data))
        pred_dir = os.getenv('PRED_DIR', os.path.join(base_work_dir, "validation_probabilities"))
        gt_root = os.getenv('GT_ROOT', os.path.join(base_data_dir, "train", "validation"))

    cfg = Config()

    thresholds_to_test = np.linspace(0.10, 0.80, 29)
    min_sizes_to_test = [2, 3, 4, 5, 6, 8, 10, 12, 15]

    pred_files = sorted(glob.glob(os.path.join(cfg.pred_dir, "*_prob.nii.gz")))
    
    gt_mask_dir = find_mask_dir(cfg.gt_root)
    if gt_mask_dir is None:
        raise FileNotFoundError(f"Could not find a ground truth/mask directory inside: {cfg.gt_root}")
    print(f"Using Validation Ground Truth masks from: {gt_mask_dir}")
    
    results = []

    print(f"Starting grid search on {len(pred_files)} validation cases...")
    for min_size in tqdm(min_sizes_to_test, desc="Min Lesion Size"):
        for threshold in thresholds_to_test:
            all_case_dices = []
            for pred_path in pred_files:
                base_name = os.path.basename(pred_path).replace('_prob.nii.gz', '')
                gt_path = os.path.join(gt_mask_dir, f"{base_name}_wmh.nii.gz")
                if not os.path.exists(gt_path): gt_path = gt_path[:-3]
                
                if not os.path.exists(gt_path): continue
                    
                pred_prob_vol = nib.load(pred_path).get_fdata()
                gt_vol = nib.load(gt_path).get_fdata()
                gt_binary = (gt_vol > 0.5).astype(np.uint8)

                pred_binary = (pred_prob_vol > threshold).astype(np.uint8)
                pred_processed = post_process_volume(pred_binary, min_lesion_size=min_size)
                
                if gt_binary.sum() > 0 and pred_processed.sum() > 0:
                    dice = metric.binary.dc(pred_processed, gt_binary)
                elif gt_binary.sum() == 0 and pred_processed.sum() == 0:
                    dice = 1.0
                else:
                    dice = 0.0
                all_case_dices.append(dice)

            avg_dice = np.mean(all_case_dices)
            results.append({'min_lesion_size': min_size, 'threshold': round(threshold, 3), 'avg_dice': round(avg_dice, 5)})

    print("\nGrid Search Complete.")
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['avg_dice'].idxmax()]
    
    print("\n" + "="*40)
    print("---           OPTIMAL PARAMETERS (from Validation Set)           ---")
    print("="*40)
    print(f"Best Threshold       : {best_result['threshold']}")
    print(f"Best Min Lesion Size : {best_result['min_lesion_size']}")
    print(f"Resulting Avg Dice   : {best_result['avg_dice']:.4f}")
    print("="*40 + "\n")
    
    print("\n--- Full Results Table ---")
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 50)
    print(results_df.pivot(index='min_lesion_size', columns='threshold', values='avg_dice'))
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', help='Directory with validation probability maps')
    parser.add_argument('--gt_root', help='Validation ground truth root')
    parser.add_argument('--data_root', help='Base data root')
    args = parser.parse_args()
    if args.pred_dir: os.environ['PRED_DIR'] = args.pred_dir
    if args.gt_root: os.environ['GT_ROOT'] = args.gt_root
    if args.data_root: os.environ['DATA_ROOT'] = args.data_root
    tune_hyperparameters()
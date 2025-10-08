import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from medpy import metric

from src.utils.utils import post_process_volume


def tune_hyperparameters(pred_dir=None, gt_root=None, thresholds=None, min_sizes=None):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_work = os.path.join(repo_root, 'work')
    default_data = os.path.join(repo_root, 'data', 'isles')

    cfg = type('C', (), {})()
    base_work_dir = os.environ.get('WORK_DIR', os.environ.get('BASE_WORK_DIR', default_work))
    data_root = os.environ.get('DATA_ROOT', default_data)
    cfg.pred_dir = pred_dir or os.environ.get('PRED_DIR', os.path.join(base_work_dir, "validation_probabilities"))
    cfg.gt_root = gt_root or os.environ.get('GT_ROOT', os.path.join(data_root, "derivatives"))

    thresholds_to_test = thresholds if thresholds is not None else np.linspace(0.10, 0.80, 29)
    min_sizes_to_test = min_sizes if min_sizes is not None else [2, 3, 4, 5, 6, 8, 10, 15, 20]
    pred_files = sorted(glob.glob(os.path.join(cfg.pred_dir, "*_prob.nii")) + glob.glob(os.path.join(cfg.pred_dir, "*_prob.nii.gz")))
    if not pred_files:
        raise FileNotFoundError(f"No probability maps found in {cfg.pred_dir}.")

    results = []
    print(f"Starting grid search on {len(pred_files)} validation cases...")

    for min_size in tqdm(min_sizes_to_test, desc="Min Lesion Size"):
        for threshold in thresholds_to_test:
            all_case_dices = []
            for pred_path in pred_files:
                base_name = os.path.basename(pred_path).replace('_prob.nii', '')
                try:
                    gt_path_list = glob.glob(os.path.join(cfg.gt_root, base_name, 'ses-0001', '*_msk.nii'))
                    if not gt_path_list: continue
                    gt_path = gt_path_list[0]
                    
                    pred_prob_vol = nib.load(pred_path).get_fdata()
                    gt_binary = (nib.load(gt_path).get_fdata() > 0.5).astype(np.uint8)
                    pred_binary = (pred_prob_vol > threshold).astype(np.uint8)
                    pred_processed = post_process_volume(pred_binary, min_lesion_size=min_size)
                    
                    dice = metric.binary.dc(pred_processed, gt_binary) if (gt_binary.sum() > 0 or pred_processed.sum() > 0) else 1.0
                    all_case_dices.append(dice)
                except Exception:
                    continue

            if all_case_dices:
                avg_dice = np.mean(all_case_dices)
                results.append({'min_lesion_size': min_size, 'threshold': round(threshold, 3), 'avg_dice': round(avg_dice, 5)})

    if not results:
        print("FATAL: No results generated.")
        return
        
    print("\nGrid Search Complete.")
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['avg_dice'].idxmax()]
    
    print("\n" + "="*60)
    print("--- OPTIMAL PARAMETERS (from Validation Set) ---")
    print("="*60)
    print(f"Best Threshold       : {best_result['threshold']}")
    print(f"Best Min Lesion Size : {best_result['min_lesion_size']}")
    print(f"Resulting Avg Dice   : {best_result['avg_dice']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune ISLES post-processing hyperparameters')
    parser.add_argument('--pred_dir', type=str, help='Directory containing probability maps')
    parser.add_argument('--gt_root', type=str, help='Ground-truth derivatives root')
    args = parser.parse_args()

    tune_hyperparameters(pred_dir=args.pred_dir, gt_root=args.gt_root)
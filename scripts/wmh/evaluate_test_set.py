import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from medpy import metric
from scipy.ndimage import label as cc_label

from src.utils.helpers.wmh_helpers import post_process_volume, calculate_lesion_metrics

def evaluate_test_set_performance():
    class Config:
        OPTIMAL_THRESHOLD = float(os.getenv('OPTIMAL_THRESHOLD', 0.1))
        OPTIMAL_MIN_LESION_SIZE = int(os.getenv('OPTIMAL_MIN_LESION_SIZE', 2))
        
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_work = os.path.join(repo_root, 'work')
    default_data = os.path.join(repo_root, 'data', 'wmh_split_data')

    base_work_dir = os.getenv('WORK_DIR', os.getenv('BASE_WORK_DIR', default_work))
    base_data_dir = os.getenv('DATA_ROOT', os.getenv('BASE_DATA_DIR', default_data))
    prediction_dir = os.getenv('PREDICTION_DIR', os.path.join(base_work_dir, "final_test_segmentations"))
    test_gt_root = os.getenv('TEST_GT_ROOT', os.path.join(base_data_dir, "test", "train"))
    results_dir = os.getenv('RESULTS_DIR', os.path.join(base_work_dir, "final_report_and_plots"))

    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(cfg.prediction_dir, "*_wmh_pred.nii.gz")))
    from src.data_loaders.dataset_wmh import find_mask_dir
    gt_mask_dir = find_mask_dir(cfg.test_gt_root)
    
    if not pred_files: raise FileNotFoundError(f"No prediction files found in: {cfg.prediction_dir}.")
    if gt_mask_dir is None: raise FileNotFoundError(f"Could not find a ground truth/mask directory inside: {cfg.test_gt_root}")

    all_results = []
    for pred_path in tqdm(pred_files, desc="Evaluating Test Set"):
        base_name = os.path.basename(pred_path).replace('_wmh_pred.nii.gz', '')
        gt_path = os.path.join(gt_mask_dir, f"{base_name}_wmh.nii.gz")
        if not os.path.exists(gt_path): gt_path = gt_path[:-3]
        if not os.path.exists(gt_path): continue
            
        pred_processed = nib.load(pred_path).get_fdata().astype(np.uint8)
        gt_binary = (nib.load(gt_path).get_fdata() > 0.5).astype(np.uint8)

        dice = metric.binary.dc(pred_processed, gt_binary) if (gt_binary.sum()>0 or pred_processed.sum()>0) else 1.0
        hd95 = metric.binary.hd95(pred_processed, gt_binary) if (gt_binary.sum()>0 and pred_processed.sum()>0) else 0.0
        
        Vml = float(gt_binary.sum())
        Val = float(pred_processed.sum())
        if Vml > 0:
            avd = abs(Vml - Val) / Vml * 100.0
        else:
            avd = 0.0 if Val == 0 else 500.0 
        
        f1, recall = calculate_lesion_metrics(pred_processed, gt_binary)
        
        all_results.append({'Subject': base_name, 'DSC': dice, 'HD95 (mm)': hd95,
                            'F1-score': f1, 'recall': recall, 'AVD (%)': avd,
                            'GT_Volume': Vml})

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(cfg.results_dir, "final_test_set_metrics_paper_format.csv")
    results_df.to_csv(csv_path, index=False)
    
    summary = results_df[['DSC', 'HD95 (mm)', 'AVD (%)', 'recall', 'F1-score']].agg(['mean', 'std'])
    
    print("\n" + "="*70); print("---      FINAL TEST SET PERFORMANCE (Mean +/- SD)      ---"); print("="*70)
    print(summary.round(4)); print("="*70)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='Base data directory (overrides config)')
    parser.add_argument('--work_dir', help='Working directory (overrides config)')
    parser.add_argument('--prediction_dir', help='Directory with prediction files')
    parser.add_argument('--results_dir', help='Directory to write results')
    parser.add_argument('--test_gt_root', help='Root folder containing test ground truth masks')
    args = parser.parse_args()
    if args.data_root: os.environ['DATA_ROOT'] = args.data_root
    if args.work_dir: os.environ['WORK_DIR'] = args.work_dir
    if args.prediction_dir: os.environ['PREDICTION_DIR'] = args.prediction_dir
    if args.results_dir: os.environ['RESULTS_DIR'] = args.results_dir
    if args.test_gt_root: os.environ['TEST_GT_ROOT'] = args.test_gt_root
    evaluate_test_set_performance()
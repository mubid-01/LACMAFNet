import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from medpy import metric
from scipy.ndimage import label as cc_label
from src.utils.utils import post_process_volume, calculate_lesion_metrics

def evaluate_test_set_performance():
    class Config:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        default_work = os.path.join(repo_root, 'work')
        default_data = os.path.join(repo_root, 'data', 'isles')

        base_work_dir = os.getenv('WORK_DIR', os.getenv('BASE_WORK_DIR', default_work))
        data_root = os.getenv('DATA_ROOT', os.getenv('BASE_DATA_DIR', default_data))
        prediction_dir = os.getenv('PREDICTION_DIR', os.path.join(base_work_dir, "final_test_segmentations"))
        test_gt_root = os.getenv('TEST_GT_ROOT', os.path.join(data_root, "derivatives"))
        results_dir = os.getenv('RESULTS_DIR', os.path.join(base_work_dir, "final_report_and_plots"))

    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(cfg.prediction_dir, "*_pred.nii")))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in: {cfg.prediction_dir}. Please run test_inference_final.py first.")

    all_results = []
    for pred_path in tqdm(pred_files, desc="Evaluating Final Test Set"):
        base_name = os.path.basename(pred_path).replace('_pred.nii', '')
        try:
            gt_path_list = glob.glob(os.path.join(cfg.test_gt_root, base_name, 'ses-0001', f'{base_name}_ses-0001_msk.nii'))
            
            if not gt_path_list:
                print(f"Warning: No ground truth found for {base_name}. Skipping.")
                continue
            gt_path = gt_path_list[0]
            
            pred_processed = nib.load(pred_path).get_fdata().astype(np.uint8)
            gt_binary = (nib.load(gt_path).get_fdata() > 0.5).astype(np.uint8)

            dice = metric.binary.dc(pred_processed, gt_binary) if (gt_binary.sum() > 0 or pred_processed.sum() > 0) else 1.0
            hd95 = metric.binary.hd95(pred_processed, gt_binary) if (gt_binary.sum() > 0 and pred_processed.sum() > 0) else 0.0
            gt_sum, pred_sum = float(gt_binary.sum()), float(pred_processed.sum())
            avd = abs(gt_sum - pred_sum) / gt_sum * 100.0 if gt_sum > 0 else 0.0
            f1, recall = calculate_lesion_metrics(pred_processed, gt_binary)
            
            all_results.append({'Subject': base_name, 'DSC': dice, 'HD95 (mm)': hd95,
                                'F1-score': f1, 'recall': recall, 'AVD (%)': avd})
        except Exception as e:
            print(f"Could not process subject {base_name}. Error: {e}")
            continue

    if not all_results:
        print("\nFATAL ERROR: No subjects were successfully processed. The 'all_results' list is empty.")
        print("Please check the file paths and glob patterns for finding ground truth files.")
        return

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(cfg.results_dir, "final_test_set_metrics_isles22.csv")
    results_df.to_csv(csv_path, index=False)
    
    summary = results_df[['DSC', 'HD95 (mm)', 'AVD (%)', 'recall', 'F1-score']].agg(['mean', 'std'])
    
    print("\n" + "="*70)
    print("---      FINAL TEST SET PERFORMANCE (Mean +/- SD)      ---")
    print("="*70)
    print(summary.round(4))
    print("="*70)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='ISLES data root')
    parser.add_argument('--prediction_dir', help='Directory with predictions')
    parser.add_argument('--results_dir', help='Directory for results CSV')
    parser.add_argument('--test_gt_root', help='Root folder containing test ground truth masks')
    args = parser.parse_args()
    if args.data_root: os.environ['DATA_ROOT'] = args.data_root
    if args.prediction_dir: os.environ['PREDICTION_DIR'] = args.prediction_dir
    if args.results_dir: os.environ['RESULTS_DIR'] = args.results_dir
    if args.test_gt_root: os.environ['TEST_GT_ROOT'] = args.test_gt_root
    evaluate_test_set_performance()
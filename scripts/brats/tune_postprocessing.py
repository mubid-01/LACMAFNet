import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

from src.data_loaders.dataset_brats import BraTS_dataset
from src.utils.helpers.brats_helpers import post_process_volume, compute_brats_metrics_detailed

def tune_postprocessing(pred_dir, val_ds_root, min_sizes=(10,25,50,100), device='cpu'):
    ds = BraTS_dataset(base_dir=val_ds_root, split='val', transform=None)
    results = []
    for min_size in min_sizes:
        metrics_accum = []
        for i in tqdm(range(len(ds)), desc=f'Tuning min_size={min_size}'):
            vols = ds._load_volume(i); mask_vol = vols['mask']; mask_vol[mask_vol==4]=3
            subj_name = ds.subject_files[i]['name'] if 'subject_files' in dir(ds) else f'subj_{i}'
            pred_path = os.path.join(pred_dir, f"{subj_name}_pred.nii")
            if not os.path.exists(pred_path):
                continue
            pred = nib.load(pred_path).get_fdata().astype(np.uint8)
            for cls in [1,2,3]: pred = post_process_volume(pred, class_idx=cls, min_size=min_size)
            metrics_accum.append(compute_brats_metrics_detailed(pred, mask_vol))
        if metrics_accum:
            import pandas as pd
            df = pd.DataFrame(metrics_accum)
            results.append({'min_size': min_size, **df.mean().to_dict()})
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--val_root', required=True)
    parser.add_argument('--out', default='post_tuning_results.csv')
    args = parser.parse_args()
    res = tune_postprocessing(args.pred_dir, args.val_root)
    import pandas as pd
    pd.DataFrame(res).to_csv(args.out, index=False)
    print('Saved tuning results to', args.out)

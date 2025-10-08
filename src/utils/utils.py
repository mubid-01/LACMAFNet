import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import label as cc_label
from scipy.ndimage import distance_transform_edt

def post_process_volume(pred_vol_binary, min_lesion_size=3):
    labeled_vol, num_components = cc_label(pred_vol_binary, structure=np.ones((3, 3, 3)))
    if num_components == 0:
        return pred_vol_binary
    component_sizes = np.bincount(labeled_vol.ravel())
    too_small_labels = np.where(component_sizes[1:] < min_lesion_size)[0] + 1
    cleaned_mask = ~np.isin(labeled_vol, too_small_labels)
    return pred_vol_binary & cleaned_mask

def counts_from_logits(logits, targets, thr=0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > thr).to(targets.dtype)
        p, t = preds.view(preds.size(0), -1), targets.view(targets.size(0), -1)
        tp = (p * t).sum(dim=1); fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1); tn = ((1 - p) * (1 - t)).sum(dim=1)
        return float(tp.sum().item()), float(fp.sum().item()), float(fn.sum().item()), float(tn.sum().item())

def compute_case_metrics(pred_vol, gt_vol, min_lesion_size=3):
    gt = (gt_vol > 0.5).astype(np.uint8)
    max_dist_penalty = np.sqrt(sum([dim ** 2 for dim in gt.shape]))
    best_dice, best_thr = -1.0, 0.5
    for thr in np.linspace(0.05, 0.95, 19):
        pred_binary_thr = (pred_vol > thr).astype(np.uint8)
        pred_processed_thr = post_process_volume(pred_binary_thr, min_lesion_size=min_lesion_size)
        if gt.sum() == 0 and pred_processed_thr.sum() == 0: current_dice = 1.0
        elif gt.sum() > 0 and pred_processed_thr.sum() > 0: current_dice = metric.binary.dc(pred_processed_thr, gt)
        else: current_dice = 0.0
        if current_dice > best_dice: best_dice, best_thr = current_dice, thr
    
    pred_binary = (pred_vol > best_thr).astype(np.uint8)
    pred_processed = post_process_volume(pred_binary, min_lesion_size=min_lesion_size)
    
    if gt.sum() > 0 and pred_processed.sum() > 0:
        try: hd95 = metric.binary.hd95(pred_processed, gt)
        except RuntimeError: hd95 = max_dist_penalty
    else: hd95 = max_dist_penalty
    
    Vml, Val = float(gt.sum()), float(pred_processed.sum())
    avd_percent = min(abs(Vml - Val) / Vml * 100.0 if Vml > 0 else (0.0 if Val == 0 else 500.0), 500.0)
    
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_lbl, n_gt = cc_label(gt, structure=structure); pr_lbl, n_pr = cc_label(pred_processed, structure=structure)
    
    if n_gt > 0:
        overlap = (gt_lbl > 0) & (pr_lbl > 0)
        matched_gt = np.unique(gt_lbl[overlap]); matched_gt = matched_gt[matched_gt > 0]
        N_TP = int(len(matched_gt)); N_FN = int(n_gt - N_TP)
    else: N_TP, N_FN = 0, 0
    
    if n_pr > 0:
        overlap = (gt_lbl > 0) & (pr_lbl > 0)
        matched_pr = np.unique(pr_lbl[overlap]); matched_pr = matched_pr[matched_pr > 0]
        N_FP = int(n_pr - len(matched_pr))
    else: N_FP = 0
        
    recall_lesion = (N_TP / (N_TP + N_FN)) if (N_TP + N_FN) > 0 else (1.0 if n_gt == 0 else 0.0)
    denom = (2 * N_TP + N_FN + N_FP)
    f1_lesion = (2 * N_TP / denom) if denom > 0 else (1.0 if (n_gt == 0 and n_pr == 0) else 0.0)
    acc = np.mean(pred_processed == gt)
    return best_dice, hd95, recall_lesion, acc, avd_percent, f1_lesion

def downsample_mask(mask_tensor, logits_tensor):
    return F.interpolate(mask_tensor, size=logits_tensor.shape[2:], mode='nearest')

def calculate_lesion_metrics(pred_binary, gt_binary):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_lbl, n_gt = cc_label(gt_binary, structure=structure)
    pr_lbl, n_pr = cc_label(pred_binary, structure=structure)
    
    if n_gt > 0:
        overlap = (gt_lbl > 0) & (pr_lbl > 0)
        matched_gt = np.unique(gt_lbl[overlap]); matched_gt = matched_gt[matched_gt > 0]
        N_TP = int(len(matched_gt))
    else: N_TP = 0

    N_FN = int(n_gt - N_TP)

    if n_pr > 0:
        overlap = (gt_lbl > 0) & (pr_lbl > 0)
        matched_pr = np.unique(pr_lbl[overlap]); matched_pr = matched_pr[matched_pr > 0]
        N_FP = int(n_pr - len(matched_pr))
    else: N_FP = 0

    recall_lesion = (N_TP / (N_TP + N_FN)) if (N_TP + N_FN) > 0 else (1.0 if n_gt == 0 else 0.0)
    f1_lesion = (2 * N_TP) / (2 * N_TP + N_FP + N_FN) if (2 * N_TP + N_FP + N_FN) > 0 else (1.0 if (n_gt == 0 and n_pr == 0) else 0.0)
    
    return f1_lesion, recall_lesion
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import label as cc_label

def post_process_volume(pred_vol, class_idx, min_size=10):
    binary_mask = (pred_vol == class_idx)
    if not binary_mask.any():
        return pred_vol
    
    labeled_vol, num_components = cc_label(binary_mask, structure=np.ones((3, 3, 3)))
    if num_components == 0:
        return pred_vol
    
    component_sizes = np.bincount(labeled_vol.ravel())
    too_small_labels = np.where(component_sizes[1:] < min_size)[0] + 1
    removal_mask = np.isin(labeled_vol, too_small_labels)
    
    cleaned_pred_vol = pred_vol.copy()
    cleaned_pred_vol[removal_mask] = 0
    return cleaned_pred_vol

class DiceLoss(nn.Module):
    def __init__(self, to_onehot_y=True, softmax=True, squared_pred=True,
                 smooth_nr=0.0, smooth_dr=1e-5):
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

    def forward(self, logits, targets):
        if self.softmax:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits
        if self.to_onehot_y:
            y_onehot = torch.zeros_like(probs)
            y_onehot.scatter_(1, targets.unsqueeze(1), 1)
        else:
            y_onehot = targets
        intersection = torch.sum(probs * y_onehot, dim=[0, 2, 3])
        if self.squared_pred:
            ground_o = torch.sum(y_onehot**2, dim=[0, 2, 3])
            pred_o = torch.sum(probs**2, dim=[0, 2, 3])
        else:
            ground_o = torch.sum(y_onehot, dim=[0, 2, 3])
            pred_o = torch.sum(probs, dim=[0, 2, 3])
        cardinality = ground_o + pred_o
        dice_score = (2.0 * intersection + self.smooth_nr) / (cardinality + self.smooth_dr)
        dice_loss = (1.0 - dice_score).mean()
        return dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        mod_factor = (1.0 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
        loss = alpha_factor * mod_factor * bce
        return loss.mean() if self.reduction == 'mean' else loss


def compute_brats_metrics_detailed(pred_vol, gt_vol):
    max_dist_penalty = np.sqrt(sum([dim ** 2 for dim in gt_vol.shape]))
    
    def dice(p, g):
        return 1.0 if (np.sum(p) + np.sum(g)) == 0 else (2. * np.sum(p * g)) / (np.sum(p) + np.sum(g) + 1e-6)
    
    def hd95(p, g):
        if not p.any() and not g.any(): return 0.0
        if not p.any() or not g.any(): return max_dist_penalty
        try: return metric.binary.hd95(p, g)
        except RuntimeError: return max_dist_penalty
        
    pred_net = (pred_vol == 1); gt_net = (gt_vol == 1)
    pred_ed = (pred_vol == 2); gt_ed = (gt_vol == 2)
    pred_et = (pred_vol == 3); gt_et = (gt_vol == 3)

    pred_wt = pred_net | pred_ed | pred_et; gt_wt = gt_net | gt_ed | gt_et
    pred_tc = pred_net | pred_et; gt_tc = gt_net | gt_et

    return {
        'Dice_WT': dice(pred_wt, gt_wt), 'HD95_WT': hd95(pred_wt, gt_wt),
        'Dice_TC': dice(pred_tc, gt_tc), 'HD95_TC': hd95(pred_tc, gt_tc),
        'Dice_ET': dice(pred_et, gt_et), 'HD95_ET': hd95(pred_et, gt_et),
    }

def downsample_mask(mask_tensor, logits_tensor):
    if mask_tensor.ndim == 4:
        return F.interpolate(mask_tensor, size=logits_tensor.shape[2:], mode='nearest')
    elif mask_tensor.ndim == 3:
        return F.interpolate(mask_tensor.unsqueeze(1).float(), size=logits_tensor.shape[2:],
                           mode='nearest').squeeze(1).long()
    else:
        raise ValueError(f"Unsupported mask tensor ndim: {mask_tensor.ndim}")
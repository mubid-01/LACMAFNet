import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import label as cc_label
from scipy.ndimage import distance_transform_edt

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        mod_factor = (1.0 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
        loss = alpha_factor * mod_factor * bce
        return loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    def forward(self, pred_probs, target):
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        tgt_flat = target.view(target.size(0), -1)
        tp = (pred_flat * tgt_flat).sum(1)
        fp = (pred_flat * (1 - tgt_flat)).sum(1)
        fn = ((1 - pred_flat) * tgt_flat).sum(1)
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, wf=0.5, wt=0.5, alpha=0.25, gamma=2.0, tversky_beta=0.7):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        tversky_alpha = 1.0 - tversky_beta
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.wf, self.wt = wf, wt
    def forward(self, logits, target):
        focal_loss = self.focal(logits, target)
        probs = torch.sigmoid(logits)
        tversky_loss = self.tversky(probs, target)
        return self.wf * focal_loss + self.wt * tversky_loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, probs, gt):
        gt_numpy = gt.cpu().numpy().astype(np.uint8)
        dist_maps = []
        for i in range(gt_numpy.shape[0]):
            dist = distance_transform_edt(1 - gt_numpy[i, 0])
            dist_maps.append(dist)
        dist_maps = torch.from_numpy(np.array(dist_maps)).float().to(probs.device)
        loss = (probs[:, 0, :, :] * dist_maps).mean()
        return loss
    

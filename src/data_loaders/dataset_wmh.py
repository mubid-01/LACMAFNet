import os
import glob
import random
from functools import lru_cache

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
from scipy.ndimage import map_coordinates, gaussian_filter

def resolve_split_root(base_dir, split):
    target = 'training' if split == 'train' else 'validation'
    candidates = [
        os.path.join(base_dir, target),
        os.path.join(base_dir, 'train', target),
        os.path.join(base_dir, 'training', target),
        os.path.join(base_dir, 'train', 'training') if split == 'train' else os.path.join(base_dir, 'train', 'validation'),
        os.path.join(base_dir, 'validation') if split == 'val' else os.path.join(base_dir, 'training')
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'flair')):
            return c
    search = glob.glob(os.path.join(base_dir, '**', 'flair'), recursive=True)
    if search:
        for s in search:
            if split == 'train' and ('train' in s or 'training' in s):
                return os.path.dirname(s)
            if split == 'val' and ('val' in s or 'validation' in s):
                return os.path.dirname(s)
        return os.path.dirname(search[0])
    return None

def find_mask_dir(root):
    candidates = ['Ground Truth', 'Ground_Truth', 'ground_truth', 'GroundTruth', 'masks', 'mask']
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    for sub in os.listdir(root):
        sp = os.path.join(root, sub)
        if os.path.isdir(sp):
            if glob.glob(os.path.join(sp, '*_wmh.nii*')):
                return sp
    return None

class RandomGeneratorFromScratch:
    def __init__(self, output_size=(208, 208)):
        self.output_size = tuple(output_size)

    def _apply_elastic_transform(self, image, label, alpha=100, sigma=12):
        shape = image.shape[:2]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        deformed_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            deformed_image[:, :, c] = map_coordinates(image[:, :, c], indices, order=1, mode='reflect').reshape(shape)
        deformed_label = map_coordinates(label, indices, order=0, mode='constant', cval=0).reshape(shape)
        return deformed_image, deformed_label

    def _apply_coarse_dropout(self, image, label, max_holes=8, min_holes=4, max_height=20, min_height=10, max_width=20, min_width=10, fill_value=0):
        h, w, _ = image.shape
        num_holes = random.randint(min_holes, max_holes)
        for _ in range(num_holes):
            hole_h = random.randint(min_height, max_height)
            hole_w = random.randint(min_width, max_width)
            y1 = random.randint(0, h - hole_h) if h - hole_h > 0 else 0
            x1 = random.randint(0, w - hole_w) if w - hole_w > 0 else 0
            y2, x2 = y1 + hole_h, x1 + hole_w
            image[y1:y2, x1:x2, :] = fill_value
            label[y1:y2, x1:x2] = fill_value
        return image, label

    def _apply_gamma_correction(self, image):
        if random.random() < 0.5:
            gamma = random.uniform(0.7, 1.5)
            img_min, img_max = image.min(), image.max()
            image_pos = image - img_min
            img_range = img_max - img_min
            if img_range > 1e-5:
                image_pos = np.power(image_pos / img_range, gamma) * img_range
                image = image_pos + img_min
        return image

    def _apply_additive_noise(self, image):
        if random.random() < 0.3:
            std = random.uniform(0, 0.1) * (image.max() - image.min())
            noise = np.random.normal(0, std, image.shape)
            image = image + noise
        return image

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        H, W, C = image.shape
        
        if random.random() < 0.5:
            image, label = cv2.flip(image, 1), cv2.flip(label, 1)
        if random.random() < 0.5:
            image, label = cv2.flip(image, 0), cv2.flip(label, 0)

        if random.random() < 0.75:
            center = (W // 2, H // 2)
            angle, scale = random.uniform(-20, 20), random.uniform(0.8, 1.2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            shear_x, shear_y = random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)
            M[0, 2] += shear_x * center[1]
            M[1, 2] += shear_y * center[0]
            image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            label = cv2.warpAffine(label, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        if random.random() < 0.5:
            image, label = self._apply_elastic_transform(image, label)

        image = self._apply_gamma_correction(image)
        image = self._apply_additive_noise(image)
        
        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.2, 0.2)
            beta = random.uniform(-0.2, 0.2)
            image = image * alpha + beta

        if random.random() < 0.5:
            image, label = self._apply_coarse_dropout(image, label)

        h, w, _ = image.shape
        pad_h, pad_w = max(self.output_size[0] - h, 0), max(self.output_size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top, pad_left = pad_h // 2, pad_w // 2
            image = cv2.copyMakeBorder(image, pad_top, pad_h-pad_top, pad_left, pad_w-pad_left, cv2.BORDER_CONSTANT, value=0)
            label = cv2.copyMakeBorder(label, pad_top, pad_h-pad_top, pad_left, pad_w-pad_left, cv2.BORDER_CONSTANT, value=0)
        
        h, w, _ = image.shape
        start_h = random.randint(0, h - self.output_size[0])
        start_w = random.randint(0, w - self.output_size[1])
        
        image = image[start_h:start_h+self.output_size[0], start_w:start_w+self.output_size[1]]
        label = label[start_h:start_h+self.output_size[0], start_w:start_w+self.output_size[1]]

        image_t = torch.from_numpy(image.copy().astype(np.float32)).permute(2, 0, 1)
        label_t = torch.from_numpy(label.copy().astype(np.float32)).unsqueeze(0)
        
        return {'image': image_t, 'label': label_t}

class WMH_dataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None, oversample_factor=4,
                 difficulty_percentile=25, difficulty_bonus=4):
        super().__init__()
        self.split = split
        self.transform = transform
        
        root = resolve_split_root(base_dir, split)
        if root is None: raise RuntimeError(f"Could not locate data under base_dir={base_dir}.")
        
        flair_dir, t1_dir, mask_dir = os.path.join(root, 'flair'), os.path.join(root, 't1'), find_mask_dir(root)
        if not all(os.path.isdir(d) for d in [flair_dir, t1_dir, mask_dir]):
            raise RuntimeError(f"Expected flair/t1/mask directories not found under {root}")

        self.root = root
        flair_files = sorted(glob.glob(os.path.join(flair_dir, "*.nii*")))
        self.file_triplets = []
        for flair_path in flair_files:
            base_name = os.path.basename(flair_path).replace('_flair.nii.gz','').replace('_flair.nii','')
            t1_path = os.path.join(t1_dir, f"{base_name}_t1.nii.gz")
            if not os.path.exists(t1_path): t1_path = t1_path[:-3]
            mask_path = os.path.join(mask_dir, f"{base_name}_wmh.nii.gz")
            if not os.path.exists(mask_path): mask_path = mask_path[:-3]
            if os.path.exists(t1_path) and os.path.exists(mask_path):
                self.file_triplets.append({'flair': flair_path, 't1': t1_path, 'mask': mask_path, 'name': base_name})
        
        if not self.file_triplets: raise RuntimeError(f"No valid file triplets found under {root}.")

        self.slice_map = []
        if self.split == 'train':
            print("Analyzing lesion sizes to determine difficulty threshold...")
            lesion_volumes = []
            for trip in tqdm(self.file_triplets, desc="Pre-scanning masks"):
                try:
                    mask_vol = nib.load(trip['mask']).get_fdata()
                    for s in range(mask_vol.shape[2]):
                        lesion_sum = mask_vol[:, :, s].sum()
                        if lesion_sum > 0:
                            lesion_volumes.append(lesion_sum)
                except Exception as e:
                    print(f"\nWarning during pre-scan for {trip['name']}: {e}")

            if not lesion_volumes:
                print("Warning: No lesions found in training set. Difficulty sampling disabled.")
                difficulty_threshold = 0
            else:
                difficulty_threshold = np.percentile(lesion_volumes, difficulty_percentile)
                print(f"Data-Driven Difficulty Threshold set to: {difficulty_threshold:.2f} (the {difficulty_percentile}th percentile of lesion sizes)")

            print(f"Building slice map with Difficulty-Aware Sampling...")
            for i, trip in enumerate(tqdm(self.file_triplets, desc="Building Slice Map")):
                try:
                    mask_vol = nib.load(trip['mask']).get_fdata()
                    for s in range(mask_vol.shape[2]):
                        lesion_sum = mask_vol[:, :, s].sum()
                        has_lesion = lesion_sum > 0
                        
                        self.slice_map.append({'subject_idx': i, 'slice_idx': s, 'has_lesion': has_lesion})
                        
                        if has_lesion:
                            bonus = difficulty_bonus if lesion_sum < difficulty_threshold else 0
                            total_oversamples = (oversample_factor - 1) + bonus
                            for _ in range(total_oversamples):
                                self.slice_map.append({'subject_idx': i, 'slice_idx': s, 'has_lesion': True})
                except Exception as e:
                    print(f"\nWarning: reading data for {trip['name']}: {e}")
            
            total_slices = len(self.slice_map)
            positive_slices = sum(1 for s in self.slice_map if s['has_lesion'])
            negative_slices = total_slices - positive_slices
            print("\n--- Training Set Slice Balance ---")
            print(f"Total slices in training playlist: {total_slices}")
            print(f"Slices with lesions (positive): {positive_slices} ({positive_slices/total_slices:.2%})")
            print(f"Slices without lesions (negative): {negative_slices} ({negative_slices/total_slices:.2%})")
            print("------------------------------------\n")

    def __len__(self):
        return len(self.slice_map) if self.split=='train' else len(self.file_triplets)

    @lru_cache(maxsize=8)
    def _load_volume(self, subject_idx):
        trip = self.file_triplets[subject_idx]
        return {'flair': nib.load(trip['flair']).get_fdata(dtype=np.float32),
                't1'   : nib.load(trip['t1']).get_fdata(dtype=np.float32),
                'mask' : nib.load(trip['mask']).get_fdata(dtype=np.float32)}

    def __getitem__(self, idx):
        if self.split == 'train':
            mi = self.slice_map[idx]
            subj_idx, sidx = mi['subject_idx'], mi['slice_idx']
            vols = self._load_volume(subj_idx)
            
            img_slice = np.stack([vols['flair'][:,:,sidx], vols['t1'][:,:,sidx]], axis=-1)
            lbl_slice = (vols['mask'][:,:,sidx] > 0.5).astype(np.float32)

            sample = {'image': img_slice, 'label': lbl_slice}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
        else:
            return {'triplet': self.file_triplets[idx]}
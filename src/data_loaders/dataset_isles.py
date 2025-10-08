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
        gamma = random.uniform(0.7, 1.5)
        img_min, img_max = image.min(), image.max()
        img_range = img_max - img_min
        if img_range > 1e-5:
            image_pos = image - img_min
            image_pos = np.power(image_pos / img_range, gamma) * img_range
            image = image_pos + img_min
        return image

    def _apply_additive_noise(self, image):
        std = random.uniform(0, 0.1) * (image.max() - image.min())
        noise = np.random.normal(0, std, image.shape)
        return image + noise

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        H, W, C = image.shape
        
        if random.random() < 0.5: image, label = cv2.flip(image, 1), cv2.flip(label, 1)
        if random.random() < 0.5: image, label = cv2.flip(image, 0), cv2.flip(label, 0)
        
        if random.random() < 0.75:
            center = (W // 2, H // 2)
            angle = random.uniform(-15, 15)
            scale = random.uniform(0.85, 1.15)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            shear_x, shear_y = random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)
            M[0, 2] += shear_x * center[1]; M[1, 2] += shear_y * center[0]
            image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            label = cv2.warpAffine(label, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
        if random.random() < 0.3: image, label = self._apply_elastic_transform(image, label)
        
        if random.random() < 0.3: image = self._apply_gamma_correction(image)
        if random.random() < 0.3: image = self._apply_additive_noise(image)
        
        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.2, 0.2)
            beta = random.uniform(-0.2, 0.2)
            image = image * alpha + beta
            
        if random.random() < 0.25: image, label = self._apply_coarse_dropout(image, label)
        
        h, w, _ = image.shape
        pad_h = max(self.output_size[0] - h, 0)
        pad_w = max(self.output_size[1] - w, 0)
        
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2; pad_right = pad_w - pad_left
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            label = cv2.copyMakeBorder(label, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        
        h, w, _ = image.shape
        start_h = random.randint(0, h - self.output_size[0]) if h > self.output_size[0] else 0
        start_w = random.randint(0, w - self.output_size[1]) if w > self.output_size[1] else 0
        
        image = image[start_h:start_h+self.output_size[0], start_w:start_w+self.output_size[1]]
        label = label[start_h:start_h+self.output_size[0], start_w:start_w+self.output_size[1]]
        
        image_t = torch.from_numpy(image.copy().astype(np.float32)).permute(2, 0, 1)
        label_t = torch.from_numpy(label.copy().astype(np.float32)).unsqueeze(0)
        
        return {'image': image_t, 'label': label_t}


class stroke_dataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None, oversample_factor=4,
                 difficulty_percentile=25, difficulty_bonus=4):
        super().__init__()
        self.split = split
        self.transform = transform
        
        self.root = os.path.join(base_dir, 'ISLES-2022/ISLES-2022')
        
        all_subject_dirs = sorted([d for d in os.listdir(self.root) if d.startswith('sub-') and os.path.isdir(os.path.join(self.root, d))])
        
        subject_selection = []
        if split == 'train':
            subject_selection = all_subject_dirs[:150]
        else:
            subject_selection = all_subject_dirs[150:]
        
        print(f"Found {len(subject_selection)} potential subjects for {split} split. Verifying complex file paths...")
        
        self.file_triplets = []
        skipped_count = 0
        
        for sub_dir in tqdm(subject_selection, desc=f"Verifying {split} files"):
            dwi_base_dir = os.path.join(self.root, sub_dir, 'ses-0001', 'dwi')
            mask_base_dir = os.path.join(self.root, 'derivatives', sub_dir, 'ses-0001')

            dwi_path, adc_path, mask_path = None, None, None

            try:
                dwi_subdir_list = glob.glob(os.path.join(dwi_base_dir, '*_dwi.nii'))
                if not dwi_subdir_list: raise IndexError("DWI intermediate directory not found")
                dwi_subdir = dwi_subdir_list[0]
                
                dwi_path_list = glob.glob(os.path.join(dwi_subdir, '*.nii*'))
                if not dwi_path_list: raise IndexError("Actual DWI file not found")
                dwi_path = dwi_path_list[0]

                adc_subdir_list = glob.glob(os.path.join(dwi_base_dir, '*_adc.nii'))
                if not adc_subdir_list: raise IndexError("ADC intermediate directory not found")
                adc_subdir = adc_subdir_list[0]
                
                adc_path_list = glob.glob(os.path.join(adc_subdir, '*.nii*'))
                if not adc_path_list: raise IndexError("Actual ADC file not found")
                adc_path = adc_path_list[0]
                
                mask_path_list = glob.glob(os.path.join(mask_base_dir, '*_msk.nii*'))
                if not mask_path_list: raise IndexError("Mask file not found")
                mask_path = mask_path_list[0]
                
                nib.load(dwi_path)
                nib.load(adc_path)
                nib.load(mask_path)
                
                self.file_triplets.append({'dwi': dwi_path, 'adc': adc_path, 'mask': mask_path, 'name': sub_dir})

            except (IndexError, nib.filebasedimages.ImageFileError, EOFError):
                skipped_count += 1
                continue
                
        if skipped_count > 0:
            print(f"Skipped a total of {skipped_count} subjects due to missing, corrupted, or structurally incorrect files.")
        
        if not self.file_triplets: 
            raise RuntimeError(f"FATAL: No valid, readable cases found for the '{split}' split. Please double-check the unique directory structure.")
        
        print(f"Successfully located and verified {len(self.file_triplets)} valid cases for the '{split}' split.")

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
                print(f"Data-Driven Difficulty Threshold set to: {difficulty_threshold:.2f} (the {difficulty_percentile}th percentile)")

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

    @lru_cache(maxsize=16)
    def _load_volume(self, subject_idx):
        trip = self.file_triplets[subject_idx]
        return {
            'dwi': nib.load(trip['dwi']).get_fdata(dtype=np.float32),
            'adc': nib.load(trip['adc']).get_fdata(dtype=np.float32),
            'mask': nib.load(trip['mask']).get_fdata(dtype=np.float32)
        }

    def __getitem__(self, idx):
        if self.split == 'train':
            map_info = self.slice_map[idx]
            subj_idx, slice_idx = map_info['subject_idx'], map_info['slice_idx']
            vols = self._load_volume(subj_idx)
            
            img_slice = np.stack([vols['dwi'][:,:,slice_idx], vols['adc'][:,:,slice_idx]], axis=-1)
            lbl_slice = (vols['mask'][:,:,slice_idx] > 0.5).astype(np.float32)
            
            sample = {'image': img_slice, 'label': lbl_slice}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
        else:
            return self.file_triplets[idx]
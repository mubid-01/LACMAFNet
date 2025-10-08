import os
import glob
import random
from functools import lru_cache

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from tqdm import tqdm

class BraTS_Augmentation:
    def __init__(self, output_size=(208, 208)):
        self.output_size = tuple(output_size)
        self.pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                shear=(-10, 10),
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.2
            ),
            A.GaussNoise(p=0.15),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.PadIfNeeded(
                min_height=self.output_size[0],
                min_width=self.output_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.RandomCrop(
                height=self.output_size[0],
                width=self.output_size[1]
            )
        ])

    def __call__(self, sample):
        img, lbl = sample['image'], sample['label']
        augmented = self.pipeline(image=img, mask=lbl)
        out_img, out_lbl = augmented['image'], augmented['mask']
        img_t = torch.from_numpy(out_img.astype(np.float32)).permute(2, 0, 1)
        lbl_t = torch.from_numpy(out_lbl.astype(np.int64))
        return {'image': img_t, 'label': lbl_t}

class BraTS_dataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None, oversample_factor=4,
                 num_subjects=None, val_split_fraction=0.2, seed=42):
        self.split = split
        self.transform = transform
        self.root = base_dir

        if not os.path.isdir(self.root):
            raise RuntimeError(f"The provided base_dir does not exist: {self.root}")

        all_subject_folders = sorted([
            os.path.join(self.root, d) for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])

        if num_subjects is not None and num_subjects > 0:
            print(f"Found {len(all_subject_folders)} total subjects. Using the first {num_subjects}.")
            all_subject_folders = all_subject_folders[:num_subjects]
        else:
            print(f"Found {len(all_subject_folders)} total subjects. Using all of them.")

        if not all_subject_folders:
            raise RuntimeError(f"No subject folders found under {self.root}")

        random.seed(seed)
        random.shuffle(all_subject_folders)
        
        split_index = int(len(all_subject_folders) * (1 - val_split_fraction))
        if self.split == 'train':
            folders_to_use = all_subject_folders[:split_index]
            print(f"Using {len(folders_to_use)} subjects for training.")
        else:
            folders_to_use = all_subject_folders[split_index:]
            print(f"Using {len(folders_to_use)} subjects for validation.")

        self.subject_files = []
        for sub_folder in tqdm(folders_to_use, desc=f"Finding files for {split} split"):
            base_name = os.path.basename(sub_folder)
            files = {
                't1c': os.path.join(sub_folder, f"{base_name}-t1c.nii"),
                't1n': os.path.join(sub_folder, f"{base_name}-t1n.nii"),
                't2f': os.path.join(sub_folder, f"{base_name}-t2f.nii"),
                't2w': os.path.join(sub_folder, f"{base_name}-t2w.nii"),
                'seg': os.path.join(sub_folder, f"{base_name}-seg.nii"),
            }
            if all(os.path.exists(p) for p in files.values()):
                files['name'] = base_name
                self.subject_files.append(files)
        
        if not self.subject_files:
            raise RuntimeError(f"No valid BraTS subjects found for the '{self.split}' split.")

        self.slice_map = []
        if self.split == 'train':
            for i, subject in enumerate(tqdm(self.subject_files, desc="Building training slice map")):
                mask_vol = nib.load(subject['seg']).get_fdata()
                for s in range(mask_vol.shape[2]):
                    if mask_vol[:, :, s].sum() > 0:
                        for _ in range(oversample_factor):
                            self.slice_map.append({'subject_idx': i, 'slice_idx': s})
                    else:
                        self.slice_map.append({'subject_idx': i, 'slice_idx': s})

    def __len__(self):
        return len(self.slice_map) if self.split == 'train' else len(self.subject_files)

    @lru_cache(maxsize=4)
    def _load_volume(self, subject_idx):
        subject = self.subject_files[subject_idx]
        return {
            't1c': nib.load(subject['t1c']).get_fdata(dtype=np.float32),
            't1n': nib.load(subject['t1n']).get_fdata(dtype=np.float32),
            't2f': nib.load(subject['t2f']).get_fdata(dtype=np.float32),
            't2w': nib.load(subject['t2w']).get_fdata(dtype=np.float32),
            'mask': nib.load(subject['seg']).get_fdata().astype(np.int64)
        }

    def __getitem__(self, idx):
        if self.split == 'train':
            mi = self.slice_map[idx]
            subj_idx, sidx = mi['subject_idx'], mi['slice_idx']
            vols = self._load_volume(subj_idx)
            
            img_slice = np.stack([
                vols['t1c'][:, :, sidx],
                vols['t1n'][:, :, sidx],
                vols['t2f'][:, :, sidx],
                vols['t2w'][:, :, sidx]
            ], axis=-1)
            
            lbl_slice = vols['mask'][:, :, sidx]
            lbl_slice[lbl_slice == 4] = 3 
            
            sample = {'image': img_slice.astype(np.float32), 'label': lbl_slice}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
        else:
            # For validation, we load the whole volume
            vols = self._load_volume(idx)
            return {'volumes': vols, 'subject_info': self.subject_files[idx]}
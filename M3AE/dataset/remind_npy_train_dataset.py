"""
remind_npy_train_dataset.py

Training dataset for ReMIND .npy files compatible with m3ae training pipeline.

Place in: /workspace/IMFuse/IM-Fuse/m3ae/dataset/remind_npy_train_dataset.py
"""

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from .transforms import RandomRotion, RandomIntensityChange, RandomFlip, Compose, NumpyType, RandCrop3D

mask_array = np.array([
    # [True,  False, False, False],
    # [False, True,  False, False],
    # [False, False, True,  False],
    # [False, False, False, True],
    # [True,  True,  False, False],
    # [True,  False, True,  False],
    # [True,  False, False, True],
    # [False, True,  True,  False],
    # [False, True,  False, True],
    # [False, False, True,  True],
    # [True,  True,  True,  False],
    # [True,  True,  False, True],
    # [True,  False, True,  True],
    # [False, True,  True,  True],
    [False,  True,  False,  True],
])


class ReMIND_NpyTrainDataset(Dataset):
    """
    Training dataset — loads ReMIND .npy files with random crops
    and random missing modality masks.

    Returns same format as m3ae's Brats training dataset:
        data["image"]       → float32 tensor (4, 128, 128, 128)
        data["label"]       → float32 tensor (3, 128, 128, 128)  [ET, TC, WT]
        data["patient_id"]  → str
        data["crop_indexes"]→ ((zmin,zmax),(ymin,ymax),(xmin,xmax))
        data["mask_modal"]  → list of missing modality indices
    """

    def __init__(self, root, train_file, patch_shape=128):
        """
        Args:
            root:       path to np folder e.g. /workspace/remind/ReMIND_Intra_operative/np
            train_file: txt file with one case name per line e.g. remind_train.txt
            patch_shape: size of random crop (default 128)
        """
        with open(train_file, 'r') as f:
            self.names = [l.strip() for l in f.readlines() if l.strip()]

        self.volpaths = [os.path.join(root, 'vol', f"{n}_vol.npy") for n in self.names]
        self.segpaths = [os.path.join(root, 'seg', f"{n}_seg.npy") for n in self.names]
        self.patch_shape = patch_shape

    def __getitem__(self, idx):
        # ---- load vol (X, Y, Z, 4) → (4, X, Y, Z) ----
        vol = np.load(self.volpaths[idx]).astype(np.float32)
        vol = np.transpose(vol, (3, 0, 1, 2))   # (4, X, Y, Z)

        # ---- load seg (X, Y, Z) ----
        seg = np.load(self.segpaths[idx]).astype(np.uint8)

        # ---- convert raw labels → 3-channel binary [ET, TC, WT] ----
        ET = (seg == 3).astype(np.float32)
        TC = np.logical_or(seg == 1, seg == 3).astype(np.float32)
        WT = (seg > 0).astype(np.float32)
        label = np.stack([ET, TC, WT], axis=0)   # (3, X, Y, Z)

        # ---- pad to at least patch_shape in every dim ----
        MIN = self.patch_shape

        def pad_to_min(arr):
            pads = [(0, 0)]
            for s in arr.shape[1:]:
                p = max(0, MIN - s)
                pads.append((p // 2, p - p // 2))
            return np.pad(arr, pads, mode='constant')

        vol   = pad_to_min(vol)    # (4, >=128, >=128, >=128)
        label = pad_to_min(label)  # (3, >=128, >=128, >=128)

        # ---- random crop to patch_shape ----
        _, D, H, W = vol.shape
        P = self.patch_shape
        
        # random crop — current
        # zs = np.random.randint(0, max(1, D - P + 1))
        # ys = np.random.randint(0, max(1, H - P + 1))
        # xs = np.random.randint(0, max(1, W - P + 1))
        
        # center crop — new
        zs = (D - P) // 2
        ys = (H - P) // 2
        xs = (W - P) // 2
        
        vol   = vol[:,   zs:zs+P, ys:ys+P, xs:xs+P]
        label = label[:, zs:zs+P, ys:ys+P, xs:xs+P]

        crop_indexes = (
            (zs, zs + P),
            (ys, ys + P),
            (xs, xs + P),
        )

        # ---- random augmentation ----
        # transpose to (1, X, Y, Z, C) for transforms then back
        vol_t   = vol.transpose(1, 2, 3, 0)[None, ...]    # (1, D, H, W, 4)
        label_t = label.transpose(1, 2, 3, 0)[None, ...]  # (1, D, H, W, 3)

        # random flip
        if np.random.random() > 0.5:
            axis = np.random.randint(0, 3)
            vol_t   = np.flip(vol_t,   axis=axis+1).copy()
            label_t = np.flip(label_t, axis=axis+1).copy()

        # random intensity change
        factor = 1.0 + np.random.uniform(-0.1, 0.1)
        vol_t = vol_t * factor

        vol   = vol_t[0].transpose(3, 0, 1, 2)    # (4, D, H, W)
        label = label_t[0].transpose(3, 0, 1, 2)  # (3, D, H, W)

        # ---- random missing modality mask ----
        mask_idx   = np.random.randint(0, 1)
        mask_modal = [i for i, v in enumerate(mask_array[mask_idx]) if not v]  # missing indices

        # zero out missing modalities
        vol_masked = vol.copy()
        for m in mask_modal:
            vol_masked[m] = 0.0

        vol_tensor   = torch.from_numpy(vol_masked.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.float32))

        return dict(
            patient_id   = self.names[idx],
            image        = vol_tensor,    # (4, 128, 128, 128)
            label        = label_tensor,  # (3, 128, 128, 128)
            crop_indexes = crop_indexes,
            mask_modal   = mask_modal,
        )

    def __len__(self):
        return len(self.names)


class ReMIND_NpyValDataset(Dataset):
    """
    Validation/test dataset — loads ReMIND .npy files with center crop
    and fixed missing modality mask per sample from CSV.

    Returns same format as m3ae's BratsEval validation dataset.
    """

    def __init__(self, root, csv_file, patch_shape=128):
        """
        Args:
            root:     path to np folder
            csv_file: CSV with 'case' and 'mask' columns
                      mask is a list like [True, False, True, False]
                      True = modality available, False = missing
        """
        import ast
        df = pd.read_csv(csv_file)
        self.names    = df['case'].tolist()
        self.volpaths = [os.path.join(root, 'vol', f"{n}_vol.npy") for n in self.names]
        self.segpaths = [os.path.join(root, 'seg', f"{n}_seg.npy") for n in self.names]
        self.masks    = df['mask'].apply(ast.literal_eval).tolist()
        self.patch_shape = patch_shape

    def __getitem__(self, idx):
        vol = np.load(self.volpaths[idx]).astype(np.float32)
        vol = np.transpose(vol, (3, 0, 1, 2))

        seg = np.load(self.segpaths[idx]).astype(np.uint8)

        ET = (seg == 3).astype(bool)
        TC = np.logical_or(seg == 1, seg == 3).astype(bool)
        WT = (seg > 0).astype(bool)
        label = np.stack([ET, TC, WT], axis=0)

        et_present = 1 if ET.sum() >= 1 else 0

        # brain bbox crop
        z_idx, y_idx, x_idx = np.nonzero(vol.sum(axis=0) != 0)
        if len(z_idx) == 0:
            zmin, ymin, xmin = 0, 0, 0
            zmax, ymax, xmax = vol.shape[1], vol.shape[2], vol.shape[3]
        else:
            zmin = max(0, int(z_idx.min()) - 1)
            ymin = max(0, int(y_idx.min()) - 1)
            xmin = max(0, int(x_idx.min()) - 1)
            zmax = int(z_idx.max()) + 1
            ymax = int(y_idx.max()) + 1
            xmax = int(x_idx.max()) + 1

        vol   = vol[:,   zmin:zmax, ymin:ymax, xmin:xmax]
        label = label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        # pad to min 128
        MIN = self.patch_shape
        def pad_to_min(arr):
            pads = [(0, 0)]
            for s in arr.shape[1:]:
                p = max(0, MIN - s)
                pads.append((p // 2, p - p // 2))
            return np.pad(arr, pads, mode='constant')

        vol   = pad_to_min(vol).astype(np.float16)
        label = pad_to_min(label).astype(bool)

        # get missing modalities from mask
        mask       = self.masks[idx]   # e.g. [True, False, True, False]
        mask_modal = [i for i, v in enumerate(mask) if not v]

        vol   = torch.from_numpy(vol)
        label = torch.from_numpy(label)

        return dict(
            patient_id   = self.names[idx],
            image        = vol,
            label        = label,
            crop_indexes = ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            mask_modal   = mask_modal,
            et_present   = et_present,
            supervised   = True,
            idx          = idx,
        )

    def __len__(self):
        return len(self.names)
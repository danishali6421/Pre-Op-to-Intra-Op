# """
# remind_npy_dataset.py

# Bridge dataset — loads ReMIND .npy files and returns data in the
# same format as m3ae's BratsEval so the eval_step function works
# without any changes.

# Place this file in: /workspace/IMFuse/IM-Fuse/m3ae/dataset/remind_npy_dataset.py
# """

# import os
# import numpy as np
# import torch
# import pandas as pd
# from torch.utils.data import Dataset


# class ReMIND_NpyDataset(Dataset):
#     """
#     Loads pre-processed ReMIND .npy files and returns dicts
#     compatible with m3ae's eval_step:

#         data["image"]       → float16 tensor (4, 155, 240, 240)
#         data["label"]       → bool tensor   (3, 155, 240, 240)  [ET, TC, WT]
#         data["patient_id"]  → str
#         data["crop_indexes"]→ ((zmin,zmax),(ymin,ymax),(xmin,xmax))
#         data["et_present"]  → int
#         data["supervised"]  → True
#         data["idx"]         → int
#     """

#     def __init__(self, root, csv_file):
#         """
#         Args:
#             root:     path to np folder  e.g. /workspace/remind/ReMIND_Intra_operative/np
#             csv_file: CSV with a 'case' column  e.g. /workspace/remind.csv
#         """
#         df = pd.read_csv(csv_file)
#         self.names    = df['case'].tolist()
#         self.volpaths = [os.path.join(root, 'vol', f"{n}_vol.npy") for n in self.names]
#         self.segpaths = [os.path.join(root, 'seg', f"{n}_seg.npy") for n in self.names]

#     def __getitem__(self, idx):
#         # ---- load vol (X, Y, Z, 4) → (4, X, Y, Z) ----
#         vol = np.load(self.volpaths[idx]).astype(np.float32)
#         vol = np.transpose(vol, (3, 0, 1, 2))   # (4, X, Y, Z)

#         # ---- load seg (X, Y, Z) ----
#         seg = np.load(self.segpaths[idx]).astype(np.uint8)

#         # ---- convert raw labels → 3-channel binary [ET, TC, WT] ----
#         ET = (seg == 3)
#         TC = np.logical_or(seg == 1, seg == 3)
#         WT = seg > 0
#         label = np.stack([ET, TC, WT], axis=0)   # (3, X, Y, Z)

#         et_present = 1 if ET.sum() >= 1 else 0

#         # ---- crop to brain bounding box ----
#         z_idx, y_idx, x_idx = np.nonzero(vol.sum(axis=0) != 0)
#         if len(z_idx) == 0:
#             zmin, ymin, xmin = 0, 0, 0
#             zmax = vol.shape[1]
#             ymax = vol.shape[2]
#             xmax = vol.shape[3]
#         else:
#             zmin = max(0, int(z_idx.min()) - 1)
#             ymin = max(0, int(y_idx.min()) - 1)
#             xmin = max(0, int(x_idx.min()) - 1)
#             zmax = int(z_idx.max()) + 1
#             ymax = int(y_idx.max()) + 1
#             xmax = int(x_idx.max()) + 1

#         vol   = vol[:,   zmin:zmax, ymin:ymax, xmin:xmax]
#         label = label[:, zmin:zmax, ymin:ymax, xmin:xmax]

#         # ---- pad OR crop to exactly 155x240x240 to match limage in Unet.py ----
#         # limage is fixed at (1, 4, 155, 240, 240) so volumes must match exactly
#         TARGET = (155, 240, 240)

#         def pad_or_crop_to_size(arr, target):
#             # arr: (C, Z, Y, X)
#             # step 1: crop if any dim is bigger than target
#             slices = [slice(None)]  # channel dim — no crop
#             for s, t in zip(arr.shape[1:], target):
#                 slices.append(slice(0, min(s, t)))
#             arr = arr[tuple(slices)]

#             # step 2: pad if any dim is smaller than target
#             pads = [(0, 0)]  # channel dim — no pad
#             for s, t in zip(arr.shape[1:], target):
#                 p = max(0, t - s)
#                 pads.append((p // 2, p - p // 2))
#             return np.pad(arr, pads, mode='constant')

#         vol   = pad_or_crop_to_size(vol,   TARGET)   # (4, 155, 240, 240)
#         label = pad_or_crop_to_size(label, TARGET)   # (3, 155, 240, 240)

#         print(f"  [{self.names[idx]}] final vol shape: {vol.shape}", flush=True)

#         # ---- cast to float16 / bool like BratsEval ----
#         vol   = vol.astype(np.float16)
#         label = label.astype(bool)

#         vol   = torch.from_numpy(vol)
#         label = torch.from_numpy(label)

#         return dict(
#             patient_id   = self.names[idx],
#             image        = vol,
#             label        = label,
#             crop_indexes = ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
#             et_present   = et_present,
#             supervised   = True,
#             idx          = idx,
#         )

#     def __len__(self):
#         return len(self.names)


"""
remind_npy_dataset.py

Bridge dataset — loads ReMIND .npy files and returns data in the
same format as m3ae's BratsEval so the eval_step function works
without any changes.

Place this file in: /workspace/IMFuse/IM-Fuse/m3ae/dataset/remind_npy_dataset.py
"""

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class ReMIND_NpyDataset(Dataset):
    """
    Loads pre-processed ReMIND .npy files and returns dicts
    compatible with m3ae's eval_step:

        data["image"]       → float16 tensor (4, >=128, >=128, >=128)
        data["label"]       → bool tensor   (3, >=128, >=128, >=128)  [ET, TC, WT]
        data["patient_id"]  → str
        data["crop_indexes"]→ ((zmin,zmax),(ymin,ymax),(xmin,xmax))
        data["et_present"]  → int
        data["supervised"]  → True
        data["idx"]         → int
    """

    def __init__(self, root, csv_file):
        df = pd.read_csv(csv_file)
        self.names    = df['case'].tolist()
        self.volpaths = [os.path.join(root, 'vol', f"{n}_vol.npy") for n in self.names]
        self.segpaths = [os.path.join(root, 'seg', f"{n}_seg.npy") for n in self.names]

    def __getitem__(self, idx):
        # ---- load vol (X, Y, Z, 4) → (4, X, Y, Z) ----
        vol = np.load(self.volpaths[idx]).astype(np.float32)
        vol = np.transpose(vol, (3, 0, 1, 2))   # (4, X, Y, Z)

        # ---- load seg (X, Y, Z) ----
        seg = np.load(self.segpaths[idx]).astype(np.uint8)

        # ---- convert raw labels → 3-channel binary [ET, TC, WT] ----
        ET = (seg == 3)
        TC = np.logical_or(seg == 1, seg == 3)
        WT = seg > 0
        label = np.stack([ET, TC, WT], axis=0)   # (3, X, Y, Z)

        et_present = 1 if ET.sum() >= 1 else 0

        # ---- crop to brain bounding box ----
        z_idx, y_idx, x_idx = np.nonzero(vol.sum(axis=0) != 0)
        if len(z_idx) == 0:
            zmin, ymin, xmin = 0, 0, 0
            zmax = vol.shape[1]
            ymax = vol.shape[2]
            xmax = vol.shape[3]
        else:
            zmin = max(0, int(z_idx.min()) - 1)
            ymin = max(0, int(y_idx.min()) - 1)
            xmin = max(0, int(x_idx.min()) - 1)
            zmax = int(z_idx.max()) + 1
            ymax = int(y_idx.max()) + 1
            xmax = int(x_idx.max()) + 1

        vol   = vol[:,   zmin:zmax, ymin:ymax, xmin:xmax]
        label = label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        # ---- pad to at least 128 in every spatial dim (NO cropping) ----
        # sliding window needs at least 128x128x128 to extract patches
        MIN = 128

        def pad_to_min(arr):
            # arr: (C, Z, Y, X)
            pads = [(0, 0)]   # channel dim — no pad
            for s in arr.shape[1:]:
                p = max(0, MIN - s)
                pads.append((p // 2, p - p // 2))
            return np.pad(arr, pads, mode='constant')

        vol   = pad_to_min(vol)    # (4, >=128, >=128, >=128)
        label = pad_to_min(label)  # (3, >=128, >=128, >=128)

        print(f"  [{self.names[idx]}] vol shape: {vol.shape}", flush=True)

        # ---- cast to float16 / bool like BratsEval ----
        vol   = vol.astype(np.float16)
        label = label.astype(bool)

        vol   = torch.from_numpy(vol)
        label = torch.from_numpy(label)

        return dict(
            patient_id   = self.names[idx],
            image        = vol,
            label        = label,
            crop_indexes = ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            et_present   = et_present,
            supervised   = True,
            idx          = idx,
        )

    def __len__(self):
        return len(self.names)
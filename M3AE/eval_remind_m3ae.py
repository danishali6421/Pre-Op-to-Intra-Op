# import argparse
# import os
# import pathlib
# import time
# import csv

# from medpy.metric import binary

# import numpy as np
# import torch
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.nn as nn
# import nibabel as nib
# from monai.data import decollate_batch
# from tensorboardX import SummaryWriter

# from loss import EDiceLoss
# from loss.dice import EDiceLoss_Val
# from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
#     count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
#     dice_metric_batch, reload_ckpt
# from model.Unet import Unet_missing

# # bridge dataset for ReMIND npy files
# from dataset.remind_npy_dataset import ReMIND_NpyDataset

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
# torch.cuda.set_device(0)

# masks = [
#     [False, True, False, True]
# ]

# mask_name = [
#     't2'
# ]

# parser = argparse.ArgumentParser(description='EVAL ReMIND with m3ae')
# parser.add_argument('--modal_list', nargs='+')
# parser.add_argument('-j', '--workers', default=4, type=int)
# parser.add_argument('--mdp', default=0, type=int)
# parser.add_argument('-b', '--batch-size', default=1, type=int)
# parser.add_argument('--lr', default=3e-4, type=float, dest='lr')
# parser.add_argument('--wd', default=0, type=float, dest='weight_decay')
# parser.add_argument('--devices', default='0', type=str)
# parser.add_argument('--checkpoint', default='/workspace/IMFuse/IM-Fuse/m3ae/runs/m3ae_train/model_1model_best_259.pth.tar', type=str)
# parser.add_argument('--exp_name', default='remind_eval', type=str)
# parser.add_argument('--fold', default=0, type=int)
# parser.add_argument('--num_classes', type=int, default=3)
# parser.add_argument('--seed', type=int, default=1234)
# parser.add_argument('--resume', default=True, type=bool)
# parser.add_argument('--mae_imp', default=True, type=bool)
# parser.add_argument('--datapath', default='/workspace/remind/ReMIND_Intra_operative/np', type=str)
# parser.add_argument('--csv_file', default='/workspace/remind.csv', type=str)
# parser.add_argument('--nifti_aligned_root', default='/workspace/remind/ReMIND_Intra_operative/NIFTI_ALIGNED', type=str)
# parser.add_argument('--pred_save_dir', default='./ReMIND_Intra_operative/pred_m3ae', type=str)

# device = torch.device("cuda:0")


# def main(args):
#     ngpus = torch.cuda.device_count()
#     print(f"Working with {ngpus} GPUs")
#     print(f"Checkpoint: {args.checkpoint}")

#     args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
#     args.save_folder_1.mkdir(parents=True, exist_ok=True)
#     args.seg_folder_1 = args.save_folder_1 / "segs"
#     args.seg_folder_1.mkdir(parents=True, exist_ok=True)
#     args.save_folder_1 = args.save_folder_1.resolve()
#     save_args_1(args)

#     t_writer_1 = SummaryWriter(str(args.save_folder_1))
#     print(args)

#     if args.modal_list:
#         args.modal_list = [int(l) for l in args.modal_list]
#     else:
#         args.modal_list = []

#     # ---- model ----
#     model_1 = Unet_missing(
#         input_shape=[128, 128, 128], out_channels=3, mdp=3,
#         init_channels=16, pre_train=False,
#         mask_modal=args.modal_list, patch_shape=128
#     )
#     model_1 = nn.DataParallel(model_1)
#     if args.resume:
#         ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
#         model_1.load_state_dict(ck["state_dict"], strict=False)

#     print(f"Total trainable parameters: {count_parameters(model_1)}")
#     model_1 = model_1.cuda()

#     # ---- dataset ----
#     print(f"Loading ReMIND data from: {args.datapath}")
#     print(f"Using CSV: {args.csv_file}")

#     l_test_dataset = ReMIND_NpyDataset(
#         root=args.datapath,
#         csv_file=args.csv_file
#     )
#     print(f"Test dataset size: {len(l_test_dataset)}")

#     test_loader = torch.utils.data.DataLoader(
#         l_test_dataset, batch_size=1, shuffle=False,
#         pin_memory=True, num_workers=args.workers
#     )

#     # ---- evaluation ----
#     criterian_val = EDiceLoss_Val().cuda()
#     metric = criterian_val.metric
#     all_dice = []
#     output_path = args.save_folder_1 / "test_final.txt"

#     for m_idx, m in enumerate(masks):
#         print(f"\n========== Mask: {mask_name[m_idx]} ==========")
#         test_loss, test_metrics, dice_mean = eval_step(
#             test_loader, model_1, metric, t_writer_1,
#             mask=m,
#             mask_name=mask_name[m_idx],
#             save_folder=output_path,
#             nifti_aligned_root=args.nifti_aligned_root,
#             pred_save_dir=args.pred_save_dir,
#         )
#         all_dice.append(dice_mean)

#     dice_avg = np.array(all_dice).mean(axis=0)

#     with open(output_path, 'a') as file:
#         file.write(
#             "Overall Averages: WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n".format(
#                 dice_avg[2], dice_avg[1], dice_avg[0]
#             )
#         )
#     print(f"\nAll results saved → {output_path}")


# def eval_step(data_loader, model, metric, writer, mask, mask_name,
#               save_folder=None, nifti_aligned_root=None, pred_save_dir=None):

#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time  = AverageMeter('Data', ':6.3f')
#     losses     = AverageMeter('Loss', ':.4e')

#     batch_per_epoch = len(data_loader)
#     progress = ProgressMeter(
#         batch_per_epoch,
#         [batch_time, data_time, losses],
#         prefix=f"Eval [{mask_name}]"
#     )

#     end = time.perf_counter()
#     metrics      = []
#     odice_metric = []
#     hd_metric    = []
#     hd95_metric  = []

#     model.module.mask_modal = [i for i, value in enumerate(mask) if value == False]

#     for i, val_data in enumerate(data_loader):
#         data_time.update(time.perf_counter() - end)

#         case_name = val_data["patient_id"][0]

#         model.eval()
#         with torch.no_grad():
#             val_inputs = val_data["image"].cuda()
#             val_labels = val_data["label"].cuda()

#             val_outputs   = inference(val_inputs, model)
#             val_outputs_1 = [post_trans(j) for j in decollate_batch(val_outputs)]

#             segs    = val_outputs
#             targets = val_labels
#             dice_metric(y_pred=val_outputs_1, y=val_labels)

#         metric_ = metric(segs, targets)
#         metrics.extend(metric_)

#         hd   = []
#         hd95 = []
#         dice = []
#         for l in range(segs.shape[1]):
#             if targets[0, l].cpu().numpy().sum() == 0:
#                 hd.append(1)
#                 hd95.append(0)
#                 dice.append(metric_[0][l].cpu().numpy())
#                 continue
#             if (segs[0, l].cpu().numpy() > 0.5).sum() == 0:
#                 hd.append(0)
#                 hd95.append(0)
#                 dice.append(metric_[0][l].cpu().numpy())
#                 continue

#             hd.append(binary.hd(
#                 segs[0, l].cpu().numpy() > 0.5,
#                 targets[0, l].cpu().numpy() > 0.5,
#                 voxelspacing=None
#             ))
#             hd95.append(binary.hd95(
#                 segs[0, l].cpu().numpy() > 0.5,
#                 targets[0, l].cpu().numpy() > 0.5,
#                 voxelspacing=None
#             ))
#             dice.append(metric_[0][l].cpu().numpy())

#         hd_metric.append(hd)
#         hd95_metric.append(hd95)
#         odice_metric.append(dice)

#         if len(dice) >= 3:
#             print(f"  [{case_name}] ET: {dice[0]:.4f}  TC: {dice[1]:.4f}  WT: {dice[2]:.4f}", flush=True)

#         # ---- convert 3-channel sigmoid → integer labels ----
#         # m3ae outputs (B, 3, X, Y, Z): channels = [ET, TC, WT]
#         seg_sigmoid = segs[0].cpu().numpy()   # (3, X, Y, Z)
#         ET_pred = seg_sigmoid[0] > 0.5
#         TC_pred = seg_sigmoid[1] > 0.5
#         WT_pred = seg_sigmoid[2] > 0.5

#         pred_np = np.zeros(ET_pred.shape, dtype=np.uint8)
#         pred_np[WT_pred] = 2   # edema
#         pred_np[TC_pred] = 1   # necrotic
#         pred_np[ET_pred] = 3   # enhancing tumour

#         # ---- save NIfTI prediction ----
#         if pred_save_dir is not None and nifti_aligned_root is not None:
#             original_mri_path = os.path.join(
#                 nifti_aligned_root, case_name, "T1POST_aligned.nii.gz"
#             )
#             if os.path.exists(original_mri_path):
#                 affine = nib.load(original_mri_path).affine

#                 # one folder per mask scenario
#                 mask_save_dir = os.path.join(pred_save_dir, mask_name)
#                 os.makedirs(mask_save_dir, exist_ok=True)

#                 nifti_save_path = os.path.join(mask_save_dir, f"{case_name}.nii.gz")
#                 nib.save(nib.Nifti1Image(pred_np, affine), nifti_save_path)
#                 np.save(nifti_save_path.replace(".nii.gz", ".npy"), pred_np)
#                 print(f"  [✓] Saved: {nifti_save_path}", flush=True)
#             else:
#                 print(f"  ⚠️  MRI not found: {original_mri_path}", flush=True)

#         batch_time.update(time.perf_counter() - end)
#         end = time.perf_counter()
#         progress.display(i)

#     dice_metric.reset()
#     dice_metric_batch.reset()

#     metricss  = list(zip(*metrics))
#     metrics   = [np.nanmean(torch.tensor(d, device="cpu").numpy()) for d in metricss]
#     dice_mean = [np.nanmean(l) for l in zip(*odice_metric)]

#     with save_folder.open("a") as file:
#         file.write(
#             'Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n'.format(
#                 mask,
#                 dice_mean[2].item(),
#                 dice_mean[1].item(),
#                 dice_mean[0].item()
#             )
#         )

#     print(f"\n→ [{mask_name}] ET: {dice_mean[0]:.4f}  TC: {dice_mean[1]:.4f}  WT: {dice_mean[2]:.4f}", flush=True)

#     return losses.avg, np.nanmean(metrics), dice_mean


# if __name__ == '__main__':
#     arguments = parser.parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
#     main(arguments)





# import argparse
# import os
# import pathlib
# import time
# import csv

# from medpy.metric import binary

# import numpy as np
# import torch
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.nn as nn
# import nibabel as nib
# from monai.data import decollate_batch
# from tensorboardX import SummaryWriter

# from loss import EDiceLoss
# from loss.dice import EDiceLoss_Val
# from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
#     count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
#     dice_metric_batch, reload_ckpt
# from model.Unet import Unet_missing

# # bridge dataset for ReMIND npy files
# from dataset.remind_npy_dataset import ReMIND_NpyDataset

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
# torch.cuda.set_device(0)

# # all 15 mask combinations
# masks = [
#     [False, True, False, True]
# ]

# mask_name = [
#     't2'
# ]

# parser = argparse.ArgumentParser(description='EVAL ReMIND with m3ae')
# parser.add_argument('--modal_list', nargs='+')
# parser.add_argument('-j', '--workers', default=4, type=int)
# parser.add_argument('--mdp', default=0, type=int)
# parser.add_argument('-b', '--batch-size', default=1, type=int)
# parser.add_argument('--lr', default=3e-4, type=float, dest='lr')
# parser.add_argument('--wd', default=0, type=float, dest='weight_decay')
# parser.add_argument('--devices', default='0', type=str)
# parser.add_argument('--checkpoint', default='/workspace/IMFuse/IM-Fuse/m3ae/runs/remind_finetune/model_1last.pth.tar', type=str)
# # parser.add_argument('--checkpoint', default='/workspace/IMFuse/IM-Fuse/m3ae/runs/m3ae_train/model_1model_best_259.pth.tar', type=str)
# parser.add_argument('--exp_name', default='remind_eval', type=str)
# parser.add_argument('--fold', default=0, type=int)
# parser.add_argument('--num_classes', type=int, default=3)
# parser.add_argument('--seed', type=int, default=1234)
# parser.add_argument('--resume', default=True, type=bool)
# parser.add_argument('--mae_imp', default=True, type=bool)
# parser.add_argument('--datapath', default='/workspace/remind/ReMIND_Intra_operative/np', type=str)
# parser.add_argument('--csv_file', default='/workspace/remind_val.csv', type=str)
# parser.add_argument('--nifti_aligned_root', default='/workspace/remind/ReMIND_Intra_operative/NIFTI_ALIGNED', type=str)
# parser.add_argument('--pred_save_dir', default='./ReMIND_Intra_operative/pred_m3ae', type=str)

# device = torch.device("cuda:0")


# def main(args):
#     ngpus = torch.cuda.device_count()
#     print(f"Working with {ngpus} GPUs")
#     print(f"Checkpoint: {args.checkpoint}")

#     args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
#     args.save_folder_1.mkdir(parents=True, exist_ok=True)
#     args.seg_folder_1 = args.save_folder_1 / "segs"
#     args.seg_folder_1.mkdir(parents=True, exist_ok=True)
#     args.save_folder_1 = args.save_folder_1.resolve()
#     save_args_1(args)

#     t_writer_1 = SummaryWriter(str(args.save_folder_1))
#     print(args)

#     if args.modal_list:
#         args.modal_list = [int(l) for l in args.modal_list]
#     else:
#         args.modal_list = []

#     # ---- model ----
#     model_1 = Unet_missing(
#         input_shape=[128, 128, 128], out_channels=3, mdp=3,
#         init_channels=16, pre_train=False,
#         mask_modal=args.modal_list, patch_shape=128
#     )
#     model_1 = nn.DataParallel(model_1)
#     if args.resume:
#         ck = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=False)
#         model_1.load_state_dict(ck["state_dict"], strict=False)

#     # ← add this
#     if 'epoch' in ck:
#         print(f"Model trained up to epoch: {ck['epoch']}, flush=True")
#     else:
#         print("No epoch info in checkpoint")

#     print(f"Total trainable parameters: {count_parameters(model_1)}")
#     model_1 = model_1.cuda()

#     # ---- dataset ----
#     print(f"Loading ReMIND data from: {args.datapath}")
#     print(f"Using CSV: {args.csv_file}")

#     l_test_dataset = ReMIND_NpyDataset(
#         root=args.datapath,
#         csv_file=args.csv_file
#     )
#     print(f"Test dataset size: {len(l_test_dataset)}")

#     test_loader = torch.utils.data.DataLoader(
#         l_test_dataset, batch_size=1, shuffle=False,
#         pin_memory=True, num_workers=args.workers
#     )

#     # ---- evaluation ----
#     criterian_val = EDiceLoss_Val().cuda()
#     metric = criterian_val.metric
#     all_dice = []
#     output_path = args.save_folder_1 / "test_final.txt"

#     for m_idx, m in enumerate(masks):
#         print(f"\n========== Mask: {mask_name[m_idx]} ==========")
#         test_loss, test_metrics, dice_mean = eval_step(
#             test_loader, model_1, metric, t_writer_1,
#             mask=m,
#             mask_name=mask_name[m_idx],
#             save_folder=output_path,
#             nifti_aligned_root=args.nifti_aligned_root,
#             pred_save_dir=args.pred_save_dir,
#         )
#         all_dice.append(dice_mean)

#     dice_avg = np.array(all_dice).mean(axis=0)

#     with open(output_path, 'a') as file:
#         file.write(
#             "Overall Averages: WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n".format(
#                 dice_avg[2], dice_avg[1], dice_avg[0]
#             )
#         )
#     print(f"\nAll results saved → {output_path}")


# def eval_step(data_loader, model, metric, writer, mask, mask_name,
#               save_folder=None, nifti_aligned_root=None, pred_save_dir=None):

#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time  = AverageMeter('Data', ':6.3f')
#     losses     = AverageMeter('Loss', ':.4e')

#     batch_per_epoch = len(data_loader)
#     progress = ProgressMeter(
#         batch_per_epoch,
#         [batch_time, data_time, losses],
#         prefix=f"Eval [{mask_name}]"
#     )

#     end = time.perf_counter()
#     metrics      = []
#     odice_metric = []
#     hd_metric    = []
#     hd95_metric  = []

#     # set missing modalities for this mask
#     # mask[i] = False means modality i is MISSING
#     model.module.mask_modal = [i for i, value in enumerate(mask) if value == False]
#     print(f"Missing modalities: {model.module.mask_modal}", flush=True)

#     for i, val_data in enumerate(data_loader):
#         data_time.update(time.perf_counter() - end)

#         case_name = val_data["patient_id"][0]

#         model.eval()
#         with torch.no_grad():
#             val_inputs = val_data["image"].cuda()
#             val_labels = val_data["label"].cuda()

#             val_outputs   = inference(val_inputs, model)
#             val_outputs_1 = [post_trans(j) for j in decollate_batch(val_outputs)]

#             segs    = val_outputs
#             targets = val_labels
#             dice_metric(y_pred=val_outputs_1, y=val_labels)

#         metric_ = metric(segs, targets)
#         metrics.extend(metric_)

#         hd   = []
#         hd95 = []
#         dice = []
#         for l in range(segs.shape[1]):
#             if targets[0, l].cpu().numpy().sum() == 0:
#                 hd.append(1)
#                 hd95.append(0)
#                 dice.append(metric_[0][l].cpu().numpy())
#                 continue
#             if (segs[0, l].cpu().numpy() > 0.5).sum() == 0:
#                 hd.append(0)
#                 hd95.append(0)
#                 dice.append(metric_[0][l].cpu().numpy())
#                 continue

#             hd.append(binary.hd(
#                 segs[0, l].cpu().numpy() > 0.5,
#                 targets[0, l].cpu().numpy() > 0.5,
#                 voxelspacing=None
#             ))
#             hd95.append(binary.hd95(
#                 segs[0, l].cpu().numpy() > 0.5,
#                 targets[0, l].cpu().numpy() > 0.5,
#                 voxelspacing=None
#             ))
#             dice.append(metric_[0][l].cpu().numpy())

#         hd_metric.append(hd)
#         hd95_metric.append(hd95)
#         odice_metric.append(dice)

#         if len(dice) >= 3:
#             print(f"  [{case_name}] ET: {dice[0]:.4f}  TC: {dice[1]:.4f}  WT: {dice[2]:.4f}", flush=True)

#         # ---- convert 3-channel sigmoid → integer labels ----
#         # m3ae outputs (B, 3, X, Y, Z): channels = [ET, TC, WT]
#         # pred_sig = torch.sigmoid(segs)   # (B, 3, H, W, D)
#         # pred_bin = (pred_sig > 0.5).float()
#         # ---- convert 3-channel sigmoid → integer labels ----
#         seg_sigmoid = segs[0].cpu().numpy()   # (3, X, Y, Z)
#         ET_pred = seg_sigmoid[0] >= 0.5
#         TC_pred = seg_sigmoid[1] >= 0.5
#         WT_pred = seg_sigmoid[2] >= 0.5
        
#         pred_np = np.zeros(ET_pred.shape, dtype=np.uint8)
#         pred_np[WT_pred] = 2
#         pred_np[TC_pred] = 1
#         pred_np[ET_pred] = 3
        
#         # ← ADD THIS BLOCK HERE — reconstruct full volume from crop
#         meta_path = os.path.join(
#             os.path.dirname(os.path.dirname(nifti_aligned_root)),  # go up to np folder
#             'np', 'meta', f"{case_name}_crop.npy"
#         )
#         # or use direct path
#         meta_path = f"/workspace/remind/ReMIND_Intra_operative/np/meta/{case_name}_crop.npy"
        
#         crop_info  = np.load(meta_path, allow_pickle=True).item()
#         x_min, x_max = crop_info["x_min"], crop_info["x_max"]
#         y_min, y_max = crop_info["y_min"], crop_info["y_max"]
#         z_min, z_max = crop_info["z_min"], crop_info["z_max"]
#         orig_shape   = crop_info["orig_shape"]
        
#         # pred_np is in padded+cropped space — trim to crop size then place in full volume
#         pred_crop = pred_np[:x_max-x_min, :y_max-y_min, :z_max-z_min]
#         full_mask  = np.zeros(orig_shape, dtype=np.uint8)
#         full_mask[x_min:x_max, y_min:y_max, z_min:z_max] = pred_crop
        
#         # ---- save NIfTI prediction ----
#         # if pred_save_dir is not None and nifti_aligned_root is not None:
#         #     original_mri_path = os.path.join(
#         #         nifti_aligned_root, case_name, "T1POST_aligned.nii.gz"
#         #     )
#         #     if os.path.exists(original_mri_path):
#         #         affine = nib.load(original_mri_path).affine
#         #         mask_save_dir = os.path.join(pred_save_dir, mask_name)
#         #         os.makedirs(mask_save_dir, exist_ok=True)
#         #         nifti_save_path = os.path.join(mask_save_dir, f"{case_name}.nii.gz")
#         #         nib.save(nib.Nifti1Image(full_mask, affine), nifti_save_path)  # ← full_mask not pred_np
#         #         np.save(nifti_save_path.replace(".nii.gz", ".npy"), pred_np)
#         #         print(f"  [✓] Saved: {nifti_save_path}", flush=True)
#         #     else:
#         #         print(f"  ⚠️  MRI not found: {original_mri_path}", flush=True)

#         batch_time.update(time.perf_counter() - end)
#         end = time.perf_counter()
#         progress.display(i)

#     dice_metric.reset()
#     dice_metric_batch.reset()

#     metricss  = list(zip(*metrics))
#     metrics   = [np.nanmean(torch.tensor(d, device="cpu").numpy()) for d in metricss]
#     dice_mean = [np.nanmean(l) for l in zip(*odice_metric)]

#     with save_folder.open("a") as file:
#         file.write(
#             'Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n'.format(
#                 mask,
#                 dice_mean[2].item(),
#                 dice_mean[1].item(),
#                 dice_mean[0].item()
#             )
#         )

#     print(f"\n→ [{mask_name}] ET: {dice_mean[0]:.4f}  TC: {dice_mean[1]:.4f}  WT: {dice_mean[2]:.4f}", flush=True)

#     return losses.avg, np.nanmean(metrics), dice_mean


# if __name__ == '__main__':
#     arguments = parser.parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
#     main(arguments)















# Matrics computation



import argparse
import os
import pathlib
import time
import csv

from medpy.metric import binary  # for hd (plain Hausdorff only -- see note below)

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
import nibabel as nib
from monai.data import decollate_batch
from tensorboardX import SummaryWriter
from scipy.ndimage import binary_dilation, distance_transform_edt

from loss import EDiceLoss
from loss.dice import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch, reload_ckpt
from model.Unet import Unet_missing

# bridge dataset for ReMIND npy files
from dataset.remind_npy_dataset import ReMIND_NpyDataset

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.cuda.set_device(0)


# =============================================================================
# SURFACE DISTANCE METRICS -- same dilation-based functions used in the
# IM-Fuse and BraTS eval scripts, so HD95/ASD/NSD here are numerically
# comparable to those. binary.hd() (plain, non-95th-percentile Hausdorff)
# has no equivalent here and is left on medpy (erosion-based border).
# =============================================================================

def compute_surface_distances(pred_mask, gt_mask):
    """Compute surface distances between pred and gt binary masks."""
    pred_border = pred_mask ^ binary_dilation(pred_mask)
    gt_border   = gt_mask   ^ binary_dilation(gt_mask)

    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return None, None

    dt_pred = distance_transform_edt(~pred_mask)
    dt_gt   = distance_transform_edt(~gt_mask)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    return d_pred_to_gt, d_gt_to_pred


def compute_hd95(pred_mask, gt_mask):
    """95th percentile Hausdorff Distance."""
    d1, d2 = compute_surface_distances(pred_mask, gt_mask)
    if d1 is None:
        return np.nan
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95))


def compute_asd(pred_mask, gt_mask):
    """Average Symmetric Surface Distance."""
    d1, d2 = compute_surface_distances(pred_mask, gt_mask)
    if d1 is None:
        return np.nan
    return float((d1.mean() + d2.mean()) / 2.0)


def compute_nsd(pred_mask, gt_mask, tolerance=1.0):
    """Normalized Surface Dice (NSD) at given tolerance (mm/voxels)."""
    d1, d2 = compute_surface_distances(pred_mask, gt_mask)
    if d1 is None:
        return np.nan
    pred_border = pred_mask ^ binary_dilation(pred_mask)
    gt_border   = gt_mask   ^ binary_dilation(gt_mask)
    n_pred = pred_border.sum()
    n_gt   = gt_border.sum()
    overlap = ((d1 <= tolerance).sum() + (d2 <= tolerance).sum())
    return float(overlap / (n_pred + n_gt)) if (n_pred + n_gt) > 0 else np.nan


# all 15 mask combinations
masks = [
    [False, True, False, True]
]

mask_name = [
    't2'
]

parser = argparse.ArgumentParser(description='EVAL ReMIND with m3ae')
parser.add_argument('--modal_list', nargs='+')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--mdp', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('--lr', default=3e-4, type=float, dest='lr')
parser.add_argument('--wd', default=0, type=float, dest='weight_decay')
parser.add_argument('--devices', default='0', type=str)
parser.add_argument('--checkpoint', default='./runs/remind_finetune_local/model_1last.pth.tar', type=str)
# parser.add_argument('--checkpoint', default='/workspace/IMFuse/IM-Fuse/m3ae/runs/m3ae_train/model_1model_best_259.pth.tar', type=str)
parser.add_argument('--exp_name', default='remind_eval', type=str)
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--resume', default=True, type=bool)
parser.add_argument('--mae_imp', default=True, type=bool)
# parser.add_argument('--datapath', default='/home/danish/data/ReMIND_preprocessed_data/miss_mod', type=str)
parser.add_argument('--datapath', default='../ReMIND_Intra_operative/np', type=str)
parser.add_argument('--csv_file', default='../remind_val_intra.csv', type=str)
parser.add_argument('--nifti_aligned_root', default='../ReMIND_Intra_operative/NIFTI_ALIGNED', type=str)
parser.add_argument('--pred_save_dir', default='./ReMIND_Intra_operative/pred_m3ae', type=str)

device = torch.device("cuda:0")


def main(args):
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    print(f"Checkpoint: {args.checkpoint}")

    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)

    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    print(args)

    if args.modal_list:
        args.modal_list = [int(l) for l in args.modal_list]
    else:
        args.modal_list = []

    # ---- model ----
    model_1 = Unet_missing(
        input_shape=[128, 128, 128], out_channels=3, mdp=3,
        init_channels=16, pre_train=False,
        mask_modal=args.modal_list, patch_shape=128
    )
    model_1 = nn.DataParallel(model_1)
    if args.resume:
        ck = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=False)
        model_1.load_state_dict(ck["state_dict"], strict=False)

    # ← add this
    if 'epoch' in ck:
        print(f"Model trained up to epoch: {ck['epoch']}, flush=True")
    else:
        print("No epoch info in checkpoint")

    print(f"Total trainable parameters: {count_parameters(model_1)}")
    model_1 = model_1.cuda()

    # ---- dataset ----
    print(f"Loading ReMIND data from: {args.datapath}")
    print(f"Using CSV: {args.csv_file}")

    l_test_dataset = ReMIND_NpyDataset(
        root=args.datapath,
        csv_file=args.csv_file
    )
    print(f"Test dataset size: {len(l_test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        l_test_dataset, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=args.workers
    )

    # ---- evaluation ----
    criterian_val = EDiceLoss_Val().cuda()
    metric = criterian_val.metric
    all_dice = []
    all_hd95 = []
    all_asd = []
    all_nsd = []
    output_path = args.save_folder_1 / "test_final.txt"

    for m_idx, m in enumerate(masks):
        print(f"\n========== Mask: {mask_name[m_idx]} ==========")
        test_loss, test_metrics, dice_mean, surface_stats = eval_step(
            test_loader, model_1, metric, t_writer_1,
            mask=m,
            mask_name=mask_name[m_idx],
            save_folder=output_path,
            nifti_aligned_root=args.nifti_aligned_root,
            pred_save_dir=args.pred_save_dir,
        )
        all_dice.append(dice_mean)
        all_hd95.append(surface_stats['hd95_mean'])
        all_asd.append(surface_stats['asd_mean'])
        all_nsd.append(surface_stats['nsd_mean'])

    dice_avg = np.array(all_dice).mean(axis=0)
    hd95_avg = np.nanmean(np.array(all_hd95), axis=0)
    asd_avg  = np.nanmean(np.array(all_asd), axis=0)
    nsd_avg  = np.nanmean(np.array(all_nsd), axis=0)

    with open(output_path, 'a') as file:
        file.write(
            "Overall Averages: WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n".format(
                dice_avg[2], dice_avg[1], dice_avg[0]
            )
        )
        file.write(
            "Overall Surface Metrics: "
            "HD95 (WT={:.2f}, TC={:.2f}, ET={:.2f}), "
            "ASD (WT={:.2f}, TC={:.2f}, ET={:.2f}), "
            "NSD (WT={:.4f}, TC={:.4f}, ET={:.4f})\n".format(
                hd95_avg[2], hd95_avg[1], hd95_avg[0],
                asd_avg[2], asd_avg[1], asd_avg[0],
                nsd_avg[2], nsd_avg[1], nsd_avg[0]
            )
        )

    print("Overall dice:", dice_avg)
    print("Overall HD95:", hd95_avg)
    print("Overall ASD:", asd_avg)
    print("Overall NSD:", nsd_avg)
    print(f"\nAll results saved → {output_path}")


def eval_step(data_loader, model, metric, writer, mask, mask_name,
              save_folder=None, nifti_aligned_root=None, pred_save_dir=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    losses     = AverageMeter('Loss', ':.4e')

    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"Eval [{mask_name}]"
    )

    end = time.perf_counter()
    metrics      = []
    odice_metric = []
    hd_metric    = []
    hd95_metric  = []
    asd_metric   = []
    nsd_metric   = []

    # set missing modalities for this mask
    # mask[i] = False means modality i is MISSING
    model.module.mask_modal = [i for i, value in enumerate(mask) if value == False]
    print(f"Missing modalities: {model.module.mask_modal}", flush=True)

    for i, val_data in enumerate(data_loader):
        data_time.update(time.perf_counter() - end)

        case_name = val_data["patient_id"][0]

        model.eval()
        with torch.no_grad():
            val_inputs = val_data["image"].cuda()
            val_labels = val_data["label"].cuda()

            val_outputs   = inference(val_inputs, model)
            val_outputs_1 = [post_trans(j) for j in decollate_batch(val_outputs)]

            segs    = val_outputs
            targets = val_labels
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        hd   = []
        hd95 = []
        asd  = []
        nsd  = []
        dice = []
        for l in range(segs.shape[1]):
            pred_l = segs[0, l].cpu().numpy() > 0.5
            gt_l   = targets[0, l].cpu().numpy() > 0.5
            dice_val = metric_[0][l].cpu().numpy()

            # Dice keeps its own sentinel convention (0/1 are meaningful Dice
            # values for a missed/empty region). For HD95 specifically: if
            # both pred and GT are empty for this region AND Dice ~= 1 (a
            # true negative / perfect agreement, correctly predicting
            # "nothing here"), HD95 = 0 rather than excluded, matching the
            # BraTS challenge convention. A genuine miss (one side empty,
            # the other isn't, Dice ~= 0) has no meaningful surface distance
            # and HD95 stays NaN, excluded from the nanmean aggregation
            # below. ASD/NSD keep the prior NaN-exclusion rule -- only HD95
            # was asked to change.
            if gt_l.sum() == 0 or pred_l.sum() == 0:
                if np.isclose(dice_val, 1.0, atol=1e-3):
                    hd95.append(0.0)
                else:
                    hd95.append(np.nan)
                hd.append(np.nan)
                asd.append(np.nan)
                nsd.append(np.nan)
                dice.append(dice_val)
                continue

            # plain (non-95th-percentile) Hausdorff distance -- no dilation-
            # based equivalent used elsewhere, kept on medpy
            hd.append(binary.hd(pred_l, gt_l, voxelspacing=None))

            # hd95 / asd / nsd -- dilation-based, matching the IM-Fuse and
            # BraTS eval scripts so results are directly comparable
            hd95.append(compute_hd95(pred_l, gt_l))
            asd.append(compute_asd(pred_l, gt_l))
            nsd.append(compute_nsd(pred_l, gt_l, tolerance=1.0))

            dice.append(dice_val)

        hd_metric.append(hd)
        hd95_metric.append(hd95)
        asd_metric.append(asd)
        nsd_metric.append(nsd)
        odice_metric.append(dice)

        if len(dice) >= 3:
            print(f"  [{case_name}] ET: {dice[0]:.4f}  TC: {dice[1]:.4f}  WT: {dice[2]:.4f}", flush=True)
            hd95_str = ' '.join(f"{v:.2f}" if not np.isnan(v) else "nan" for v in hd95)
            print(f"    HD95 [ET,TC,WT]: {hd95_str}", flush=True)

        # ---- convert 3-channel sigmoid → integer labels ----
        # m3ae outputs (B, 3, X, Y, Z): channels = [ET, TC, WT]
        # pred_sig = torch.sigmoid(segs)   # (B, 3, H, W, D)
        # pred_bin = (pred_sig > 0.5).float()
        # ---- convert 3-channel sigmoid → integer labels ----
        seg_sigmoid = segs[0].cpu().numpy()   # (3, X, Y, Z)
        ET_pred = seg_sigmoid[0] >= 0.5
        TC_pred = seg_sigmoid[1] >= 0.5
        WT_pred = seg_sigmoid[2] >= 0.5

        pred_np = np.zeros(ET_pred.shape, dtype=np.uint8)
        pred_np[WT_pred] = 2
        pred_np[TC_pred] = 1
        pred_np[ET_pred] = 3

        # ← ADD THIS BLOCK HERE — reconstruct full volume from crop
        # meta_path = os.path.join(
        #     os.path.dirname(os.path.dirname(nifti_aligned_root)),  # go up to np folder
        #     'np', 'meta', f"{case_name}_crop.npy"
        # )
        # # or use direct path
        # meta_path = f"/workspace/remind/ReMIND_Intra_operative/np/meta/{case_name}_crop.npy"

        # crop_info  = np.load(meta_path, allow_pickle=True).item()
        # x_min, x_max = crop_info["x_min"], crop_info["x_max"]
        # y_min, y_max = crop_info["y_min"], crop_info["y_max"]
        # z_min, z_max = crop_info["z_min"], crop_info["z_max"]
        # orig_shape   = crop_info["orig_shape"]

        # # pred_np is in padded+cropped space — trim to crop size then place in full volume
        # pred_crop = pred_np[:x_max-x_min, :y_max-y_min, :z_max-z_min]
        # full_mask  = np.zeros(orig_shape, dtype=np.uint8)
        # full_mask[x_min:x_max, y_min:y_max, z_min:z_max] = pred_crop

        # ---- save NIfTI prediction ----
        # if pred_save_dir is not None and nifti_aligned_root is not None:
        #     original_mri_path = os.path.join(
        #         nifti_aligned_root, case_name, "T1POST_aligned.nii.gz"
        #     )
        #     if os.path.exists(original_mri_path):
        #         affine = nib.load(original_mri_path).affine
        #         mask_save_dir = os.path.join(pred_save_dir, mask_name)
        #         os.makedirs(mask_save_dir, exist_ok=True)
        #         nifti_save_path = os.path.join(mask_save_dir, f"{case_name}.nii.gz")
        #         nib.save(nib.Nifti1Image(full_mask, affine), nifti_save_path)  # ← full_mask not pred_np
        #         np.save(nifti_save_path.replace(".nii.gz", ".npy"), pred_np)
        #         print(f"  [✓] Saved: {nifti_save_path}", flush=True)
        #     else:
        #         print(f"  ⚠️  MRI not found: {original_mri_path}", flush=True)

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        progress.display(i)

    dice_metric.reset()
    dice_metric_batch.reset()

    metricss  = list(zip(*metrics))
    metrics   = [np.nanmean(torch.tensor(d, device="cpu").numpy()) for d in metricss]

    # per-label (ET, TC, WT) aggregates -- np.nanmean correctly excludes
    # genuine-miss cases (NaN); perfect-match empty-region cases now
    # contribute HD95=0.
    hd_mean   = [np.nanmean(l) for l in zip(*hd_metric)]
    hd95_mean = [np.nanmean(l) for l in zip(*hd95_metric)]
    asd_mean  = [np.nanmean(l) for l in zip(*asd_metric)]
    nsd_mean  = [np.nanmean(l) for l in zip(*nsd_metric)]
    dice_mean = [np.nanmean(l) for l in zip(*odice_metric)]

    # how many cases actually contributed to each label's surface-metric average
    hd95_n = [int(np.sum(~np.isnan(l))) for l in zip(*hd95_metric)]
    asd_n  = [int(np.sum(~np.isnan(l))) for l in zip(*asd_metric)]
    nsd_n  = [int(np.sum(~np.isnan(l))) for l in zip(*nsd_metric)]

    with save_folder.open("a") as file:
        file.write(
            'Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n'.format(
                mask,
                dice_mean[2].item(),
                dice_mean[1].item(),
                dice_mean[0].item()
            )
        )
        file.write(
            'Surface metrics missing scenario = {}: '
            'HD95 (WT={:.2f} n={}, TC={:.2f} n={}, ET={:.2f} n={}), '
            'ASD (WT={:.2f} n={}, TC={:.2f} n={}, ET={:.2f} n={}), '
            'NSD (WT={:.4f} n={}, TC={:.4f} n={}, ET={:.4f} n={})\n'.format(
                mask,
                hd95_mean[2], hd95_n[2], hd95_mean[1], hd95_n[1], hd95_mean[0], hd95_n[0],
                asd_mean[2], asd_n[2], asd_mean[1], asd_n[1], asd_mean[0], asd_n[0],
                nsd_mean[2], nsd_n[2], nsd_mean[1], nsd_n[1], nsd_mean[0], nsd_n[0]
            )
        )

    print(f"\n→ [{mask_name}] ET: {dice_mean[0]:.4f}  TC: {dice_mean[1]:.4f}  WT: {dice_mean[2]:.4f}", flush=True)
    print(f"→ [{mask_name}] HD95  ET: {hd95_mean[0]:.2f} (n={hd95_n[0]})  "
          f"TC: {hd95_mean[1]:.2f} (n={hd95_n[1]})  WT: {hd95_mean[2]:.2f} (n={hd95_n[2]})", flush=True)

    surface_stats = dict(hd_mean=hd_mean, hd95_mean=hd95_mean, asd_mean=asd_mean, nsd_mean=nsd_mean,
                          hd95_n=hd95_n, asd_n=asd_n, nsd_n=nsd_n)

    return losses.avg, np.nanmean(metrics), dice_mean, surface_stats


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
import argparse
import os
import pathlib
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from monai.data import decollate_batch
from tensorboardX import SummaryWriter

from loss import EDiceLoss
from loss.dice import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, \
    count_parameters, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch, setup_seed, save_last
from model.Unet import Unet_missing

from dataset.remind_npy_train_dataset import ReMIND_NpyTrainDataset, ReMIND_NpyValDataset

val_check = [1000]

parser = argparse.ArgumentParser(description='m3ae Finetune on ReMIND')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--patch_shape', default=128, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=1e-5, type=float, dest='lr')
parser.add_argument('--wd', default=0.00001, type=float, dest='weight_decay')
parser.add_argument('--devices', default='0', type=str)
parser.add_argument('--exp_name', default='remind_finetune_local', type=str)
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--debug', action='store_true', default=False)

# pretrain checkpoint — BraTS trained weights
parser.add_argument('--pretrain_checkpoint',
                    default='./runs/m3ae_train/model_1model_best_259.pth.tar',
                    type=str)

# resume — set to None to start fresh, or path to finetuning checkpoint
parser.add_argument('--resume', default='./runs/m3ae_train/model_1model_best_259.pth.tar', type=str,
                    help='path to finetuning checkpoint. None = start fresh from pretrain_checkpoint.')

# ReMIND specific
parser.add_argument('--datapath', default='/home/danish/data/ReMIND_preprocessed_data/miss_mod', type=str)
parser.add_argument('--train_file', default='./remind_train.txt', type=str)
parser.add_argument('--val_csv', default='./test15splits.csv', type=str)
parser.add_argument('--test_csv', default='./test15splits.csv', type=str)


def main(args):
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    setup_seed(args.seed)

    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    print(args)

    # ---- model ----
    model_1 = Unet_missing(
        input_shape=[128, 128, 128], init_channels=16, out_channels=3,
        mdp=3, pre_train=False, patch_shape=args.patch_shape
    )
    model_1 = nn.DataParallel(model_1)

    # step 1 — always load BraTS pretrained weights first
    print(f"Loading pretrained checkpoint: {args.pretrain_checkpoint}")
    ck = torch.load(args.pretrain_checkpoint, map_location=torch.device('cpu'), weights_only=False)
    for key in ['module.unet.up1conv.weight', 'module.unet.up1conv.bias',
                'module.unet.ds_out.0.weight', 'module.unet.ds_out.0.bias',
                'module.unet.ds_out.1.weight', 'module.unet.ds_out.1.bias']:
        if key in ck['state_dict']:
            del ck['state_dict'][key]
    model_1.load_state_dict(ck['state_dict'], strict=False)
    print("Pretrained BraTS checkpoint loaded!")

    model_1 = model_1.cuda()
    model_1.module.raw_input = model_1.module.raw_input.cpu()
    print(f"Total trainable parameters: {count_parameters(model_1)}")

    # ---- loss ----
    criterion     = EDiceLoss().cuda()   # 0.7*Dice + 0.3*BCE
    criterian_val = EDiceLoss_Val().cuda()
    metric        = criterian_val.metric

    # ---- optimizer ----
    optimizer = torch.optim.Adam(
        model_1.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- datasets ----
    print(f"Train file: {args.train_file}")
    print(f"Val CSV:    {args.val_csv}")
    print(f"Test CSV:   {args.test_csv}")

    train_dataset = ReMIND_NpyTrainDataset(
        root=args.datapath, train_file=args.train_file, patch_shape=args.patch_shape
    )
    val_dataset = ReMIND_NpyValDataset(
        root=args.datapath, csv_file=args.val_csv, patch_shape=args.patch_shape
    )
    test_dataset = ReMIND_NpyValDataset(
        root=args.datapath, csv_file=args.test_csv, patch_shape=args.patch_shape
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=args.workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")

    best_1      = 0.0
    start_epoch = 0

    # step 2 — resume from finetuning checkpoint if available
    finetune_last = os.path.join(
        f"./runs/{args.exp_name}/model_1", "model_last.pth.tar"
    )
    if args.resume is not None and os.path.exists(args.resume):
        checkpoint  = torch.load(args.resume, weights_only=False)
        model_1.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        best_1      = checkpoint.get('best_1', 0.0)
        print(f"Resumed finetuning from epoch {checkpoint['epoch']} ({args.resume})")
    elif os.path.exists(finetune_last):
        checkpoint  = torch.load(finetune_last, weights_only=False)
        model_1.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        best_1      = checkpoint.get('best_1', 0.0)
        print(f"Auto-resumed finetuning from epoch {checkpoint['epoch']} ({finetune_last})")
    else:
        print("Starting fresh finetuning from BraTS pretrained weights (epoch 0)")

    # ---- training loop ----
    print("Starting training!")

    for epoch in range(start_epoch, args.epochs):
        ts = time.perf_counter()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time  = AverageMeter('Data', ':6.3f')
        losses_    = AverageMeter('Loss', ':.4e')

        model_1.train()

        batch_per_epoch = len(train_loader)
        progress = ProgressMeter(
            batch_per_epoch,
            [batch_time, data_time, losses_],
            prefix=f"Train Epoch: [{epoch}]"
        )
        end = time.perf_counter()

        et_dice_sum = 0.0
        tc_dice_sum = 0.0
        wt_dice_sum = 0.0
        et_count    = 0
        tc_count    = 0
        wt_count    = 0

        for i, batch in enumerate(train_loader):
            data_time.update(time.perf_counter() - end)

            inputs = batch["image"].cuda()           # (B, 4, 128, 128, 128)
            labels = batch["label"].float().cuda()   # (B, 3, 128, 128, 128) ET/TC/WT binary

            optimizer.zero_grad()

            # pass None — avoids limage coordinate issues
            segs, _, style, content = model_1(inputs, None)
            # segs: (B, 3, 128, 128, 128) raw logits

            loss = criterion(segs, labels)   # 0.7*Dice + 0.3*BCE
            loss.backward()
            optimizer.step()

            if not np.isnan(loss.item()):
                losses_.update(loss.item())
            else:
                print("NaN loss!")

            # per-batch dice monitoring
            # per-batch dice monitoring
            with torch.no_grad():
                pred_bin = (torch.sigmoid(segs) > 0.5).float()
                for b in range(labels.shape[0]):
                    et_d, tc_d, wt_d = 0.0, 0.0, 0.0
                    for c in range(3):
                        p = pred_bin[b, c]
                        t = labels[b, c]
                        inter = (p * t).sum()
                        denom = p.sum() + t.sum()
                        if denom > 0:
                            d = (2 * inter / denom).item()
                            if c == 0: et_dice_sum += d; et_count += 1; et_d = d
                            if c == 1: tc_dice_sum += d; tc_count += 1; tc_d = d
                            if c == 2: wt_dice_sum += d; wt_count += 1; wt_d = d
            
                    # ← print per sample dice
                    case_name = batch["patient_id"][b]
                    print(f"  [{case_name}] ET: {et_d:.4f}  TC: {tc_d:.4f}  WT: {wt_d:.4f}", flush=True)

            t_writer_1.add_scalar("Loss/train", loss.item(),
                                  global_step=batch_per_epoch * epoch + i)
            t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                  global_step=epoch * batch_per_epoch + i)

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()
            progress.display(i)

            if args.debug:
                break

        if scheduler is not None:
            scheduler.step()

        # ---- epoch summary ----
        elapsed  = time.perf_counter() - ts
        et_avg   = et_dice_sum / max(et_count, 1)
        tc_avg   = tc_dice_sum / max(tc_count, 1)
        wt_avg   = wt_dice_sum / max(wt_count, 1)

        print(f"\n{'='*70}")
        print(f"Epoch [{epoch}/{args.epochs-1}]  |  Time: {elapsed:.1f}s  |  Loss: {losses_.avg:.4f}")
        print(f"Train Dice  -->  ET: {et_avg:.4f}  |  TC: {tc_avg:.4f}  |  WT: {wt_avg:.4f}")
        print(f"{'='*70}\n")

        t_writer_1.add_scalar("Dice/train_ET",    et_avg,      epoch)
        t_writer_1.add_scalar("Dice/train_TC",    tc_avg,      epoch)
        t_writer_1.add_scalar("Dice/train_WT",    wt_avg,      epoch)
        t_writer_1.add_scalar("SummaryLoss/train", losses_.avg, epoch)

        # ---- save last checkpoint ----
        save_last(dict(
            epoch=epoch,
            state_dict=model_1.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best_1=best_1
        ), save_folder=args.save_folder_1)

        # ---- validation ----
        if (epoch + 1) in val_check or args.debug:
            print("Validating...")
            val_loss, val_dice, val_metrics = step(
                val_loader, model_1, criterian_val, metric,
                epoch, t_writer_1, save_folder=args.save_folder_1,
                patch_shape=args.patch_shape, debug=args.debug
            )
            print(f"Val  Dice --> ET:{val_metrics[0]:.4f} TC:{val_metrics[1]:.4f} "
                  f"WT:{val_metrics[2]:.4f} Mean:{val_dice:.4f}")

            test_loss, test_dice, test_metrics = step(
                test_loader, model_1, criterian_val, metric,
                epoch, t_writer_1, save_folder=args.save_folder_1,
                patch_shape=args.patch_shape, debug=args.debug
            )
            print(f"Test Dice --> ET:{test_metrics[0]:.4f} TC:{test_metrics[1]:.4f} "
                  f"WT:{test_metrics[2]:.4f} Mean:{test_dice:.4f}")

            t_writer_1.add_scalar("Dice/val_mean",  val_dice,  epoch)
            t_writer_1.add_scalar("Dice/test_mean", test_dice, epoch)

            if val_dice > best_1:
                print(f"Saving best model at epoch {epoch} with DSC {val_dice:.4f}")
                best_1 = val_dice
                save_checkpoint(dict(
                    epoch=epoch,
                    state_dict=model_1.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    best_1=best_1,
                ), save_folder=args.save_folder_1)


def step(data_loader, model, criterion, metric, epoch, writer,
         save_folder=None, patch_shape=128, debug=False):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    losses     = AverageMeter('Loss', ':.4e')

    model.eval()
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"Val Epoch: [{epoch}]"
    )

    end = time.perf_counter()
    metrics = []

    for i, val_data in enumerate(data_loader):
        data_time.update(time.perf_counter() - end)
        model.module.mask_modal = val_data["mask_modal"]

        with torch.no_grad():
            val_inputs = val_data["image"].cuda()
            val_labels = val_data["label"].float().cuda()

            val_outputs   = inference(val_inputs, model, patch_shape=patch_shape)
            val_outputs_1 = [post_trans(j) for j in decollate_batch(val_outputs)]

            segs    = val_outputs
            targets = val_labels
            loss_   = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        if not np.isnan(loss_.item()):
            losses.update(loss_.item())

        writer.add_scalar("Loss/val", loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        progress.display(i)

        if debug:
            break

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar("SummaryLoss/val", losses.avg, epoch)

    dice_metric.reset()
    dice_metric_batch.reset()

    metrics = list(zip(*metrics))
    metrics = [np.nanmean(torch.tensor(d, device="cpu").numpy()) for d in metrics]

    return losses.avg, np.nanmean(metrics), metrics


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
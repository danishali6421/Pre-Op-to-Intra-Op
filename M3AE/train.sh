# #!/bin/bash
# # ---------------------------------------------
# # Train IM-Fuse model (with nohup for persistence)
# # ---------------------------------------------

# # ✅ Activate the correct virtual environment
# source /workspace/IMFuse/IM-Fuse/m3ae/m3ae_venv/bin/activate

# # ✅ Move to the project root
# cd /workspace/IMFuse/IM-Fuse/m3ae

# ✅ Start training in the background with nohup
# nohup python pretrain.py \
# --exp_name m3ae_pretrain \
# --batch_size 2 \
# --mdp 3 \
# --dataset brats23 \
# --mask_ratio 0.875 \
# --lr 0.0003 \
#   > training.log 2>&1 &

# nohup python train.py \
# --exp_name m3ae_train \
# --batch_size 2 \
# --lr 0.0003 \
# --model_type cnnnet \
# --seed 999 \
# --weight_kl 0.1 \
# --feature_level 2 \
# --epochs 300 \
# --mdp 3 \
# --wd 0.0001 \
# --deep_supervised \
# --patch_shape 128 \
#   > training.log 2>&1 &

# # ✅ Print confirmation and PID
# echo "🚀 Training started in background!"
# echo "📜 Logs: tail -f /workspace/IMFuse/IM-Fuse/mmFormer/training.log"
# echo "🔢 Process ID (PID): $!"






#!/bin/bash

source /workspace/IMFuse/IM-Fuse/m3ae/m3ae_venv/bin/activate
cd /workspace/IMFuse/IM-Fuse/m3ae

nohup python train_remind_m3ae.py \
    --datapath /workspace/remind/ReMIND_Intra_operative/np2 \
    --train_file /workspace/remind_train.txt \
    --exp_name remind_finetune \
    --epochs 500 \
    --devices 0 \
    > training.log 2>&1 &

echo "🚀 Training started in background!"
echo "📜 Logs: tail -f /workspace/IMFuse/IM-Fuse/m3ae/training.log"
echo "🔢 Process ID (PID): $!"
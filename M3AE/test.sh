#!/bin/bash

# source /workspace/IMFuse/IM-Fuse/m3ae/m3ae_venv/bin/activate

# cd /workspace/IMFuse/IM-Fuse/m3ae

# python eval_remind_m3ae.py \
#     --datapath /workspace/remind/ReMIND_Intra_operative/np \
#     --exp_name remind_eval \
#     --nifti_aligned_root /workspace/remind/ReMIND_Intra_operative/NIFTI_ALIGNED \
#     --pred_save_dir ./ReMIND_Intra_operative/pred_m3ae \
#     --devices 0 > test.log 2>&1 &

# echo "Started! PID: $!"
# echo "Logs: tail -f /workspace/IMFuse/IM-Fuse/m3ae/eval_remind.log"

source /workspace/IMFuse/IM-Fuse/m3ae/m3ae_venv/bin/activate

cd /workspace/IMFuse/IM-Fuse/m3ae

nohup python test.py \
    > test.log 2>&1 &

echo "Started! PID: $!"
echo "Logs: tail -f /workspace/IMFuse/IM-Fuse/m3ae/eval_remind.log"
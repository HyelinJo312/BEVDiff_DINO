#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS=$1
PORT=${PORT:-28508}

CONFIG="./projects/configs/bevdiffuser/bevformer_tiny_pretrain_dino.py"
# LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RESUME_FROM="../results_pretrain/BEVDiffuser_pretrain-tiny_1denoise_attention-fuser/iter_25000.pth"
RUN_NAME="BEVDiffuser_pretrain-tiny_1denoise_concatfuser"
WORK_DIR="../results_pretrain/${RUN_NAME}"
BEV_CHECKPOINT="./ckpts/bevformer_tiny_epoch_24.pth"
DENOISE_WEIGHT=1.0

export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
torchrun --nproc_per_node=4 --master_port=29505 \
    $(dirname "$0")/train_pretrain.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir $WORK_DIR \
    --denoise_loss_weight $DENOISE_WEIGHT \
    --bev_checkpoint $BEV_CHECKPOINT \
    --report_to 'tensorboard' \
    --resume-from=$RESUME_FROM \
    # --load_from=$LOAD_FROM \

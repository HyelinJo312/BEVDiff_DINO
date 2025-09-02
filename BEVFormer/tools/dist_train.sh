#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=$1
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/layout_tiny.py"
# UNET_CHECKPOINT_DIR="../../results/BEVDiffuser_BEVFormer_tiny/checkpoint-50000"
UNET_CHECKPOINT_DIR="./train/BEVDiffuser_BEVFormer_tiny_original/checkpoint-30000"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RESUME_FROM=None
RUN_NAME="BEVFormer_tiny_with_BEVDiffuser"
WORK_DIR="../results_diffbevformer"

export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
torchrun --nproc_per_node=4 --master_port=29505 \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    --resume_from=$RESUME_FROM \

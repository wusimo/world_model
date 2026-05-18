#!/bin/bash
set -e
REPO=/root/autodl-tmp/world_model
LOG=/root/autodl-tmp/logs/train_phase2_flow_only.log
cd $REPO
echo "[phase2-flow_only] $(date)"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  $REPO/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node=5 --master_port=29503 \
  $REPO/scripts/phase2/train_generative_ddp.py \
  --cfg configs/phase2/paper_scale.yaml \
  --variant flow_only \
  --out /root/autodl-tmp/results/phase2_ddp_flow_only \
  >> $LOG 2>&1

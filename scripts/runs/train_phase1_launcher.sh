#!/bin/bash
set -e
REPO=/root/autodl-tmp/world_model
LOG=/root/autodl-tmp/logs/train_phase1_ddp.log
cd $REPO
echo "[train] launching 5-GPU DDP at $(date) (resume from ckpt 5000)"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  $REPO/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node=5 --master_port=29501 \
  -m src.phase1.train_ddp \
  --cfg configs/phase1/paper_scale.yaml \
  --name vggt_noact \
  --out_root /root/autodl-tmp/results/phase1_ddp \
  --resume \
  >> $LOG 2>&1
echo "[train] exit=$?"

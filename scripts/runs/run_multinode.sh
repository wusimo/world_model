#!/bin/bash
# Multi-node DDP launcher for the H100 cluster.
#
# Usage:
#   bash run_multinode.sh <phase> <node_rank>
#     phase     : phase1 | phase2 | phase3
#     node_rank : 0..3 (h100-1..h100-4)
#
# Set on each of the 4 nodes simultaneously (each with its own node_rank).
# Logs go to /root/<phase>_node<rank>.log.

set -e
PHASE=$1
NODE_RANK=$2
if [ -z "$PHASE" ] || [ -z "$NODE_RANK" ]; then
  echo "Usage: $0 <phase1|phase2|phase3> <node_rank>"
  exit 1
fi

REPO=/data/world_model_workspace/world_model
MASTER_ADDR=172.27.0.4
NPROC_PER_NODE=8
NNODES=4

# H100 cluster NCCL flags — see /etc/profile.d/h100-nccl.sh, also embedded here
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=bond0.1411
export GLOO_SOCKET_IFNAME=bond0.1411
export NCCL_IGNORE_DISABLED_P2P=1
# Use hf-mirror.com for any HF call (no proxy needed)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

cd $REPO

case "$PHASE" in
  phase1)
    MASTER_PORT=29510
    CFG=configs/phase1/h100_paper_scale.yaml
    OUT=/shared/results/phase1_h100
    SCRIPT="-m src.phase1.train_ddp --cfg $CFG --name vggt_noact --out_root $OUT"
    ;;
  phase2)
    MASTER_PORT=29511
    CFG=configs/phase2/h100_paper_scale.yaml
    OUT=/shared/results/phase2_h100
    SCRIPT="scripts/phase2/train_generative_ddp.py --cfg $CFG --variant flow_coupled --out $OUT"
    ;;
  phase3)
    MASTER_PORT=29512
    CFG=configs/phase3/h100_paper_scale.yaml
    OUT=/shared/results/phase3_h100
    SCRIPT="scripts/phase3/train_action_ddp.py --cfg $CFG --out $OUT"
    ;;
  *)
    echo "Unknown phase: $PHASE"
    exit 1
    ;;
esac

LOG_DIR=/tmp/torchrun_${PHASE}
mkdir -p $LOG_DIR

echo "[$(date)] launching $PHASE on node_rank=$NODE_RANK"
echo "  master=${MASTER_ADDR}:${MASTER_PORT}  total ranks=$((NNODES * NPROC_PER_NODE))"
echo "  script: $SCRIPT"

.venv/bin/python -m torch.distributed.run \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --log-dir=$LOG_DIR \
  --redirects=3 --tee=3 \
  $SCRIPT

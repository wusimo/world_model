#!/bin/bash
# Tier-1 Full OXE pull. Pulls ALL shards of the 5 sub-datasets.
# Resumable / idempotent — skips shards already on disk.
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
cd /root/autodl-tmp/world_model
PY=/root/miniconda3/bin/python
echo "[full-oxe] starting at $(date)"
$PY scripts/download_oxe.py \
  --datasets jaco_play taco_play kuka fractal20220817_data bridge \
  --out-dir /root/autodl-tmp/oxe \
  --workers 4 \
  --max-shards-per-dataset 0  # 0 = all shards
echo "[full-oxe] DONE at $(date)"
du -sh /root/autodl-tmp/oxe/*/

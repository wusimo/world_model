#!/bin/bash
# Sharded cache: split paper_scale.json into 5 sub-manifests, run cache.py per GPU.
set -e
REPO=/root/autodl-tmp/world_model
PY=$REPO/.venv/bin/python
MANIFEST=$REPO/data/manifests/paper_scale.json
SHARD_DIR=$REPO/data/manifests/paper_scale_shards
LOG_DIR=/root/autodl-tmp/logs
mkdir -p $SHARD_DIR $LOG_DIR

cd $REPO

# Slice manifest into N shards, write a per-shard config + manifest.
echo "[cache_run] slicing manifest into 5 shards..."
$PY <<'PY'
import json, yaml, os
from pathlib import Path
manifest = json.load(open("data/manifests/paper_scale.json"))
N = 5
shard_dir = Path("data/manifests/paper_scale_shards"); shard_dir.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open("configs/phase1/paper_scale.yaml"))
cfg_dir = Path("configs/phase1/paper_scale_shards"); cfg_dir.mkdir(parents=True, exist_ok=True)
total = len(manifest)
print(f"manifest has {total} clips, splitting into {N}")
for i in range(N):
    a = (total * i) // N
    b = (total * (i+1)) // N
    sub = manifest[a:b]
    sm = shard_dir / f"shard_{i}.json"
    json.dump(sub, open(sm, "w"))
    sc = dict(cfg)
    sc["cache"] = dict(cfg["cache"])
    sc["cache"]["manifest"] = str(sm)
    sc["cache"]["out_dir"] = cfg["cache"]["out_dir"]  # shared out dir, npz names are unique
    sc_path = cfg_dir / f"shard_{i}.yaml"
    yaml.safe_dump(sc, open(sc_path, "w"))
    print(f"  shard {i}: clips {a}..{b} ({b-a}) -> {sm}")
PY

# Launch 5 cache processes, one per GPU
for i in 0 1 2 3 4; do
  LOG=$LOG_DIR/cache_gpu${i}.log
  echo "[cache_run] launching cache on GPU $i -> $LOG"
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
    CUDA_VISIBLE_DEVICES=$i \
    nohup $PY -m src.phase1.cache --cfg configs/phase1/paper_scale_shards/shard_${i}.yaml > $LOG 2>&1 &
  echo "  pid=$!"
done
echo "[cache_run] all 5 launched at $(date)"
sleep 5
echo "--- after 5s ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

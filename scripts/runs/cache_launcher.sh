#!/bin/bash
# Cache stage 2: VGGT token cache → train.
# Waits for manifest, then runs cache.py with the paper_scale config.
set -e
REPO=/root/autodl-tmp/world_model
MANIFEST=$REPO/data/manifests/paper_scale.json
PY=$REPO/.venv/bin/python
LOG=/root/autodl-tmp/logs/cache.log
SMOKE_LOG=/root/autodl-tmp/logs/cache_smoke.log

echo "[cache_launcher] waiting for manifest at $MANIFEST..."
until [ -f "$MANIFEST" ]; do sleep 30; done
echo "[cache_launcher] manifest ready ($(wc -c <"$MANIFEST") bytes, $(python3 -c "import json;print(len(json.load(open('$MANIFEST'))))" 2>/dev/null) clips)"

# Smoke test cache on 5 clips first to measure throughput
echo "[cache_launcher] running cache smoke (limit=5) at $(date)..."
cd $REPO
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  CUDA_VISIBLE_DEVICES=0 \
  $PY -m src.phase1.cache --cfg configs/phase1/paper_scale.yaml --limit 5 > $SMOKE_LOG 2>&1
echo "[cache_launcher] smoke done — last 10 log lines:"
tail -10 $SMOKE_LOG

# Estimate full-cache time from smoke
SMOKE_TIME=$(grep 'elapsed=' $SMOKE_LOG | tail -5 | awk -F'elapsed=' '{print $2}' | awk '{print $1}' | sed 's/s//' | python3 -c "import sys; ts=[float(x) for x in sys.stdin]; print(sum(ts)/len(ts) if ts else 0)")
N_CLIPS=$(python3 -c "import json;print(len(json.load(open('$MANIFEST'))))")
EST=$(python3 -c "print($SMOKE_TIME * $N_CLIPS / 5 / 60)")
echo "[cache_launcher] smoke avg per-clip: ${SMOKE_TIME}s, total clips=$N_CLIPS, single-GPU ETA: ${EST}min"
echo "[cache_launcher] NOTE: full cache requires distributing $N_CLIPS clips across 5 GPUs — see cache_run.sh"


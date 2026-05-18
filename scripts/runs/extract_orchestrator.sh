#!/bin/bash
# Pipeline orchestrator: extracts each dataset as it lands, then builds manifest.
# Designed to run alongside the download script. Uses log markers to detect dataset completion.
set -e
PY=/root/miniconda3/bin/python
DL_LOG=/root/autodl-tmp/logs/oxe_download.log
EXTRACT_LOG=/root/autodl-tmp/logs/extract.log
MANIFEST_LOG=/root/autodl-tmp/logs/manifest.log
REPO=/root/autodl-tmp/world_model

mkdir -p /root/autodl-tmp/oxe_extracted /root/autodl-tmp/logs
cd "$REPO"

DATASETS=(jaco_play taco_play kuka fractal20220817_data bridge)
EXPECTED=(2 11 13 25 37)

echo "[orchestrator] starting at $(date)" >> "$EXTRACT_LOG"

for i in "${!DATASETS[@]}"; do
  D="${DATASETS[$i]}"
  N="${EXPECTED[$i]}"
  echo "[orchestrator] waiting for $D download to finish (need $N shards)..." >> "$EXTRACT_LOG"
  # Wait until that dataset's tar count reaches N AND the next dataset marker shows up
  # (or for the last dataset, "=== ALL DONE ===")
  while true; do
    COUNT=$(ls /root/autodl-tmp/oxe/$D/*.tar 2>/dev/null | wc -l)
    if [ "$COUNT" -ge "$N" ]; then
      # Verify download moved on (next dataset's "===" line appeared) or ALL DONE
      if grep -q "=== ALL DONE ===\|=== ${DATASETS[$((i+1))]:-NONEXISTENT}" "$DL_LOG" 2>/dev/null; then
        break
      fi
    fi
    sleep 30
  done
  echo "[orchestrator] $D download complete, extracting..." >> "$EXTRACT_LOG"
  $PY scripts/extract_oxe_frames.py \
    --in-dir /root/autodl-tmp/oxe \
    --out-dir /root/autodl-tmp/oxe_extracted \
    --datasets "$D" \
    --log /root/autodl-tmp/logs/extract_${D}.log \
    >> "$EXTRACT_LOG" 2>&1
  echo "[orchestrator] $D extraction done at $(date)" >> "$EXTRACT_LOG"
done

echo "[orchestrator] all extractions done, building manifest..." >> "$EXTRACT_LOG"
$PY scripts/build_paper_scale_manifest.py \
  --extracted-dir /root/autodl-tmp/oxe_extracted \
  --repo-root "$REPO" \
  >> "$MANIFEST_LOG" 2>&1
echo "[orchestrator] manifest built at $(date)" >> "$EXTRACT_LOG"
echo "[orchestrator] DONE PIPELINE STAGE 1-3 at $(date)" >> "$EXTRACT_LOG"

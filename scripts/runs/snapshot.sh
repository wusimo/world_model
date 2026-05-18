#!/bin/bash
OUT=/root/autodl-tmp/SNAPSHOT.md
while true; do
  {
    echo "# AutoDL run snapshot"
    echo "_Updated $(date '+%Y-%m-%d %H:%M:%S %Z')_"
    echo
    echo "## Processes"
    pgrep -af "install_torch|download_oxe|extract_oxe|build_paper_scale|cache.py|train|orchestrator|install_deps" | grep -v grep | grep -v snapshot.sh | grep -v 'bash -c' | sed 's/^/    /' || echo "    (none)"
    echo
    echo "## Disk usage"
    echo '```'
    du -sh /root/autodl-tmp/oxe/*/ 2>/dev/null
    du -sh /root/autodl-tmp/oxe_extracted/*/ 2>/dev/null
    du -sh /root/autodl-tmp/cache/*/ 2>/dev/null
    echo '```'
    echo
    echo "## Torch install (last 6 meaningful lines)"
    echo '```'
    grep -vE 'HTTP Request|Looking in|Collecting|Downloading|━+|kB/s|MB/s eta|Stored in|^$' /root/autodl-tmp/logs/torch_install.log 2>/dev/null | tail -6
    echo '```'
    if [ -f /root/autodl-tmp/world_model/.venv/lib/python*/site-packages/torch/__init__.py ]; then
      echo "**torch importable:** $(/root/autodl-tmp/world_model/.venv/bin/python -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1)"
    fi
    echo
    echo "## OXE download (last 12 progress lines)"
    echo '```'
    grep -E 'downloaded |cached |DONE in|=== |ERROR|Traceback' /root/autodl-tmp/logs/oxe_download.log 2>/dev/null | tail -12
    echo '```'
    if [ -f /root/autodl-tmp/logs/extract.log ]; then
      echo
      echo "## Extraction (last 20 lines)"
      echo '```'
      grep -vE 'HTTP|━+|^\s*$' /root/autodl-tmp/logs/extract.log 2>/dev/null | tail -20
      echo '```'
    fi
    if [ -f /root/autodl-tmp/logs/manifest.log ]; then
      echo
      echo "## Manifest (last 10 lines)"
      echo '```'
      tail -10 /root/autodl-tmp/logs/manifest.log
      echo '```'
    fi
    if [ -f /root/autodl-tmp/logs/cache.log ]; then
      echo
      echo "## Token cache (last 20 lines)"
      echo '```'
      grep -vE 'HTTP|━+' /root/autodl-tmp/logs/cache.log 2>/dev/null | tail -20
      echo '```'
    fi
    if [ -f /root/autodl-tmp/logs/train_phase1.log ]; then
      echo
      echo "## Phase 1 training (last 25 lines)"
      echo '```'
      grep -vE 'HTTP|━+' /root/autodl-tmp/logs/train_phase1.log 2>/dev/null | tail -25
      echo '```'
    fi
  } > "$OUT.tmp" 2>&1
  mv "$OUT.tmp" "$OUT"
  sleep 60
done

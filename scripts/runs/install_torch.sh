#!/bin/bash
set -e
# Don't use /etc/network_turbo here — it's been corrupting small PyPI wheels.
# Use Tsinghua TUNA mirror for non-pytorch deps; pytorch.org direct for torch wheels.
cd /root/autodl-tmp/world_model
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --no-cache-dir \
  --timeout 300 --retries 5 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.7.0 torchvision torchaudio
echo "--- torch import test ---"
.venv/bin/python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'ngpu=', torch.cuda.device_count())"

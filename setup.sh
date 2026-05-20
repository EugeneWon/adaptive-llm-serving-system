#!/bin/bash
# Run this INSIDE the ewon_paper1_gpu0 container:
#   docker exec -it ewon_paper1_gpu0 bash
#   chmod +x /workspace/setup.sh && /workspace/setup.sh

set -e

cd /workspace

echo "=== [1/2] Installing required Python packages ==="
pip install \
  transformers \
  accelerate \
  pandas \
  matplotlib \
  tqdm \
  pynvml \
  --quiet

echo "=== [2/2] Verifying setup ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
import transformers
print('transformers:', transformers.__version__)
import pynvml
print('pynvml: OK')
print('torch.compile available:', hasattr(torch, 'compile'))
"

echo ""
echo "=== Setup complete! (Paper 1 - Profiling-Guided Adaptive Optimization) ==="
echo ""
echo "--- Run experiment ---"
echo "python run_experiment.py"

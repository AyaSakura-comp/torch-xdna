#!/bin/bash
source ~/src/Triton-XDNA/npu-env.sh
cd ~/src/torch-xdna
PY=~/src/omnivoice-npu/venv/bin/python
for args in "--policy npu" "--policy auto" "--policy cpu" "--policy npu --fallback eager" "--eager"; do
  tag=$(echo $args | tr -d ' -' )
  echo "=== $args"
  $PY examples/omnivoice_xdna.py $args --wav-dir results/wav --out results/omni_$tag.json 2>&1 | grep -E '^\{|warmup|NPU|npu|cpu |auto|fallback|Error|Traceback' | cut -c1-260
done

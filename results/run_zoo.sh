#!/bin/bash
# model zoo, int8 then bf16, strictly one NPU process at a time
source ~/src/Triton-XDNA/npu-env.sh
cd ~/src/torch-xdna
rm -f results/model_zoo_int8.jsonl results/model_zoo_bf16.jsonl
for p in int8 bf16; do
  ZOO_PRECISION=$p HF_HUB_OFFLINE=1 ~/src/omnivoice-npu/venv/bin/python tests/model_zoo.py
done

#!/bin/bash
export EDGELLM_PLUGIN_PATH=/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
cd /home/nvidia/edgeqwen
/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build/examples/llm/llm_build \
  --onnxDir /home/nvidia/edgeqwen/onnx/llm \
  --engineDir /home/nvidia/edgeqwen/engines/llm \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxBatchSize 1 \
  --debug 2>&1 | tee /home/nvidia/edgeqwen/logs/stage7_llm_build_tmux.log

#!/bin/bash
export EDGELLM_PLUGIN_PATH=/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
cd /home/nvidia/edgeqwen

echo "========================================" | tee -a /home/nvidia/edgeqwen/logs/stage7_llm_build_interactive.log
echo "Starting LLM Engine Build - $(date)" | tee -a /home/nvidia/edgeqwen/logs/stage7_llm_build_interactive.log
echo "========================================" | tee -a /home/nvidia/edgeqwen/logs/stage7_llm_build_interactive.log

/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build/examples/llm/llm_build \
  --onnxDir /home/nvidia/edgeqwen/onnx/llm \
  --engineDir /home/nvidia/edgeqwen/engines/llm \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxBatchSize 1 \
  --debug 2>&1 | tee -a /home/nvidia/edgeqwen/logs/stage7_llm_build_interactive.log

EXIT_CODE=${PIPESTATUS[0]}
echo "========================================" | tee -a /home/nvidia/edgeqwen/logs/stage7_llm_build_interactive.log
echo "Build finished at $(date) with exit code: $EXIT_CODE" | tee -a /home/nvidia/edgeqwen/logs/stage7_llm_build_interactive.log
echo "========================================" | tee -a /home/nvidia/edgeqwen/logs/stage7_llm_build_interactive.log

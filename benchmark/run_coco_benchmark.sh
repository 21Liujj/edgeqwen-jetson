#!/bin/bash
# 方案B1: COCO Caption 500样本评估
# 执行环境: Jetson Orin Nano 8GB

set -e

# 配置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

EDGELLM_ROOT="/home/nvidia/edgeqwen/TensorRT-Edge-LLM"
BENCHMARK_DIR="/home/nvidia/edgeqwen/benchmark/coco"
ENGINE_LLM="/home/nvidia/edgeqwen/engines/llm"
ENGINE_VISUAL="/home/nvidia/edgeqwen/engines/visual"

NUM_SAMPLES=${1:-500}

mkdir -p $BENCHMARK_DIR

echo "========================================"
echo "COCO Caption Benchmark - Jetson LoRA-merged Engine"
echo "样本数: $NUM_SAMPLES"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

cd $EDGELLM_ROOT/examples/accuracy

# Step 1: 准备数据集
echo "[Step 1/3] 准备COCO Caption数据集 (前${NUM_SAMPLES}样本)..."
python3 scripts/prepare_dataset.py \
  --dataset COCO \
  --dataset_name_or_dir lmms-lab/COCO-Caption2017 \
  --output_dir $BENCHMARK_DIR/coco_dataset \
  --batch_size 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 1 \
  --max_generate_length 64 \
  --max_samples $NUM_SAMPLES

echo ""
echo "数据集准备完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 2: 运行TensorRT推理
echo "[Step 2/3] 运行TensorRT推理..."
cd $EDGELLM_ROOT

./build/examples/llm/llm_inference \
  --engineDir $ENGINE_LLM \
  --multimodalEngineDir $ENGINE_VISUAL \
  --inputFile $BENCHMARK_DIR/coco_dataset/dataset.json \
  --outputFile $BENCHMARK_DIR/coco_predictions.json \
  --batchSize 1 \
  --maxGenerateLength 64 2>&1 | tee $BENCHMARK_DIR/inference.log

echo ""
echo "推理完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 3: 计算ROUGE分数
echo "[Step 3/3] 计算ROUGE分数..."
cd $EDGELLM_ROOT/examples/accuracy

python3 scripts/calculate_rouge_score.py \
  --predictions_file $BENCHMARK_DIR/coco_predictions.json \
  --references_file $BENCHMARK_DIR/coco_dataset/dataset.json \
  | tee $BENCHMARK_DIR/rouge_report.txt

echo ""
echo "========================================"
echo "COCO Caption Benchmark完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "结果保存在: $BENCHMARK_DIR/"
echo "========================================"

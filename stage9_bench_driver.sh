#!/bin/bash
# Stage 9: Benchmark Baseline driver
# Workloads:
#   W1 - LLM Prefill, short context   (inputLen=32)
#   W2 - LLM Prefill, long context    (inputLen=512)
#   W3 - Visual Encoder, single image (448x448 -> 196 tokens)
#   W4 - LLM Decode, steady state     (pastKVLen=256)
#   W5 - End-to-end multimodal TTFT   (real image + text, maxGenerateLength=1)
set -u
cd /home/nvidia/edgeqwen
export EDGELLM_PLUGIN_PATH=/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
BENCH_BIN=TensorRT-Edge-LLM/build/examples/llm/llm_bench
INFER_BIN=TensorRT-Edge-LLM/build/examples/llm/llm_inference
OUTDIR=/home/nvidia/edgeqwen/benchmarks
RAWDIR=/tmp/stage9_raw
mkdir -p "$OUTDIR" "$RAWDIR"
N=20
WARMUP=3

run_bench_samples() {
  local name="$1"; shift
  local mode="$1"; shift
  local outfile="$RAWDIR/${name}.txt"
  : > "$outfile"
  for i in $(seq 1 $N); do
    csvdir="$RAWDIR/${name}_csv_$i"
    mkdir -p "$csvdir"
    $BENCH_BIN --engineDir "$@" --mode "$mode" --iterations 1 --warmup $WARMUP --outputDir "$csvdir" > "$RAWDIR/${name}_run${i}.log" 2>&1
    csv=$(ls "$csvdir"/*.csv 2>/dev/null | head -1)
    if [ -n "$csv" ]; then
      tail -1 "$csv" >> "$outfile"
    else
      echo "FAILED_RUN_$i" >> "$outfile"
    fi
    rm -rf "$csvdir"
  done
  echo "done: $name ($(wc -l < "$outfile") samples)"
}

echo "=== W1: LLM Prefill (inputLen=32) ==="
run_bench_samples "w1_prefill_short" prefill engines/llm --inputLen 32

echo "=== W2: LLM Prefill (inputLen=512) ==="
run_bench_samples "w2_prefill_long" prefill engines/llm --inputLen 512

echo "=== W3: Visual Encoder (448x448) ==="
run_bench_samples "w3_visual" visual engines/visual --imageSize 448x448

echo "=== W4: LLM Decode (pastKVLen=256) ==="
run_bench_samples "w4_decode" decode engines/llm --pastKVLen 256

echo "=== W5: End-to-end multimodal TTFT (warm-process, timestamp-diff method) ==="
# W5 methodology: a single warm process handling N repeated identical requests in one
# batch avoids conflating true per-request TTFT with process/engine/tokenizer load time
# (~5-8s), which a fresh-process-per-sample approach would include. max_generate_length=1
# isolates prefill+vision cost. TTFT per request = inter-arrival delta between consecutive
# "Processing vision inputs" log timestamps (first sample includes one-time warmup effects
# and is dropped).
w5_input="/tmp/w5_ttft20.json"
w5_log="$RAWDIR/w5_full.log"
w5_out="$RAWDIR/w5_ttft.txt"
$INFER_BIN --inputFile "$w5_input" --engineDir engines/llm --multimodalEngineDir engines/visual \
  --outputFile /tmp/w5_ttft_out.json > "$w5_log" 2>&1
rm -f /tmp/w5_ttft_out.json
awk -F'[][.]' '/Processing vision inputs/ {
  h=$2; split(h, t, ":"); ts = (t[1]*3600 + t[2]*60 + t[3]) * 1000 + $3
  if (prev != "") print ts - prev
  prev = ts
}' "$w5_log" | tail -n +2 > "$w5_out"
echo "done: w5_ttft ($(wc -l < "$w5_out") samples, first warmup sample dropped)"

echo "=== ALL RAW SAMPLES COLLECTED IN $RAWDIR ==="

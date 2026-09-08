#!/bin/bash
# Stage 10: Runtime Parameter Ablation driver
# ablation_kv.csv:      decode pastKVLen sweep + prefill reuseKVLen on/off sweep
# ablation_visual.csv:  visual encoder imageSize sweep
set -u
cd /home/nvidia/edgeqwen
export EDGELLM_PLUGIN_PATH=/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
BENCH_BIN=TensorRT-Edge-LLM/build/examples/llm/llm_bench
OUTDIR=/home/nvidia/edgeqwen/benchmarks
RAWDIR=/tmp/stage10_raw
mkdir -p "$OUTDIR" "$RAWDIR"
N=10
WARMUP=3

avg_ram_mb() {
  # sample tegrastats for ~1.5s and average the RAM used field
  timeout 1.5 tegrastats --interval 300 2>/dev/null | \
    grep -oP 'RAM \K[0-9]+' | awk '{s+=$1; c++} END {if(c>0) printf "%.0f", s/c; else print "NA"}'
}

run_samples() {
  # $1=label $2=mode, remaining args passed through to llm_bench
  local label="$1"; shift
  local mode="$1"; shift
  local outfile="$RAWDIR/${label}.txt"
  : > "$outfile"
  for i in $(seq 1 $N); do
    csvdir="$RAWDIR/${label}_csv_$i"
    mkdir -p "$csvdir"
    $BENCH_BIN --mode "$mode" --iterations 1 --warmup $WARMUP --outputDir "$csvdir" "$@" > "$RAWDIR/${label}_run${i}.log" 2>&1
    csv=$(ls "$csvdir"/*.csv 2>/dev/null | head -1)
    if [ -n "$csv" ]; then
      tail -1 "$csv" >> "$outfile"
    else
      echo "FAILED_RUN_$i" >> "$outfile"
    fi
    rm -rf "$csvdir"
  done
  ram=$(avg_ram_mb)
  echo "done: $label ($(wc -l < "$outfile") samples, ram_avg_mb=$ram)"
  echo "$ram" > "$RAWDIR/${label}.ram"
}

echo "=== ablation_kv: decode pastKVLen sweep ==="
for kv in 64 256 1024 2048 3840; do
  echo "--- decode pastKVLen=$kv ---"
  run_samples "kv_decode_${kv}" decode --engineDir engines/llm --pastKVLen "$kv"
done

echo "=== ablation_kv: prefill reuseKVLen sweep (inputLen=512, reuse off/half/near-full) ==="
for reuse in 0 256 480; do
  echo "--- prefill inputLen=512 reuseKVLen=$reuse ---"
  run_samples "kv_reuse_${reuse}" prefill --engineDir engines/llm --inputLen 512 --reuseKVLen "$reuse"
done

echo "=== ablation_visual: imageSize sweep ==="
for size in 224x224 448x448 640x640 896x448; do
  tag=$(echo "$size" | tr 'x' '_')
  echo "--- visual imageSize=$size ---"
  run_samples "visual_${tag}" visual --engineDir engines/visual --imageSize "$size"
done

echo "=== ALL ABLATION SAMPLES COLLECTED IN $RAWDIR ==="

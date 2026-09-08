#!/bin/bash
# Stage 12: Stability & Thermal Analysis driver
# Sustained multimodal E2E inference under continuous tegrastats monitoring.
set -u
cd /home/nvidia/edgeqwen
export EDGELLM_PLUGIN_PATH=/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
BIN=TensorRT-Edge-LLM/build/examples/llm/llm_inference
RAWDIR=/tmp/stage12_raw
mkdir -p "$RAWDIR"

BATCHES=25
REQS_PER_BATCH=20
TOTAL=$((BATCHES * REQS_PER_BATCH))

echo "=== Stage 12: $TOTAL total multimodal E2E requests, $BATCHES batches x $REQS_PER_BATCH ==="

# Build the 20-request batch input once (reused every batch call)
python3 -c "
import json
with open('test_input.json') as f:
    base = json.load(f)
req = base['requests'][0]
base['requests'] = [req] * $REQS_PER_BATCH
with open('$RAWDIR/batch_input.json', 'w') as f:
    json.dump(base, f)
"

# Start continuous tegrastats logging for the full run duration
tegrastats --interval 1000 --logfile "$RAWDIR/tegrastats.log" &
TEGRA_PID=$!
echo "tegrastats PID=$TEGRA_PID logging to $RAWDIR/tegrastats.log"
sleep 2

echo "batch,start_epoch,end_epoch,wall_s" > "$RAWDIR/batch_timestamps.csv"

RUN_START=$(date +%s)
for b in $(seq 1 $BATCHES); do
  t0=$(date +%s)
  "$BIN" --engineDir engines/llm --multimodalEngineDir engines/visual \
    --inputFile "$RAWDIR/batch_input.json" \
    --outputFile "$RAWDIR/out_batch_${b}.json" \
    --dumpProfile --profileOutputFile "$RAWDIR/profile_batch_${b}.json" \
    --warmup 1 > "$RAWDIR/log_batch_${b}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "${b},${t0},${t1},$((t1 - t0))" >> "$RAWDIR/batch_timestamps.csv"
  if [ $rc -ne 0 ]; then
    echo "BATCH_${b}_FAILED rc=$rc" >> "$RAWDIR/failures.log"
    echo "!!! batch $b FAILED (rc=$rc) — see log_batch_${b}.log"
  else
    echo "batch $b/${BATCHES} done in $((t1 - t0))s"
  fi
done
RUN_END=$(date +%s)

kill $TEGRA_PID 2>/dev/null
sleep 1
tegrastats --stop 2>/dev/null

echo "=== Stage 12 run complete: $((RUN_END - RUN_START))s total, $TOTAL requests across $BATCHES batches ==="
echo "$((RUN_END - RUN_START))" > "$RAWDIR/total_wall_seconds.txt"

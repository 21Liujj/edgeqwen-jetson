#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${PREFLIGHT_OUT:-$ROOT/closure_runs/preflight_$(date +%Y%m%d_%H%M%S)}
PROBE=0
for arg in "$@"; do case "$arg" in --probe) PROBE=1;; --out=*) OUT=${arg#--out=};; --help|-h) printf "Usage: %s [--probe] [--out=DIR]\n" "$0"; exit 0;; *) printf "ERROR: unknown option %s\n" "$arg" >&2; exit 10;; esac; done
mkdir -p "$OUT"
exec > >(tee "$OUT/preflight.log") 2>&1
fail=0
check_file() { local p=$1; if [[ ! -e "$p" ]]; then printf "ERROR missing: %s\n" "$p"; fail=1; else printf "OK file: %s (%s bytes)\n" "$p" "$(stat -c %s "$p")"; fi; }
for p in "$ROOT/engines/llm/llm.engine" "$ROOT/engines/visual/visual.engine" "$ROOT/engines/llm/config.json" "$ROOT/engines/visual/config.json" "$ROOT/benchmark/evaluate_results.py" "$ROOT/TensorRT-Edge-LLM/build/examples/llm/llm_inference" "$ROOT/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"; do check_file "$p"; done
PLUGIN="$ROOT/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
BIN="$ROOT/TensorRT-Edge-LLM/build/examples/llm/llm_inference"
if [[ -e "$PLUGIN" ]]; then RESOLVED=$(readlink -f "$PLUGIN"); printf "plugin_link=%s\nplugin_resolved=%s\n" "$PLUGIN" "$RESOLVED"; file "$RESOLVED"; sha256sum "$PLUGIN" "$RESOLVED"; ldd "$RESOLVED" > "$OUT/plugin_ldd.txt" 2>&1 || fail=1; grep -E "not found|libnvinfer|libcudart|libcuda" "$OUT/plugin_ldd.txt" || true; fi
python3 -m py_compile "$ROOT/benchmark/evaluate_results.py" || fail=1
status=FAIL; [[ "$fail" == 0 ]] && status=PASS
printf "{\"schema_version\":1,\"status\":\"%s\",\"probe_requested\":%s,\"output_dir\":\"%s\"}\n" "$status" "$PROBE" "$OUT" > "$OUT/result.json"
[[ "$fail" == 0 ]] && exit 0 || exit 13

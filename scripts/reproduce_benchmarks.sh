#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=${REPRODUCE_OUT:-$ROOT/closure_runs/reproduce_20260831}
mkdir -p "$RUN_DIR"
"$ROOT/scripts/preflight_edgeqwen.sh" --out="$RUN_DIR/preflight" || exit 13
python3 "$ROOT/benchmark/evaluate_results.py" "$ROOT/benchmark/textvqa/textvqa_dataset/dataset.json" "$ROOT/benchmark/textvqa/textvqa_predictions.json" --dataset-type textvqa --json-output "$RUN_DIR/textvqa.json" || exit 20
python3 "$ROOT/benchmark/evaluate_results.py" "$ROOT/benchmark/scienceqa/scienceqa_dataset/dataset.json" "$ROOT/benchmark/scienceqa/scienceqa_predictions.json" --dataset-type scienceqa --json-output "$RUN_DIR/scienceqa.json" || exit 21
python3 "$ROOT/benchmark/evaluate_results.py" "$ROOT/benchmark/mmstar/mmstar_dataset/mmstar_dataset.json" "$ROOT/benchmark/mmstar/mmstar_predictions.json" --dataset-type mmstar --json-output "$RUN_DIR/mmstar.json" || exit 22
python3 - "$RUN_DIR" <<"PY"
import json,pathlib,sys
run=pathlib.Path(sys.argv[1])
d={"schema_version":1,"mode":"evaluate_existing_predictions","results":{k:json.loads((run/(k+".json")).read_text()) for k in ("textvqa","scienceqa","mmstar")}}
(run/"summary.json").write_text(json.dumps(d,indent=2,ensure_ascii=True)+"\n")
PY

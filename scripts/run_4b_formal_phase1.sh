#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/p1_reasoning_switch_4b_seed1.yaml
OUT=results/p1_qwen3_4b_seed1

echo "=== [1/5] full behavior (10000 rows) ==="
python -m experiments.p1_reasoning_switch.run_behavior --config "$CFG"

echo "=== [1b/5] group-level paired stats ==="
python scripts/behavior_paired_stats.py --behavior "$OUT/behavior.parquet"

echo "=== [2/5] cache activations (500 per condition) ==="
python -m experiments.p1_reasoning_switch.cache_activations --config "$CFG" \
  --conditions R-only S-only R+S-consistent R+S-conflict S-matched \
  --max_per_condition 500 --batch_size 8

echo "=== [3/5] probes: matched_s / conflict_state / rs_state ==="
for c in matched_s conflict_state rs_state; do
  python -m experiments.p1_reasoning_switch.run_probe \
    --activation_dir "$OUT/activations" --contrast "$c" \
    --out "$OUT/probe_$c" || true
done

echo "=== [4/5] layer x position patching, 128 pairs ==="
python -m experiments.p1_reasoning_switch.run_layer_patching --config "$CFG" \
  --layers all --kind resid \
  --positions question_end facts_end cue_end options_end last \
  --max_pairs 128

echo "=== [5/5] donor controls on top-K layers (run after inspecting [4]) ==="
echo "skipped: choose candidate layers from $OUT/patching/patching_resid_summary.csv first"

echo "phase-1 pipeline finished -> $OUT"

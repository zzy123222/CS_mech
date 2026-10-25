#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PY=${PY:-python}
CFG=configs/p1_reasoning_switch_4b_seed1.yaml
OUT=results/p1_qwen3_4b_seed1
DIR=$OUT/probe_matched_s

echo "=== [D1] head wide-screen at L24 (all heads) ==="
$PY -m experiments.p1_reasoning_switch.run_head_ablation --config $CFG \
  --layer 24 --heads $(seq 0 31) --max_samples 96 || echo "D1 FAILED"

echo "=== [D2] random-module null for candidate L24:attn ==="
$PY -m experiments.p1_reasoning_switch.run_random_module_control --config $CFG \
  --candidate L24:attn --n_random 20 --layer_band 6 --max_samples 48 || echo "D2 FAILED"

echo "=== [D3] necessity: mean-ablation of L24:attn + extended generic battery ==="
$PY -m experiments.p1_reasoning_switch.run_interventions --config $CFG \
  --mode ablation --module L24:attn --max_samples 128 --generic_battery extended || echo "D3 FAILED"

echo "=== [D4] rescue: ablate L24:attn, re-inject matched_s direction at L26 ==="
$PY -m experiments.p1_reasoning_switch.run_interventions --config $CFG \
  --mode rescue --module L24:attn --rescue_layer 26 \
  --vector $DIR/matched_s_direction_layer_026.npy \
  --alphas -1 1 2 4 --max_samples 96 || echo "D4 FAILED"

echo "=== [D5] projection ablation of matched_s direction (L26), conflict condition ==="
$PY -m experiments.p1_reasoning_switch.run_interventions --config $CFG \
  --mode projection --module L26:resid --vector $DIR/matched_s_direction_layer_026.npy \
  --alphas 0.5 1.0 1.5 --max_samples 96 --generic_battery extended \
  --target_condition R+S-conflict || echo "D5 FAILED"

echo "=== [D6] projection ablation, R-only condition (necessity for rule usage) ==="
$PY -m experiments.p1_reasoning_switch.run_interventions --config $CFG \
  --mode projection --module L26:resid --vector $DIR/matched_s_direction_layer_026.npy \
  --alphas 0.5 1.0 1.5 --max_samples 96 --target_condition R-only || echo "D6 FAILED"

echo "=== [D7] steering dose-response on S-matched (direction sufficiency) ==="
$PY -m experiments.p1_reasoning_switch.run_interventions --config $CFG \
  --mode steering --module L26:resid --vector $DIR/matched_s_direction_layer_026.npy \
  --alphas -3 -2 -1 1 2 3 --max_samples 64 --target_condition S-matched || echo "D7 FAILED"

echo "PHASE-D ALL DONE"

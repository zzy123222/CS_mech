#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.datasets.generate_graph_reasoning \
  --n_groups "${P1_GROUPS:-2000}" --seed 1 \
  --out data/generated/p1_graph.parquet

python -m src.datasets.generate_error_monitor \
  --n_samples "${P2_SAMPLES:-3000}" --seed 11 --n_steps 4 \
  --out data/generated/p2_errors.parquet

python -m src.datasets.generate_task_battery \
  --n_groups "${P3_GROUPS:-1000}" --seed 31 \
  --out data/generated/p3_task_battery.parquet

python -m src.datasets.generate_hallucination \
  --n_groups "${P5_GROUPS:-2000}" --seed 71 \
  --out data/generated/p5_hallucination.parquet

echo "All synthetic datasets generated."

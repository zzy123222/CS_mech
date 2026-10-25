#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PY=${PY:-python}
CFG=${CFG:-configs/p1_reasoning_switch_4b_seed1.yaml}
OUT=${OUT:-results/p1_qwen3_4b_seed1}
TOPK=${TOPK:-4}

LAYERS=${LAYERS:-$($PY - "$OUT" "$TOPK" <<'PY'
import sys, pandas as pd
out, k = sys.argv[1], int(sys.argv[2])
d = pd.read_csv(f"{out}/patching/patching_resid_summary.csv")
d = d[(d.position == "last") & (d.n >= 100)]
top = d.sort_values("mean_recovery", ascending=False).head(k)
print(" ".join(str(int(l)) for l in top.layer))
PY
)}
echo "candidate layers: $LAYERS"

echo "=== donor controls (wrong / same-answer) on candidate layers ==="
$PY -m experiments.p1_reasoning_switch.run_patching_controls \
  --config "$CFG" --layers $LAYERS --kind resid --positions last --max_pairs 64

for KIND in attn mlp; do
  echo "=== component decomposition: $KIND ==="
  $PY -m experiments.p1_reasoning_switch.run_layer_patching \
    --config "$CFG" --layers $LAYERS --kind $KIND \
    --positions last --max_pairs 96
done

echo "phase-C finished"

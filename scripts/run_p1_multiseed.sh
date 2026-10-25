#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(${SEEDS:-1 2 3})
MODEL_CONFIG=${MODEL_CONFIG:-configs/model_qwen3_0.6b.yaml}
DATASET=${DATASET:-data/generated/p1_graph.parquet}
MAX_PAIRS=${MAX_PAIRS:-64}

for seed in "${SEEDS[@]}"; do
  CFG=/tmp/p1_seed${seed}.yaml
  python - "$seed" <<'PY'
import sys, yaml
from pathlib import Path
seed=sys.argv[1]
cfg=yaml.safe_load(Path('configs/p1_reasoning_switch.yaml').read_text())
cfg['model_config']=__import__('os').environ.get('MODEL_CONFIG', cfg['model_config'])
cfg['dataset']=__import__('os').environ.get('DATASET', cfg['dataset'])
cfg['output_dir']=f'results/p1_reasoning_switch_seed{seed}'
cfg['seed']=int(seed)
cfg['patching']['max_pairs']=int(__import__('os').environ.get('MAX_PAIRS','64'))
Path(f'/tmp/p1_seed{seed}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
PY
  export MODEL_CONFIG DATASET MAX_PAIRS
  python -m experiments.p1_reasoning_switch.run_behavior --config "$CFG" --seed "$seed"
  python -m experiments.p1_reasoning_switch.run_layer_patching --config "$CFG" --seed "$seed" --layers all
done

python - <<'PY'
import json, glob
import pandas as pd
rows=[]
for f in sorted(glob.glob("results/p1_reasoning_switch_seed*/rri.json")):
    seed=f.split("_seed")[1].split("/")[0]
    rows.append({"seed":int(seed),"RRI":json.load(open(f))["RRI"]})
if rows:
    df=pd.DataFrame(rows); df.to_csv("results/p1_multiseed_rri.csv",index=False)
    print(df.to_string(index=False))
PY
echo "multiseed runs finished"

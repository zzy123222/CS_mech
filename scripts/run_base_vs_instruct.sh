#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

INSTRUCT_CFG=${INSTRUCT_CFG:-configs/model_qwen3_4b.yaml}
BASE_CFG=${BASE_CFG:-configs/model_qwen3_4b_base.yaml}
DATASET=${DATASET:-data/generated/p1_graph.parquet}
MAX_ROWS=${MAX_ROWS:-800}

run_one () {
  local model_cfg=$1 tag=$2
  local cfg=/tmp/p1_${tag}.yaml
  MODEL_CFG="$model_cfg" python - "$tag" <<'PY'
import os, sys, yaml
from pathlib import Path
tag=sys.argv[1]
cfg=yaml.safe_load(Path('configs/p1_reasoning_switch.yaml').read_text())
cfg['model_config']=os.environ['MODEL_CFG']
cfg['dataset']=os.environ.get('DATASET', cfg['dataset'])
cfg['output_dir']=f'results/p1_base_vs_instruct/{tag}'
Path(f'/tmp/p1_{tag}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
PY
  DATASET="$DATASET" python -m experiments.p1_reasoning_switch.run_behavior --config "$cfg" --max_rows "$MAX_ROWS"
}

run_one "$INSTRUCT_CFG" instruct
run_one "$BASE_CFG" base

python - <<'PY'
import json
import pandas as pd
rows=[]
for tag in ["instruct","base"]:
    try:
        rri=json.load(open(f"results/p1_base_vs_instruct/{tag}/rri.json"))["RRI"]
        sm=pd.read_csv(f"results/p1_base_vs_instruct/{tag}/behavior_summary.csv")
        rows.append({"model":tag,"RRI":rri,
                     **{f"margin_{r.condition}":r.mean_margin for r in sm.itertuples()}})
    except FileNotFoundError:
        pass
if rows:
    df=pd.DataFrame(rows); df.to_csv("results/p1_base_vs_instruct/summary.csv",index=False)
    print(df.to_string(index=False))
PY

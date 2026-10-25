# CS_mech

Code, data and result tables for the paper

**Causal profiling of a large language model: a format-gated key/value-group circuit for evidence arbitration and its reorganization by post-training**

The repository contains the task generators, causal-intervention library, experiment runners, controlled post-training pipeline, the analysis scripts that produce every number in the manuscript, and the script that draws all figures from the stored results.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Library: task generators (`datasets/`), model loading and answer scoring (`models/`), hooks and interventions (ablation, patching, head-level and steering; `hooks/`, `interventions/`), probes, statistics |
| `configs/` | Model and experiment configurations (model × prompt format, tasks, seeds) |
| `experiments/p1_reasoning_switch/` | Evidence-arbitration experiments: behavior, layer/component patching, head ablation and patching, model × format, key/value groups |
| `experiments/p2_error_monitor/` | Error-conditioned verification: time courses, ablation, patching, steering, specificity panel |
| `experiments/p3_control_hubs/` | Module-dependency map, 72-module null, GQA-aware null, transfer re-test, epistasis |
| `experiments/p4_development/` | Checkpoint scans |
| `posttraining/` | Controlled LoRA post-training: data arms (`data/`), training (`train/`), checkpoint assays and head tracking (`assays/`) |
| `analysis/p0/` | Condition-matched validation experiments and their analyses (Sections 4.2–4.7) |
| `analysis/make_figures.py` | Draws Figs. 1–9 and the graphical abstract from `results/` |
| `data/generated/` | Generated datasets used in the paper (graph reasoning seeds 1–3, verification trajectories, task battery incl. retrieval and planning) |
| `results/` | Result tables needed to reproduce all reported numbers and figures |
| `tests/` | Unit tests (no model download required) |

## Installation

```bash
conda env create -f environment.yml
conda activate llm-mech
```

or `pip install -r requirements.txt` in a Python ≥ 3.10 environment with a CUDA build of PyTorch. Models are loaded from the Hugging Face Hub (`Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`, `mistralai/Mistral-7B-Instruct-v0.3`). All commands are run from the repository root.

```bash
make test
make smoke
```

## Reproducing the figures and numbers from stored results

No GPU is needed for this step.

```bash
python analysis/p0/p0_1_analyze.py
python analysis/p0/p0_2_analyze.py
python analysis/p0/p0_3_summary.py
python analysis/p0/p0_6_7_summary.py
python analysis/make_figures.py
```

The analysis scripts write `results/p0/*_results.json` and `*_summary.json`; `make_figures.py` writes `figures/Fig1–Fig9` and `figures/Graphical_abstract` as PDF, PNG and TIFF.

## Re-running the experiments

| Manuscript section | Command |
|---|---|
| Data generation | `bash scripts/generate_all_data.sh` |
| 4.1 Behavior and layer patching | `python -m experiments.p1_reasoning_switch.run_behavior --config configs/p1_reasoning_switch_4b_seed1.yaml`, `python -m experiments.p1_reasoning_switch.run_layer_patching --config ...` |
| 4.1, 4.3 72-module null and direction transfer | `python analysis/p0/p0_6_7.py --part both` |
| 4.2, 4.4 Head sets, key/value groups, model × prompt format | `bash analysis/p0/run_p0_2_all.sh` |
| 4.2 Partial-batch patching check (Appendix A.1) | `python analysis/p0/p0_5b_bugrepro.py` |
| 4.5 Verification time courses (five models) | `python -m experiments.p2_error_monitor.run_timecourse --config configs/p2_error_monitor.yaml` (and `p2_06b`, `p2_base`, `p2_8b`, `p2_mistral`) |
| 4.6 Condition-matched interventions (float32) | `python analysis/p0/p0_1_specificity.py` |
| 4.7 Post-training data | `python posttraining/data/make_sft_arms.py --seed 1` (seeds 1–3) |
| 4.7 LoRA training | `python posttraining/train/train_lora.py --arm correction_sft --data posttraining/data/sft/correction_sft_seed1.jsonl --output_dir posttraining/results/formal/correction_sft_seed1 --seed 1` |
| 4.7 Checkpoint re-scan | `python analysis/p0/p0_3_rescan.py --include_instruct` |
| 4.7 Head tracking | `python posttraining/assays/track_shared_heads.py` |

Arbitration and post-training experiments use bfloat16; the verification interventions in Section 4.6 use float32.

## Notes

* Every margin is computed from one batched forward pass with one row per answer option; all interventions modify every row of the batch.
* In Qwen3-4B, query head *h* reads key/value head ⌊*h*/4⌋; query heads 28–31 form key/value group 7.
* Model checkpoints produced by the LoRA runs are not included because of their size; they can be regenerated with the commands above.

## License

This project is released under the [MIT License](LICENSE).

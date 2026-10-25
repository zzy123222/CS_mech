.PHONY: test smoke data-p1 data-p2 data-p3 data-p5 compile

compile:
	python -m compileall -q src experiments tests

test:
	python -m pytest -q

smoke:
	python scripts/smoke_no_model.py

data-p1:
	python -m src.datasets.generate_graph_reasoning --n_groups 2000 --seed 1 --out data/generated/p1_graph.parquet

data-p2:
	python -m src.datasets.generate_error_monitor --n_samples 2000 --seed 11 --n_steps 4 --out data/generated/p2_errors.parquet

data-p3:
	python -m src.datasets.generate_task_battery --n_groups 500 --seed 31 --out data/generated/p3_task_battery.parquet

data-p5:
	python -m src.datasets.generate_hallucination --n_groups 1000 --seed 71 --out data/generated/p5_hallucination.parquet

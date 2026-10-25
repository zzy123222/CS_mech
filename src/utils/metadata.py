from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _safe_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def collect_run_metadata(model_id: str | None = None, revision: str | None = None,
                         seed: int | None = None, config_file: str | None = None,
                         dataset_hash: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    md = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _safe_git_commit(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_id": model_id,
        "model_revision": revision,
        "seed": seed,
        "config_file": config_file,
        "dataset_hash": dataset_hash,
    }
    if extra:
        md.update(extra)
    return md


def write_run_metadata(path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect_run_metadata(**kwargs), f, ensure_ascii=False, indent=2)

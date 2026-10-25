from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "torch", "transformers", "accelerate", "numpy", "pandas", "scipy",
    "sklearn", "statsmodels", "yaml", "tqdm", "networkx"
]


def version_of(name: str):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception as e:
        return f"MISSING ({e})"


def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    for p in PACKAGES:
        print(f"{p:14s}: {version_of(p)}")
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        print("torch CUDA:", torch.version.cuda)
        if torch.cuda.is_available():
            print("GPU count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {prop.name}; VRAM={prop.total_memory/1024**3:.1f} GiB")
    except Exception:
        pass

    try:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version("4.51.0"):
            print("WARNING: transformers < 4.51.0; Qwen3 may fail to load.")
    except Exception:
        pass


if __name__ == "__main__":
    main()

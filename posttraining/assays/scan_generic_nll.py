from __future__ import annotations
import sys, json, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import torch

from src.metrics.generic_health import build_extended_battery, generic_nll
from src.utils.seed import set_seed


class Bundle:
    def __init__(self, model, tok):
        self.model = model
        self.tokenizer = tok
        self.chat = True

    def prepare_prompt(self, prompt, enable_thinking=None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def load_merged(base_id, adapter_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", type=int, default=[0, 200, 500, 1000])
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--out", default="posttraining/results/formal/generic_nll.csv")
    args = ap.parse_args()

    set_seed(1)
    mcfg = {"enable_thinking": False}
    pairs = build_extended_battery(n=120, seed=0)

    rows = []
    arms = [("general_sft_seed1", "A: General"),
            ("correction_sft_seed1", "B: Correction"),
            ("sham_correction_seed1", "C: Sham")]

    for model_id, tag in [(args.base_model, "Base (step0 equivalent)"),
                          ("Qwen/Qwen3-4B", "Instruct reference")]:
        model, tok = load_merged(model_id, None)
        b = Bundle(model, tok)
        nll = generic_nll(b, mcfg, pairs=pairs)
        rows.append({"arm": tag, "step": None, "generic_nll": nll})
        print(f"{tag}: NLL={nll:.4f}")
        del model, b
        gc.collect(); torch.cuda.empty_cache()

    for arm_dir, label in arms:
        for step in args.steps:
            ckpt = ROOT / "posttraining/results/formal" / arm_dir / f"step_{step:04d}"
            if not ckpt.exists():
                print(f"[skip] {ckpt}")
                continue
            model, tok = load_merged(args.base_model, str(ckpt))
            b = Bundle(model, tok)
            nll = generic_nll(b, mcfg, pairs=pairs)
            rows.append({"arm": label, "step": step, "generic_nll": nll})
            print(f"{label} step {step}: NLL={nll:.4f}", flush=True)
            del model, b
            gc.collect(); torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, json, hashlib, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024):
        self.data = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                msgs = item["messages"]
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
                enc = tokenizer(text, truncation=True, max_length=max_len,
                                add_special_tokens=False, return_tensors="pt")
                self.data.append({
                    "input_ids": enc.input_ids[0],
                    "labels": enc.input_ids[0].clone(),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["general_sft", "correction_sft", "sham_correction"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--checkpoint_steps", nargs="*", type=int,
                    default=[10, 20, 50, 100, 200, 300, 400, 500, 650, 800, 1000])
    ap.add_argument("--save_opt_state", choices=["all", "sparse", "none"], default="all",
                    help="all: optimizer/scheduler/RNG at every checkpoint (~530MB each); "
                         "sparse: only at resume points 500/1000; none: adapter only (pilot mode)")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from src.utils.seed import set_seed
    set_seed(args.seed)

    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

    print(f"Loading tokenizer: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model: {args.base_model}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto"
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading data: {args.data}")
    dataset = SFTDataset(args.data, tok)
    print(f"  {len(dataset)} samples")

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_sha = hashlib.sha256(open(args.data, "rb").read()).hexdigest()[:16]

    def save_training_state(ckpt_dir, step):
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "step": step,
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, ckpt_dir / "training_state.pt")

    def write_manifest(ckpt_dir, step):
        with open(ckpt_dir / "manifest.json", "w") as f:
            json.dump({
                "base_model": args.base_model,
                "arm": args.arm,
                "seed": args.seed,
                "optimizer_step": step,
                "normalized_progress": round(step / args.max_steps, 3),
                "data_sha256": data_sha,
                "code_sha256": code_sha,
                "train_loss": losses[-1] if losses else None,
                "hyperparams": {
                    "max_steps": args.max_steps,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_r * 2,
                    "lora_dropout": 0.05,
                    "scheduler": "CosineAnnealingLR",
                    "dtype": args.dtype,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"],
                },
            }, f, indent=2)

    losses = []

    step0_dir = out_dir / "step_0000"
    step0_dir.mkdir(exist_ok=True)
    model.save_pretrained(step0_dir)
    write_manifest(step0_dir, 0)
    if args.save_opt_state != "none":
        save_training_state(step0_dir, 0)
    print(f"  Saved checkpoint step=0")

    sparse_points = {500, args.max_steps}
    checkpoint_set = set(args.checkpoint_steps)
    device = next(model.parameters()).device
    step = 0

    print(f"\nTraining {args.arm} for {args.max_steps} steps...")
    print(f"  effective batch = {args.batch_size * args.grad_accum}, "
          f"epochs at max_steps = {args.max_steps * args.batch_size * args.grad_accum / len(dataset):.1f}")
    while step < args.max_steps:
        indices = torch.randperm(len(dataset)).tolist()

        for start in range(0, len(indices), args.batch_size * args.grad_accum):
            if step >= args.max_steps:
                break

            optimizer.zero_grad()
            batch_loss = 0.0
            n_micro = 0

            for micro_start in range(start, min(start + args.batch_size * args.grad_accum, len(indices)), args.batch_size):
                batch_idx = indices[micro_start:micro_start + args.batch_size]
                batch = [dataset[i] for i in batch_idx]
                inputs = collate(batch, tok.pad_token_id)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.cuda.amp.autocast(dtype=dtype):
                    out = model(**inputs)
                    loss = out.loss / args.grad_accum

                loss.backward()
                batch_loss += loss.item()
                n_micro += 1

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(batch_loss / max(n_micro, 1))

            if step % 50 == 0:
                recent = losses[-50:]
                print(f"  Step {step}/{args.max_steps}: loss={sum(recent)/len(recent):.4f}", flush=True)

            if step in checkpoint_set:
                ckpt_dir = out_dir / f"step_{step:04d}"
                ckpt_dir.mkdir(exist_ok=True)
                model.save_pretrained(ckpt_dir)
                write_manifest(ckpt_dir, step)
                if args.save_opt_state == "all" or (args.save_opt_state == "sparse" and step in sparse_points):
                    save_training_state(ckpt_dir, step)
                print(f"  ✓ Saved checkpoint step={step}", flush=True)

    print(f"\nTraining complete. Final loss: {losses[-1]:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()

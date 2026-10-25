from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    model_id: str
    revision: str | None
    chat: bool

    def prepare_prompt(self, prompt: str, enable_thinking: bool | None = None) -> str:
        if not self.chat:
            return prompt
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], **kwargs
        )


def load_causal_lm(model_id: str, revision: str | None = None,
                   dtype: str = "bfloat16", device_map: str | dict | None = "auto",
                   trust_remote_code: bool = False, chat: bool = True,
                   attn_implementation: str | None = None) -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_obj = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(str(dtype).lower(), dtype)

    tok = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kwargs = {
        "revision": revision,
        "torch_dtype": dtype_obj,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return ModelBundle(model=model, tokenizer=tok, model_id=model_id,
                       revision=revision, chat=chat)

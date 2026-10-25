from __future__ import annotations

import torch
from tqdm import tqdm

from src.utils.module_access import get_decoder_layers, get_attention_output_projection, get_num_attention_heads, get_input_device


def _reshape_heads(x: torch.Tensor, n_heads: int):
    if x.shape[-1] % n_heads != 0:
        raise ValueError(f"Projection input dim {x.shape[-1]} not divisible by n_heads={n_heads}")
    return x.view(*x.shape[:-1], n_heads, x.shape[-1] // n_heads)


@torch.no_grad()
def estimate_head_mean(bundle, prompt_texts: list[str], layer_id: int, head_id: int,
                       position: int = -1, max_samples: int | None = None) -> torch.Tensor:
    layers=get_decoder_layers(bundle.model); layer=layers[layer_id]; proj=get_attention_output_projection(layer); n_heads=get_num_attention_heads(bundle.model,layer); device=get_input_device(bundle.model)
    total=None; n=0
    for prompt in tqdm(prompt_texts[:max_samples],desc=f"head mean L{layer_id}H{head_id}",leave=False):
        holder={}
        def pre(_m, inputs):
            x=inputs[0]; p=position if position>=0 else x.shape[1]+position; holder["v"]=_reshape_heads(x,n_heads)[0,p,head_id].detach().float().cpu()
        h=proj.register_forward_pre_hook(pre)
        try:
            enc=bundle.tokenizer(prompt,return_tensors="pt",add_special_tokens=False); enc={k:v.to(device) for k,v in enc.items()}; bundle.model(**enc,use_cache=False)
        finally: h.remove()
        v=holder["v"]; total=v.clone() if total is None else total+v; n+=1
    if n==0: raise ValueError("No prompts")
    return total/n


def make_head_mean_ablation_prehook(mean_head_cpu: torch.Tensor, head_id: int, n_heads: int, position: int=-1):
    mean=mean_head_cpu.detach().cpu()
    def pre(_m, inputs):
        x=inputs[0]; p=position if position>=0 else x.shape[1]+position; y=x.clone(); yh=_reshape_heads(y,n_heads); yh[:,p,head_id,:]=mean.to(y.device,dtype=y.dtype); return (y,)+tuple(inputs[1:])
    return pre


def make_head_zero_ablation_prehook(head_id: int, n_heads: int, position: int=-1):
    def pre(_m, inputs):
        x=inputs[0]; p=position if position>=0 else x.shape[1]+position; y=x.clone(); yh=_reshape_heads(y,n_heads); yh[:,p,head_id,:]=0; return (y,)+tuple(inputs[1:])
    return pre


def make_slice_mean_ablation_prehook(mean_slice_cpu: torch.Tensor, dim_start: int,
                                     dim_end: int, all_positions: bool = True):
    mean = mean_slice_cpu.detach().cpu()

    def pre(_m, inputs):
        x = inputs[0]
        y = x.clone()
        m = mean.to(y.device, dtype=y.dtype)
        if all_positions:
            y[..., dim_start:dim_end] = m
        else:
            y[:, -1, dim_start:dim_end] = m
        return (y,) + tuple(inputs[1:])
    return pre


def make_slice_patch_prehook(donor_slice_cpu: torch.Tensor, dim_start: int,
                             dim_end: int, all_positions: bool = True):
    donor = donor_slice_cpu.detach().cpu()

    def pre(_m, inputs):
        x = inputs[0]
        if x.shape[1] != donor.shape[0]:
            return None
        y = x.clone()
        d = donor.to(y.device, dtype=y.dtype)
        if all_positions:
            y[..., dim_start:dim_end] = d[..., dim_start:dim_end]
        else:
            y[:, -1, dim_start:dim_end] = d[..., -1, dim_start:dim_end]
        return (y,) + tuple(inputs[1:])
    return pre


@torch.no_grad()
def estimate_proj_slice_mean(bundle, prompt_texts: list[str], layer_id: int,
                             proj_name: str, dim_start: int, dim_end: int,
                             max_samples: int | None = None) -> torch.Tensor:
    layer = get_decoder_layers(bundle.model)[layer_id]
    proj = getattr(layer.self_attn, proj_name)
    device = get_input_device(bundle.model)
    total = None
    n_tokens = 0
    for prompt in tqdm(prompt_texts[:max_samples], desc=f"{proj_name} slice mean L{layer_id}", leave=False):
        holder = {}
        def pre(_m, inputs):
            holder["v"] = inputs[0].detach().float().cpu()
        h = proj.register_forward_pre_hook(pre)
        try:
            enc = bundle.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            bundle.model(**enc, use_cache=False)
        finally:
            h.remove()
        v = holder["v"][0, :, dim_start:dim_end]
        total = v.sum(0) if total is None else total + v.sum(0)
        n_tokens += v.shape[0]
    return total / n_tokens

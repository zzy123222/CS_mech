import sys, os, json, argparse, time
ROOT = '.'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from src.hooks.core import first_tensor
from src.interventions.hooks import make_mean_ablation_hook, make_steer_hook
from src.models.load import load_causal_lm
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module, get_decoder_layers

ap = argparse.ArgumentParser()
ap.add_argument('--part', choices=['null', 'transfer', 'both'], default='both')
ap.add_argument('--n_eval', type=int, default=64)
ap.add_argument('--n_est', type=int, default=64)
ap.add_argument('--n_rand', type=int, default=20)
ap.add_argument('--out', default='results/p0/p0_6_7')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

bundle = load_causal_lm('Qwen/Qwen3-4B', dtype='bfloat16', device_map='auto', chat=True)
tok, model = bundle.tokenizer, bundle.model
layers = get_decoder_layers(model)
NL = len(layers)
g = pd.read_parquet('data/generated/p1_graph_4b_seed1.parquet')
b = pd.read_parquet('data/generated/p3_task_battery.parquet')
tasks = {'graph': g, 'retrieval': b[b.task_family == 'retrieval']}


def prep(p): return bundle.prepare_prompt(p, enable_thinking=False)
def plen(p): return len(tok(p, add_special_tokens=False)['input_ids'])


def margin(p, r, s, hooks=None):
    z = score_answer_pairs(bundle, [p, p], [str(r), str(s)], batch_size=2, hooks=hooks)
    return float(z[0] - z[1])


def sel(df, split, cond, n):
    return df[(df.split == split) & (df.condition == cond)].head(n)


def cap_all(prompt, specs):
    box, hs = {}, []
    for k, sp in specs.items():
        hs.append(resolve_module(model, sp).register_forward_hook((lambda k_: (lambda m, i, o: box.__setitem__(k_, first_tensor(o)[0, -1].detach().float().cpu())))(k)))
    try:
        enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device); model(**enc, use_cache=False)
    finally:
        for h in hs: h.remove()
    return box


t0 = time.time()
if args.part in ('null', 'both'):
    specs = {f'L{l}:{k}': ModuleSpec(l, k) for l in range(NL) for k in ('attn', 'mlp')}
    rows = []
    for task, df in tasks.items():
        est = [prep(p) for p in sel(df, 'train', 'R-only', args.n_est).prompt]
        caps = [cap_all(p, specs) for p in est]
        means = {k: torch.stack([c[k] for c in caps]).mean(0) for k in specs}
        ev = sel(df, 'test_ood', 'R+S-conflict', args.n_eval)
        ro = sel(df, 'test_ood', 'R-only', 10 ** 6).set_index('group_id')
        for _, r in tqdm(ev.iterrows(), total=len(ev), desc=f'null {task}', leave=False):
            p = prep(r.prompt); pos = plen(p) - 1
            base = margin(p, r.rule_answer, r.shortcut_answer)
            rows.append({'task': task, 'module': 'base', 'g': r.group_id, 'v': base})
            for k, sp in specs.items():
                m = margin(p, r.rule_answer, r.shortcut_answer, hooks=[(resolve_module(model, sp), make_mean_ablation_hook(means[k], pos))])
                rows.append({'task': task, 'module': k, 'g': r.group_id, 'v': m})
        pd.DataFrame(rows).to_parquet(f'{args.out}/module_null.parquet')
        print(task, 'null done', round(time.time() - t0), 's', flush=True)

if args.part in ('transfer', 'both'):
    SP = ModuleSpec(24, 'attn'); mod = resolve_module(model, SP)
    dirs, evs = {}, {}
    for task, df in tasks.items():
        cf = [prep(p) for p in sel(df, 'train', 'R+S-conflict', args.n_est).prompt]
        ro = [prep(p) for p in sel(df, 'train', 'R-only', args.n_est).prompt]
        hc = torch.stack([cap_all(p, {'x': SP})['x'] for p in cf]); hr = torch.stack([cap_all(p, {'x': SP})['x'] for p in ro])
        dirs[task] = (hc.mean(0) - hr.mean(0))
        evs[task] = sel(df, 'test_ood', 'R+S-conflict', args.n_eval)
        print(task, 'dir norm', float(dirs[task].norm()), flush=True)
    gen = torch.Generator().manual_seed(7)
    def rand_orth(v):
        w = torch.randn(v.shape, generator=gen); w = w - (w @ v) / (v @ v) * v
        return w / w.norm() * v.norm()
    alphas = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    rows = []
    for tgt in tasks:
        base = {}
        for _, r in evs[tgt].iterrows():
            p = prep(r.prompt); base[r.group_id] = (p, plen(p) - 1, r.rule_answer, r.shortcut_answer, margin(p, r.rule_answer, r.shortcut_answer))
        for src in tasks:
            vecs = {'dir': dirs[src]} | {f'rand{i}': rand_orth(dirs[src]) for i in range(args.n_rand)}
            for name, v in vecs.items():
                for a in alphas:
                    for gid, (p, pos, r_, s_, mb) in tqdm(base.items(), desc=f'{src}->{tgt} {name} a={a}', leave=False, disable=True):
                        m = margin(p, r_, s_, hooks=[(mod, make_steer_hook(v, a, pos))])
                        rows.append({'src': src, 'tgt': tgt, 'vec': name, 'alpha': a, 'g': gid, 'delta': m - mb})
            print(src, '->', tgt, 'done', round(time.time() - t0), 's', flush=True)
            pd.DataFrame(rows).to_parquet(f'{args.out}/transfer.parquet')
pd.DataFrame().to_parquet(f'{args.out}/DONE.parquet')
print('DONE', flush=True)

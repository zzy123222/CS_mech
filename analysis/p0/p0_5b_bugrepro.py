import sys, os, json
ROOT = '.'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from src.hooks.core import first_tensor, replace_first_tensor
from src.models.load import load_causal_lm
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module, get_decoder_layers

bundle = load_causal_lm('Qwen/Qwen3-4B', dtype='bfloat16', device_map='auto', chat=True)
tok, model = bundle.tokenizer, bundle.model
L = 24; layers = get_decoder_layers(model); oproj = layers[L].self_attn.o_proj; blk = resolve_module(model, ModuleSpec(L, 'attn'))
NH, HD = model.config.num_attention_heads, model.config.head_dim
HEADS = [29, 31, 28]
df = pd.read_parquet('data/generated/p1_graph_4b_seed1.parquet')
cl = df[df.condition == 'R+S-consistent'].set_index('group_id'); co = df[df.condition == 'R+S-conflict'].set_index('group_id')
ids = list(cl.index.intersection(co.index)); np.random.default_rng(1).shuffle(ids)
prep = lambda p: bundle.prepare_prompt(p, enable_thinking=False)
plen = lambda p: len(tok(p, add_special_tokens=False)['input_ids'])


def margin(p, r, s, pre=None, hook=None):
    hs = []
    if pre: hs.append(oproj.register_forward_pre_hook(pre))
    if hook: hs.append(blk.register_forward_hook(hook))
    try:
        z = score_answer_pairs(bundle, [p, p], [str(r), str(s)], batch_size=2)
        return float(z[0] - z[1])
    finally:
        for h in hs: h.remove()


def capture(p):
    box = {}
    h1 = oproj.register_forward_pre_hook(lambda m, i: box.__setitem__('h', i[0].detach().float().cpu()[0]))
    h2 = blk.register_forward_hook(lambda m, i, o: box.__setitem__('b', first_tensor(o).detach().float().cpu()[0]))
    enc = tok(p, return_tensors='pt', add_special_tokens=False).to(model.device); model(**enc, use_cache=False)
    h1.remove(); h2.remove(); return box['h'], box['b']


rows = []
for g in tqdm(ids, total=len(ids), leave=False):
    if len([r for r in rows]) >= 64: break
    cp, xp = prep(cl.loc[g].prompt), prep(co.loc[g].prompt)
    if plen(cp) != plen(xp): continue
    r, s = co.loc[g].rule_answer, co.loc[g].shortcut_answer
    T = plen(xp); hd, bd = capture(cp)
    hd = hd.view(T, NH, HD)
    mc, mx = margin(cp, r, s), margin(xp, r, s)
    rec = {'g': g, 'clean': mc, 'conflict': mx}
    for variant in ('row0', 'allrows'):
        rowsel = (lambda x: x[0:1]) if variant == 'row0' else (lambda x: x)
        def pre_last(_m, inp, rowsel=rowsel):
            x = inp[0].clone(); xv = x.view(*x.shape[:-1], NH, HD)
            xv[:xv.shape[0] if variant == 'allrows' else 1, T - 1, HEADS, :] = hd[T - 1, HEADS, :].to(x.device, x.dtype)
            return (x,) + tuple(inp[1:])
        def pre_all(_m, inp):
            x = inp[0].clone(); xv = x.view(*x.shape[:-1], NH, HD)
            nrow = xv.shape[0] if variant == 'allrows' else 1
            for hh in HEADS:
                xv[:nrow, :T, hh, :] = hd[:, hh, :].to(x.device, x.dtype)
            return (x,) + tuple(inp[1:])
        def blk_all(_m, _i, out):
            h_ = first_tensor(out); h2 = h_.clone(); nrow = h2.shape[0] if variant == 'allrows' else 1
            h2[:nrow, :T] = bd.to(h_.device, h_.dtype); return replace_first_tensor(out, h2)
        rec[f'{variant}_heads_last'] = margin(xp, r, s, pre=pre_last)
        rec[f'{variant}_heads_all'] = margin(xp, r, s, pre=pre_all)
        rec[f'{variant}_block_all'] = margin(xp, r, s, hook=blk_all)
    rows.append(rec)
d = pd.DataFrame(rows)
os.makedirs('results/p0/p0_5b', exist_ok=True)
d.to_parquet('results/p0/p0_5b/bugrepro.parquet')
den = (d.clean - d.conflict)
out = {'n': len(d)}
for v in ('row0', 'allrows'):
    for k in ('heads_last', 'heads_all', 'block_all'):
        rec_ = ((d[f'{v}_{k}'] - d.conflict) / den)
        out[f'{v}_{k}'] = {'mean_of_ratios': float(rec_.mean()), 'ratio_of_means': float((d[f'{v}_{k}'] - d.conflict).mean() / den.mean())}
print(json.dumps(out, indent=1))
json.dump(out, open('results/p0/p0_5b/bugrepro.json', 'w'), indent=1)

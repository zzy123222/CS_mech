import sys, os, json, argparse, itertools, time
ROOT = '.'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from src.datasets.generate_error_monitor import build_check_prompt
from src.hooks.core import first_tensor, replace_first_tensor, temporary_hooks
from src.interventions.hooks import make_mean_ablation_hook, make_patch_hook, make_steer_hook
from src.interventions.patching import capture_module_vector
from src.interventions.ablation import estimate_mean_vector
from src.models.load import load_causal_lm
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module

ap = argparse.ArgumentParser()
ap.add_argument('--dtype', default='float32')
ap.add_argument('--n', type=int, default=128)
ap.add_argument('--n_rand', type=int, default=10)
ap.add_argument('--n_wrong', type=int, default=5)
ap.add_argument('--out', default='results/p0/p0_1_specificity_fp32.parquet')
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

bundle = load_causal_lm('Qwen/Qwen3-4B', dtype=args.dtype, device_map='auto', chat=True)
tok = bundle.tokenizer
df = pd.read_parquet('data/generated/p2_errors.parquet')
ev, es = df.iloc[:args.n].reset_index(drop=True), df.iloc[256:256 + 128].reset_index(drop=True)
CHECK, CONT = 'CHECK', 'CONTINUE'


def mk(row, variant):
    t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
    p = bundle.prepare_prompt(build_check_prompt(row, variant, t), enable_thinking=False)
    return p, len(tok(p, add_special_tokens=False)['input_ids']) - 1


EV = {v: [mk(r, v) for _, r in ev.iterrows()] for v in ('error', 'clean')}
ES = {v: [mk(r, v) for _, r in es.iterrows()] for v in ('error', 'clean')}
M_SPEC, C_SPEC, O_SPEC, R_SPEC = ModuleSpec(20, 'resid'), ModuleSpec(22, 'mlp'), ModuleSpec(25, 'attn'), ModuleSpec(35, 'resid')
mods = {'M': resolve_module(bundle.model, M_SPEC), 'C': resolve_module(bundle.model, C_SPEC), 'O': resolve_module(bundle.model, O_SPEC)}
specs = {'M': M_SPEC, 'C': C_SPEC, 'O': O_SPEC}
readout = resolve_module(bundle.model, R_SPEC)

t0 = time.time()
mean_vec = {k: estimate_mean_vector(bundle, [p for p, _ in ES['clean']], specs[k], max_samples=128) for k in 'MCO'}


def cap(spec, items):
    return torch.stack([capture_module_vector(bundle, p, spec, position=pos) for p, pos in items])


est_cap = {k: {v: cap(specs[k], ES[v]) for v in ('error', 'clean')} for k in 'MCO'}
est_r = {v: cap(R_SPEC, ES[v]) for v in ('error', 'clean')}
dirs = {k: (est_cap[k]['error'] - est_cap[k]['clean']).mean(0) for k in 'MCO'}
unit = {k: dirs[k] / dirs[k].norm() for k in 'MCO'}
dE = (est_r['error'] - est_r['clean']).mean(0); uE = dE / dE.norm()
muE = float((est_r['clean'] @ uE).mean())
mu_clean_M = est_cap['M']['clean'].mean(0)
print('estimation done', round(time.time() - t0), 's; |dir| M,C,O,E =', [round(float(dirs[k].norm()), 3) for k in 'MCO'], round(float(dE.norm()), 3), flush=True)

g = torch.Generator().manual_seed(1234)
rand_dirs = []
for _ in range(args.n_rand):
    w = torch.randn(unit['O'].shape, generator=g)
    w = w - (w @ unit['O']) * unit['O']
    rand_dirs.append(w / w.norm())

EVC = {k: {v: cap(specs[k], EV[v]) for v in ('error', 'clean')} for k in 'MCO'}
rng = np.random.default_rng(0)
wrong_idx = [rng.choice([j for j in range(args.n) if j != i], size=args.n_wrong, replace=False) for i in range(args.n)]


def centred_proj_hook(u, mu, pos):
    def hook(_m, _i, out):
        h = first_tensor(out); h2 = h.clone()
        v = u.to(h.device, h.dtype); m = mu.to(h.device, h.dtype)
        row = h2[:, pos, :]
        h2[:, pos, :] = row - ((row - m) @ v).unsqueeze(-1) * v
        return replace_first_tensor(out, h2)
    return hook


def uncentred_proj_hook(u, pos):
    def hook(_m, _i, out):
        h = first_tensor(out); h2 = h.clone()
        v = u.to(h.device, h.dtype)
        row = h2[:, pos, :]
        h2[:, pos, :] = row - (row @ v).unsqueeze(-1) * v
        return replace_first_tensor(out, h2)
    return hook


def run(prompt, pos, hooks):
    box = {}

    def rh(_m, _i, out):
        box['h'] = first_tensor(out)[:, pos, :].detach().float().cpu()[0]
    hs = list(hooks) + [(readout, rh)]
    with temporary_hooks(hs):
        z = score_answer_pairs(bundle, [prompt, prompt], [CHECK, CONT], batch_size=2)
    return float(z[0] - z[1]), float(box['h'] @ uE) - muE


rows = []
def rec(i, cond, name, param, c, e):
    rows.append({'i': i, 'sample_id': ev.sample_id[i], 'cond': cond, 'int': name, 'param': str(param), 'C': c, 'E': e})


for i in tqdm(range(args.n)):
    for cond, other in (('error', 'clean'), ('clean', 'error')):
        p, pos = EV[cond][i]
        c, e = run(p, pos, []); rec(i, cond, 'base', '', c, e)
        for k in 'MCO':
            c, e = run(p, pos, [(mods[k], make_mean_ablation_hook(mean_vec[k], pos))]); rec(i, cond, 'abl', k, c, e)
        c, e = run(p, pos, [(mods['M'], centred_proj_hook(unit['M'], mu_clean_M, pos))]); rec(i, cond, 'proj_centred_M', '', c, e)
        c, e = run(p, pos, [(mods['M'], uncentred_proj_hook(unit['M'], pos))]); rec(i, cond, 'proj_uncentred_M', '', c, e)
        for m_on, c_on, o_on in itertools.product([True, False], repeat=3):
            hk = []
            if not m_on: hk.append((mods['M'], make_mean_ablation_hook(mean_vec['M'], pos)))
            if not c_on: hk.append((mods['C'], make_mean_ablation_hook(mean_vec['C'], pos)))
            if not o_on: hk.append((mods['O'], make_mean_ablation_hook(mean_vec['O'], pos)))
            c, _ = run(p, pos, hk)
            rec(i, cond, 'fact', f"M{int(m_on)}C{int(c_on)}O{int(o_on)}", c, np.nan)
        for a in (-2.0, -1.0, 1.0, 2.0):
            c, _ = run(p, pos, [(mods['O'], make_steer_hook(unit['O'], a, pos))]); rec(i, cond, 'steer_dir', a, c, np.nan)
            for r_i, rd in enumerate(rand_dirs):
                c, _ = run(p, pos, [(mods['O'], make_steer_hook(rd, a, pos))]); rec(i, cond, 'steer_rand', f'{a}|{r_i}', c, np.nan)
        for k in 'MCO':
            c, e = run(p, pos, [(mods[k], make_patch_hook(EVC[k][other][i], pos))]); rec(i, cond, 'patch_matched', k, c, e)
            for jj, j in enumerate(wrong_idx[i]):
                c, e = run(p, pos, [(mods[k], make_patch_hook(EVC[k][other][j], pos))]); rec(i, cond, 'patch_wrong', f'{k}|{jj}', c, e)
    if i % 16 == 15:
        pd.DataFrame(rows).to_parquet(args.out)
pd.DataFrame(rows).to_parquet(args.out)
json.dump({'dtype': args.dtype, 'n': args.n, 'n_rand': args.n_rand, 'n_wrong': args.n_wrong, 'muE': muE,
           'dir_norms': {k: float(dirs[k].norm()) for k in 'MCO'}, 'seconds': round(time.time() - t0)},
          open(args.out.replace('.parquet', '_meta.json'), 'w'))
print('done', round(time.time() - t0), 's', flush=True)

import sys, os, json, argparse, time, gc
ROOT = '.'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.datasets.generate_error_monitor import build_check_prompt
from src.hooks.core import first_tensor, temporary_hooks
from src.interventions.hooks import make_mean_ablation_hook, make_steer_hook
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module

ap = argparse.ArgumentParser()
ap.add_argument('--arms', nargs='+', default=['general_sft', 'correction_sft', 'sham_correction'])
ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
ap.add_argument('--steps', nargs='+', type=int, default=[0, 10, 20, 50, 100, 200, 500, 1000])
ap.add_argument('--n', type=int, default=64)
ap.add_argument('--n_probe', type=int, default=128)
ap.add_argument('--out', default='results/p0/p0_3_rescan.csv')
ap.add_argument('--include_instruct', action='store_true')
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class B:
    def __init__(self, model, tok): self.model, self.tokenizer = model, tok
    def prep(self, p): return self.tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load(base_id, adapter):
    m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map='auto')
    if adapter and os.path.exists(adapter):
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    t = AutoTokenizer.from_pretrained(base_id)
    if t.pad_token is None: t.pad_token = t.eos_token
    return B(m, t)


df = pd.read_parquet('data/generated/p2_errors.parquet')
EV, PR = df.iloc[:args.n].reset_index(drop=True), df.iloc[256:256 + args.n_probe].reset_index(drop=True)
CH, CO = 'CHECK', 'CONTINUE'
SPECS = {'L20': ModuleSpec(20, 'resid'), 'L22mlp': ModuleSpec(22, 'mlp'), 'L24attn': ModuleSpec(24, 'attn'), 'L25attn': ModuleSpec(25, 'attn'), 'L35': ModuleSpec(35, 'resid')}


def auc_np(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def scan(b, tag, rows):
    tok, model = b.tokenizer, b.model
    mods = {k: resolve_module(model, s) for k, s in SPECS.items()}

    def mk(row, v):
        t = max(1, min(int(row.n_steps), int(row.error_step) + 1))
        p = b.prep(build_check_prompt(row, v, t)); return p, len(tok(p, add_special_tokens=False)['input_ids']) - 1

    ev = {v: [mk(r, v) for _, r in EV.iterrows()] for v in ('error', 'clean')}
    pr = {v: [mk(r, v) for _, r in PR.iterrows()] for v in ('error', 'clean')}

    def cap(items, keys):
        out = {k: [] for k in keys}
        for p, pos in items:
            box, hs = {}, []
            for k in keys:
                hs.append(mods[k].register_forward_hook((lambda k_: (lambda m, i, o: box.__setitem__(k_, first_tensor(o)[0, -1].detach().float().cpu())))(k)))
            enc = tok(p, return_tensors='pt', add_special_tokens=False).to(model.device); model(**enc, use_cache=False)
            for h in hs: h.remove()
            for k in keys: out[k].append(box[k])
        return {k: torch.stack(v) for k, v in out.items()}

    def run(p, pos, hooks=()):
        box = {}
        rh = lambda m, i, o: box.__setitem__('h', first_tensor(o)[0, pos].detach().float().cpu())
        with temporary_hooks(list(hooks) + [(mods['L35'], rh)]):
            z = score_answer_pairs(b, [p, p], [CH, CO], batch_size=2)
        return float(z[0] - z[1]), box['h']

    keys = ['L20', 'L22mlp', 'L24attn', 'L25attn', 'L35']
    ce, cc = cap(pr['error'], keys), cap(pr['clean'], keys)
    mean = {k: cc[k].mean(0) for k in ('L20', 'L22mlp', 'L25attn')}
    u25 = (ce['L25attn'].mean(0) - cc['L25attn'].mean(0)); u25 = u25 / u25.norm()
    u35 = (ce['L35'].mean(0) - cc['L35'].mean(0)); u35 = u35 / u35.norm(); mu35 = float((cc['L35'] @ u35).mean())
    rec = {'model': tag}
    for k in ('L20', 'L22mlp', 'L24attn', 'L25attn'):
        X = torch.cat([ce[k], cc[k]]).numpy(); y = np.r_[np.ones(len(ce[k])), np.zeros(len(cc[k]))]
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            clf = LogisticRegression(C=0.01, max_iter=300).fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        rec[f'probe_lr_auc_{k}'] = float(np.mean(aucs))
        Xr = torch.cat([ce[k], cc[k]]).numpy(); a2 = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xr, y):
            d_ = Xr[tr][y[tr] == 1].mean(0) - Xr[tr][y[tr] == 0].mean(0); a2.append(roc_auc_score(y[te], Xr[te] @ d_))
        rec[f'probe_nc_auc_{k}'] = float(np.mean(a2))
    res = {v: {'C': [], 'E': [], 'ctl': [], 'mon': [], 'monE': [], 'op_p': [], 'op_m': []} for v in ('error', 'clean')}
    for v in ('error', 'clean'):
        for p, pos in ev[v]:
            c0, h0 = run(p, pos); e0 = float(h0 @ u35) - mu35
            res[v]['C'].append(c0); res[v]['E'].append(e0)
            c1, _ = run(p, pos, [(mods['L22mlp'], make_mean_ablation_hook(mean['L22mlp'], pos))]); res[v]['ctl'].append(c1 - c0)
            c2, h2 = run(p, pos, [(mods['L20'], make_mean_ablation_hook(mean['L20'], pos))]); res[v]['mon'].append(c2 - c0); res[v]['monE'].append(float(h2 @ u35) - mu35 - e0)
            c3, _ = run(p, pos, [(mods['L25attn'], make_steer_hook(u25, +1.0, pos))]); c4, _ = run(p, pos, [(mods['L25attn'], make_steer_hook(u25, -1.0, pos))])
            res[v]['op_p'].append(c3 - c0); res[v]['op_m'].append(c4 - c0)
    Ce, Cc = np.array(res['error']['C']), np.array(res['clean']['C'])
    rec.update({'C_err': Ce.mean(), 'C_clean': Cc.mean(), 'AUC': auc_np(Ce, Cc), 'P_check_err': float((Ce > 0).mean()), 'P_cont_clean': float((Cc < 0).mean()),
                'balanced_acc': float(((Ce > 0).mean() + (Cc < 0).mean()) / 2), 'dprime': float((Ce.mean() - Cc.mean()) / np.sqrt(.5 * (Ce.var(ddof=1) + Cc.var(ddof=1)) + 1e-12)),
                'criterion': float(-(Ce.mean() + Cc.mean()) / 2)})
    for nm in ('ctl', 'mon', 'monE'):
        a, c = np.array(res['error'][nm]), np.array(res['clean'][nm])
        rec[f'{nm}_err'], rec[f'{nm}_clean'], rec[f'{nm}_disc'] = a.mean(), c.mean(), (a - c).mean()
    for v in ('error', 'clean'):
        rec[f'op_slope_{v}'] = float((np.array(res[v]['op_p']) - np.array(res[v]['op_m'])).mean() / 2)
    rec['E_err'] = float(np.mean(res['error']['E'])); rec['E_clean'] = float(np.mean(res['clean']['E']))
    rows.append(rec)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(tag, {k: round(v, 3) for k, v in rec.items() if k in ('AUC', 'balanced_acc', 'P_check_err', 'P_cont_clean', 'ctl_err', 'ctl_clean', 'probe_lr_auc_L22mlp')}, flush=True)


rows = []
done = set()
if os.path.exists(args.out):
    old = pd.read_csv(args.out); rows = old.to_dict('records'); done = set(old.model)
t0 = time.time()
if args.include_instruct and 'instruct_ref' not in done:
    b = load('Qwen/Qwen3-4B', None); scan(b, 'instruct_ref', rows); del b; gc.collect(); torch.cuda.empty_cache()
for arm in args.arms:
    for seed in args.seeds:
        for step in args.steps:
            tag = f'{arm}_seed{seed}@{step}'
            if step == 0 and tag.split('@')[0] != f'{args.arms[0]}_seed{args.seeds[0]}':
                continue
            if tag in done: continue
            ck = f'posttraining/results/formal/{arm}_seed{seed}/step_{step:04d}'
            if not os.path.exists(ck): print('skip', ck); continue
            b = load('Qwen/Qwen3-4B-Base', ck); scan(b, tag, rows); del b; gc.collect(); torch.cuda.empty_cache()
            print(round(time.time() - t0), 's elapsed', flush=True)
print('DONE', flush=True)

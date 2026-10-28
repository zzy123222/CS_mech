import sys, os, argparse, time, gc
ROOT = '.'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from src.datasets.generate_error_monitor import build_check_prompt
from src.hooks.core import first_tensor
from src.utils.module_access import ModuleSpec, resolve_module

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cuda')
ap.add_argument('--threads', type=int, default=48)
ap.add_argument('--limit', type=int, default=None)
ap.add_argument('--n_probe', type=int, default=128)
ap.add_argument('--tags', nargs='*', default=None)
ap.add_argument('--out', default='results/p0/p0_3_probe_grouped.csv')
args = ap.parse_args()
torch.set_num_threads(args.threads)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SPECS = {'L20': ModuleSpec(20, 'resid'), 'L22mlp': ModuleSpec(22, 'mlp'), 'L24attn': ModuleSpec(24, 'attn'), 'L25attn': ModuleSpec(25, 'attn')}
df = pd.read_parquet('data/generated/p2_errors.parquet')
PR = df.iloc[256:256 + args.n_probe].reset_index(drop=True)
if args.limit: PR = PR.iloc[:args.limit]


def load(base_id, adapter):
    dm = 'auto' if args.device == 'cuda' else None
    m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map=dm)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    t = AutoTokenizer.from_pretrained(base_id)
    return m, t


@torch.no_grad()
def capture(m, t):
    mods = {k: resolve_module(m, s) for k, s in SPECS.items()}
    out = {v: {k: [] for k in SPECS} for v in ('error', 'clean')}
    dev = next(m.parameters()).device
    for v in ('error', 'clean'):
        for _, r in PR.iterrows():
            step = max(1, min(int(r.n_steps), int(r.error_step) + 1))
            p = t.apply_chat_template([{'role': 'user', 'content': build_check_prompt(r, v, step)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            box, hs = {}, []
            for k in SPECS:
                hs.append(mods[k].register_forward_hook((lambda k_: (lambda mm, i, o: box.__setitem__(k_, first_tensor(o)[0, -1].detach().float().cpu())))(k)))
            enc = t(p, return_tensors='pt', add_special_tokens=False).to(dev); m(**enc, use_cache=False)
            for h in hs: h.remove()
            for k in SPECS: out[v][k].append(box[k])
    return {v: {k: torch.stack(x).numpy() for k, x in d.items()} for v, d in out.items()}


def probes(acts):
    rec = {}
    n = len(acts['error']['L20']); y = np.r_[np.ones(n), np.zeros(n)]; g = np.r_[np.arange(n), np.arange(n)]
    for k in SPECS:
        X = np.concatenate([acts['error'][k], acts['clean'][k]])
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6); a_old, c_old = [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
            clf = LogisticRegression(C=0.01, max_iter=300).fit(Xs[tr], y[tr]); a_old.append(roc_auc_score(y[te], clf.decision_function(Xs[te])))
            d_ = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0); c_old.append(roc_auc_score(y[te], X[te] @ d_))
        a_new, c_new = [], []
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300)).fit(X[tr], y[tr]); a_new.append(roc_auc_score(y[te], clf.decision_function(X[te])))
            d_ = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0); c_new.append(roc_auc_score(y[te], X[te] @ d_))
        rec.update({f'probe_lr_auc_{k}': float(np.mean(a_new)), f'probe_nc_auc_{k}': float(np.mean(c_new)),
                    f'old_probe_lr_auc_{k}': float(np.mean(a_old)), f'old_probe_nc_auc_{k}': float(np.mean(c_old))})
    return rec


tags = args.tags or pd.read_csv('results/p0/p0_3_rescan.csv').model.tolist()
rows, done = [], set()
if os.path.exists(args.out) and not args.limit:
    old = pd.read_csv(args.out); rows = old.to_dict('records'); done = set(old.model)
t0 = time.time()
for tag in tags:
    if tag in done: continue
    if tag == 'instruct_ref':
        m, t = load('Qwen/Qwen3-4B', None)
    else:
        arm_seed, step = tag.split('@')
        ck = None if int(step) == 0 else f'posttraining/results/formal/{arm_seed}/step_{int(step):04d}'
        m, t = load('Qwen/Qwen3-4B-Base', ck)
    rec = {'model': tag}; rec.update(probes(capture(m, t))); rows.append(rec)
    del m; gc.collect(); torch.cuda.empty_cache()
    if not args.limit: pd.DataFrame(rows).to_csv(args.out, index=False)
    print(tag, round(time.time() - t0), 's', {k: round(v, 3) for k, v in rec.items() if k.endswith('L24attn') or k.endswith('L22mlp')}, flush=True)
print('DONE', flush=True)

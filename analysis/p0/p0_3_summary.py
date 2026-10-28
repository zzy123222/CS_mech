import json
import numpy as np, pandas as pd
R = 'results/p0'
d = pd.read_csv(f'{R}/p0_3_rescan_v2.csv')
ref = d[d.model == 'instruct_ref'].iloc[0].to_dict() if (d.model == 'instruct_ref').any() else None
base = d[d.model.str.endswith('@0')].iloc[0].to_dict()
d = d[d.model.str.contains('@')].copy()
d['arm'] = d.model.str.extract(r'^(general_sft|correction_sft|sham_correction)')[0]
d['seed'] = d.model.str.extract(r'seed(\d)')[0].astype(int); d['step'] = d.model.str.extract(r'@(\d+)')[0].astype(int)
num = [c for c in d.columns if c not in ('model', 'arm')]
for c in num: d[c] = pd.to_numeric(d[c])
rows = []
for arm in d.arm.dropna().unique():
    for s in d.seed.unique():
        r0 = {k: v for k, v in base.items()}; r0.update({'arm': arm, 'seed': s, 'step': 0}); rows.append(r0)
D = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
for c in num: D[c] = pd.to_numeric(D[c])
D = D[D.step > 0].pipe(lambda x: pd.concat([x, pd.DataFrame(rows)], ignore_index=True)).sort_values(['arm', 'seed', 'step'])
for c in num: D[c] = pd.to_numeric(D[c])
out = {'n_checkpoints': int((d.step > 0).sum()), 'base': {k: base[k] for k in ('AUC', 'balanced_acc', 'P_check_err', 'P_cont_clean', 'ctl_err', 'ctl_clean', 'ctl_disc', 'probe_lr_auc_L22mlp', 'probe_lr_auc_L20', 'probe_lr_auc_L24attn', 'probe_lr_auc_L25attn', 'monE_disc', 'op_slope_error', 'op_slope_clean')},
       'reference': ({k: ref[k] for k in ('AUC', 'balanced_acc', 'P_check_err', 'P_cont_clean', 'ctl_err', 'ctl_clean', 'ctl_disc', 'probe_lr_auc_L22mlp', 'probe_lr_auc_L20', 'probe_lr_auc_L24attn', 'probe_lr_auc_L25attn', 'monE_disc', 'op_slope_error', 'op_slope_clean')} if ref else None)}
dd = D[D.step > 0]
out['balanced_acc'] = {'mean': float(dd.balanced_acc.mean()), 'max': float(dd.balanced_acc.max()), 'n': int(len(dd)), 'n_above_0.6': int((dd.balanced_acc > 0.6).sum())}
out['by_arm_step1000'] = {}
for arm, g in D[D.step == 1000].groupby('arm'):
    out['by_arm_step1000'][arm] = {k: [float(x) for x in g.sort_values('seed')[k].values] for k in ('AUC', 'balanced_acc', 'P_check_err', 'P_cont_clean', 'ctl_err', 'ctl_clean', 'ctl_disc', 'probe_lr_auc_L22mlp', 'probe_lr_auc_L24attn', 'monE_disc', 'op_slope_error', 'op_slope_clean')}
sl = {}
for k in ('ctl_err', 'ctl_disc', 'AUC', 'probe_lr_auc_L22mlp'):
    sl[k] = {}
    for (arm, s), g in D.groupby(['arm', 'seed']):
        g = g.sort_values('step'); sl[k].setdefault(arm, []).append(float(np.polyfit(np.log1p(g.step.values), g[k].values, 1)[0]))
out['run_slopes'] = sl
rng = np.random.default_rng(0)
def contrast(k, a, b):
    xa, xb = np.array(sl[k][a]), np.array(sl[k][b]); diff = xa.mean() - xb.mean()
    bs = [rng.choice(xa, 3).mean() - rng.choice(xb, 3).mean() for _ in range(4000)]
    return {'diff': float(diff), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'n_runs_a_gt_b': int(sum(np.array(sl[k][a])[:, None] > np.array(sl[k][b])[None, :]).sum()) if False else int((np.array(sl[k][a]) > np.array(sl[k][b])).sum())}
out['contrasts'] = {k: {f'{a}-{b}': contrast(k, a, b) for a, b in (('correction_sft', 'general_sft'), ('sham_correction', 'general_sft'), ('correction_sft', 'sham_correction'))} for k in ('ctl_err', 'ctl_disc')}
early = {}
for arm in ['correction_sft', 'general_sft', 'sham_correction']:
    g = D[D.arm == arm].groupby('step')[['probe_lr_auc_L22mlp', 'probe_lr_auc_L24attn', 'ctl_err', 'ctl_disc', 'AUC', 'balanced_acc']].mean()
    early[arm] = {int(s): {k: float(v) for k, v in r.items()} for s, r in g.iterrows()}
out['by_arm_step_mean'] = early
json.dump(out, open(f'{R}/p0_3_summary.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('n_checkpoints', 'balanced_acc', 'by_arm_step1000', 'contrasts')}, indent=1)[:5000])
print('base', out['base']); print('ref', out['reference'])
for arm in early:
    print(arm, {s: (round(v['probe_lr_auc_L22mlp'], 3), round(v['ctl_err'], 3), round(v['AUC'], 3)) for s, v in early[arm].items()})

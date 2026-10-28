import pandas as pd
R = 'results/p0'
old = pd.read_csv(f'{R}/p0_3_rescan.csv')
new = pd.read_csv(f'{R}/p0_3_probe_grouped.csv')
missing = set(old.model) - set(new.model)
assert not missing, f'grouped probes missing for {sorted(missing)}'
pc = [c for c in old.columns if c.startswith('probe_')]
v2 = old.drop(columns=pc).merge(new[['model'] + pc], on='model', how='left')[old.columns]
v2.to_csv(f'{R}/p0_3_rescan_v2.csv', index=False)
cmp = old[['model'] + pc].rename(columns={c: 'orig_' + c for c in pc}).merge(new, on='model')
cmp.to_csv(f'{R}/p0_3_probe_comparison.csv', index=False)
for k in ('L20', 'L22mlp', 'L24attn', 'L25attn'):
    a, b, c = cmp[f'orig_probe_lr_auc_{k}'], cmp[f'old_probe_lr_auc_{k}'], cmp[f'probe_lr_auc_{k}']
    print(f'{k:8s} original {a.mean():.3f}  old-scheme-rerun {b.mean():.3f}  pair-grouped {c.mean():.3f}  (mean over {len(cmp)} models; max |orig-rerun| {abs(a - b).max():.3f})')
print('wrote', f'{R}/p0_3_rescan_v2.csv')

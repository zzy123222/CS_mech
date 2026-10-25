import json, os
import numpy as np, pandas as pd
R = 'results/p0'
out = {}
nd = pd.read_parquet(f'{R}/p0_6_7/module_null.parquet')
for task in ('graph', 'retrieval'):
    sub = nd[nd.task == task]
    base = sub[sub.module == 'base'].set_index('g').v
    eff = {m: float((base - sub[sub.module == m].set_index('g').v).mean()) for m in sub.module.unique() if m != 'base'}
    s = pd.Series(eff).sort_values(ascending=False)
    others = s.drop('L24:attn')
    gs = base.index.values
    dmg = {m: (base - sub[sub.module == m].set_index('g').v).reindex(gs).values for m in s.index[:5]}
    rng = np.random.default_rng(0); ix = rng.integers(0, len(gs), (2000, len(gs)))
    ci = lambda x: [float(x.mean()), float(np.percentile(x[ix].mean(1), 2.5)), float(np.percentile(x[ix].mean(1), 97.5))]
    out[task] = {'n_modules': int(len(s)), 'rank_L24attn': int(list(s.index).index('L24:attn') + 1), 'effect_L24attn': ci(dmg['L24:attn']) if 'L24:attn' in dmg else None,
                 'top5': [(m, round(float(v), 3)) for m, v in s.head(5).items()], 'null_mean': float(others.mean()), 'null_sd': float(others.std()), 'null_max': float(others.max()),
                 'z_L24attn': float((s['L24:attn'] - others.mean()) / others.std()), 'empirical_p': float((1 + (others >= s['L24:attn']).sum()) / (len(s))),
                 'n_modules_above_1sd_of_null': int((others > others.mean() + others.std()).sum())}
tf = f'{R}/p0_6_7/transfer.parquet'
if os.path.exists(tf):
    td = pd.read_parquet(tf)
    res = {}
    for (s_, t_), g in td.groupby(['src', 'tgt']):
        gs = sorted(g.g.unique()); rng = np.random.default_rng(1); ix = rng.integers(0, len(gs), (2000, len(gs)))
        def curve(vec):
            sub = g[g.vec == vec].pivot_table(index='g', columns='alpha', values='delta').reindex(gs)
            return sub
        d = curve('dir'); rands = [curve(v) for v in g.vec.unique() if v != 'dir']
        al = list(d.columns)
        e = lambda sub, a: float(sub[a].mean())
        score = lambda sub: float(np.mean([(sub[a].mean() - sub[-a].mean()) / 2 for a in (0.5, 1.0, 2.0)]))
        sd = score(d); sr = np.array([score(r) for r in rands])
        vals = d.values; bs = []
        for i in ix[:1000]:
            dd = pd.DataFrame(vals[i], columns=al); bs.append(score(dd))
        res[f'{s_}->{t_}'] = {'curve': {str(a): e(d, a) for a in al}, 'dir_score': sd, 'dir_score_ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'rand_scores_mean': float(sr.mean()), 'rand_scores_absmax': float(np.abs(sr).max()),
                              'n_rand': int(len(sr)), 'p_abs_exceeds_rand': float((1 + (np.abs(sr) >= abs(sd)).sum()) / (len(sr) + 1)), 'rand_curve_abs_mean_at_1': float(np.mean([abs(e(r, 1.0)) for r in rands])), 'n_prompts': len(gs)}
    out['transfer'] = res
json.dump(out, open(f'{R}/p0_6_7_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

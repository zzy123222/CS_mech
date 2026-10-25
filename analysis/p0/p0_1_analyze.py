import os
import json, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results/p0'
F = 'figures/p0'; os.makedirs(F, exist_ok=True)
d = pd.read_parquet(f'{R}/p0_1_specificity_fp32.parquet')
N = d.i.nunique()
rng = np.random.default_rng(0)
B = 4000
idx = rng.integers(0, N, (B, N))
DELTA = 0.03


def wide(sub, col='C'):
    return sub.pivot_table(index='i', columns=['cond'], values=col)


def bci(x, fn=np.mean):
    x = np.asarray(x); bs = fn(x[idx], axis=1)
    return float(fn(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(a, b):
    return bci(np.asarray(a) - np.asarray(b))


out = {}
base = d[d.int == 'base'].pivot_table(index='i', columns='cond', values=['C', 'E'])
ce, cc = base['C']['error'].values, base['C']['clean'].values
auc = float(((ce[:, None] > cc[None, :]).mean() + 0.5 * (ce[:, None] == cc[None, :]).mean()))
dp = float((ce.mean() - cc.mean()) / np.sqrt(0.5 * (ce.var(ddof=1) + cc.var(ddof=1))))
within = float(np.mean(ce > cc))
out['baseline'] = {'C_error': bci(ce), 'C_clean': bci(cc), 'diff_matched': paired(ce, cc), 'AUC_unpaired': auc, 'AUC_matched_pairs': within,
                   'dprime': dp, 'P_CHECK_error': float((ce > 0).mean()), 'P_CHECK_clean': float((cc > 0).mean()),
                   'E_error': bci(base['E']['error'].values), 'E_clean': bci(base['E']['clean'].values), 'N': N}

tab = []
for name, sub in [('abl M (L20 resid)', d[(d.int == 'abl') & (d.param == 'M')]), ('abl C (L22 mlp)', d[(d.int == 'abl') & (d.param == 'C')]),
                  ('abl O (L25 attn)', d[(d.int == 'abl') & (d.param == 'O')]), ('proj_centred M', d[d.int == 'proj_centred_M']),
                  ('proj_uncentred M', d[d.int == 'proj_uncentred_M'])]:
    w = sub.pivot_table(index='i', columns='cond', values=['C', 'E'])
    de_, dc_ = w['C']['error'].values - ce, w['C']['clean'].values - cc
    ee, ec = w['E']['error'].values - base['E']['error'].values, w['E']['clean'].values - base['E']['clean'].values
    tab.append({'intervention': name, 'dC_err': bci(de_), 'dC_clean': bci(dc_), 'dC_disc': bci(de_ - dc_), 'dE_err': bci(ee), 'dE_clean': bci(ec), 'dE_disc': bci(ee - ec)})
out['ablation'] = tab

st = []
S = d[d.int == 'steer_dir']; SR = d[d.int == 'steer_rand']
for a in (-2.0, -1.0, 1.0, 2.0):
    w = S[S.param == str(a)].pivot_table(index='i', columns='cond', values='C')
    de_, dc_ = w['error'].values - ce, w['clean'].values - cc
    rnd = SR[SR.param.str.startswith(f'{a}|')].copy(); rnd['r'] = rnd.param.str.split('|').str[1]
    rw = rnd.pivot_table(index=['r', 'i'], columns='cond', values='C').reset_index()
    rw['de'] = rw['error'] - np.tile(ce, rw.r.nunique()); rw['dc'] = rw['clean'] - np.tile(cc, rw.r.nunique())
    pr = rw.groupby('r')[['de', 'dc']].mean()
    st.append({'alpha': a, 'dir_err': bci(de_), 'dir_clean': bci(dc_), 'dir_disc': bci(de_ - dc_), 'rand_err_mean': float(pr.de.mean()), 'rand_err_absmax': float(pr.de.abs().max()),
               'rand_clean_mean': float(pr.dc.mean()), 'ratio_dir_vs_rand_err': float(abs(de_.mean()) / max(pr.de.abs().mean(), 1e-9))})
out['steering'] = st
al = np.array([-2., -1., 1., 2.])
slope = lambda cond: np.array([np.polyfit(al, [S[(S.param == str(a)) & (S.cond == cond) & (S.i == i)].C.values[0] for a in al], 1)[0] for i in range(N)])
se, sc = slope('error'), slope('clean')
out['steering_slope'] = {'error': bci(se), 'clean': bci(sc), 'disc': bci(se - sc), 'slope_ratio_clean_over_error': float(sc.mean() / se.mean())}

F_ = d[d.int == 'fact'].pivot_table(index=['i'], columns=['cond', 'param'], values='C')
terms = {'M': [0], 'C': [1], 'O': [2], 'MxC': [0, 1], 'MxO': [0, 2], 'CxO': [1, 2], 'MxCxO': [0, 1, 2]}
fac = {}
for out_name in ('error', 'clean', 'disc'):
    res = {}
    for t, pos in terms.items():
        per = np.zeros(N)
        for lab in [f'M{a}C{b}O{c}' for a in (1, 0) for b in (1, 0) for c in (1, 0)]:
            bits = [int(lab[1]), int(lab[3]), int(lab[5])]
            sgn = np.prod([1 if bits[p] else -1 for p in pos])
            v = (F_[('error', lab)].values if out_name == 'error' else F_[('clean', lab)].values if out_name == 'clean' else F_[('error', lab)].values - F_[('clean', lab)].values)
            per += sgn * v / 8.0
        m, lo, hi = bci(per)
        se_ = per.std(ddof=1) / np.sqrt(N); tt = stats.ttest_1samp(per, 0)
        t1 = (per.mean() + DELTA) / se_; t2 = (per.mean() - DELTA) / se_
        p_tost = float(max(1 - stats.t.cdf(t1, N - 1), stats.t.cdf(t2, N - 1)))
        res[t] = {'beta': m, 'ci': [lo, hi], 'p': float(tt.pvalue), 'tost_p': p_tost, 'equivalent': bool(p_tost < 0.05)}
    fac[out_name] = res
out['factorial'] = fac

gap = float((ce - cc).mean())
pt = []
for k, nm in (('M', 'L20 resid'), ('C', 'L22 mlp'), ('O', 'L25 attn')):
    m = d[(d.int == 'patch_matched') & (d.param == k)].pivot_table(index='i', columns='cond', values='C')
    w = d[(d.int == 'patch_wrong') & (d.param.str.startswith(k + '|'))].groupby(['cond', 'i']).C.mean().unstack(0)
    nz_m, nz_w = m['error'].values - ce, w['error'].values - ce
    dn_m, dn_w = m['clean'].values - cc, w['clean'].values - cc
    rat = lambda x, sgn: bci(sgn * x / gap, fn=np.mean)
    pt.append({'module': nm, 'noising_matched_dC': bci(nz_m), 'noising_wrong_dC': bci(nz_w), 'denoising_matched_dC': bci(dn_m), 'denoising_wrong_dC': bci(dn_w),
               'noising_matched_frac_of_gap': rat(nz_m, -1), 'denoising_matched_frac_of_gap': rat(dn_m, 1),
               'noising_matched_minus_wrong': bci(nz_m - nz_w), 'denoising_matched_minus_wrong': bci(dn_m - dn_w)})
out['patch'] = pt
out['gap_error_minus_clean'] = gap
json.dump(out, open(f'{R}/p0_1_results.json', 'w'), indent=1, default=float)

fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
ax = axs[0]
labs = [t['intervention'].replace(' (', '\n(') for t in tab[:4]]
x = np.arange(len(labs)); w = 0.27
for j, (k, c, l) in enumerate([('dC_err', '#b5462a', 'error'), ('dC_clean', '#11808c', 'clean'), ('dC_disc', '#444444', 'error − clean')]):
    m = np.array([t[k][0] for t in tab[:4]]); lo = np.array([t[k][1] for t in tab[:4]]); hi = np.array([t[k][2] for t in tab[:4]])
    ax.bar(x + (j - 1) * w, m, w, yerr=[m - lo, hi - m], color=c, label=l, capsize=2)
ax.axhline(0, color='k', lw=.6); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel('Δ C (nats, fp32)'); ax.legend(frameon=False, fontsize=7); ax.set_title('Ablations act on both conditions')
ax = axs[1]
for cond, c, l in (('error', '#b5462a', 'error'), ('clean', '#11808c', 'clean')):
    ms = []; ls = []; hs = []
    for a in al:
        w_ = S[(S.param == str(a)) & (S.cond == cond)].C.values - (ce if cond == 'error' else cc)
        mm, lo, hi = bci(w_); ms.append(mm); ls.append(mm - lo); hs.append(hi - mm)
    ax.errorbar(al, ms, yerr=[ls, hs], fmt='-o', color=c, label=f'L25 direction ({l})', ms=4, capsize=2)
rr = SR.copy(); rr['a'] = rr.param.str.split('|').str[0].astype(float)
for cond, c in (('error', '#b5462a'), ('clean', '#11808c')):
    base_ = ce if cond == 'error' else cc
    mm = [rr[(rr.a == a) & (rr.cond == cond)].C.mean() - base_.mean() for a in al]
    ax.plot(al, mm, ':s', color=c, ms=3, alpha=.7, label=f'random ({cond})')
ax.axhline(0, color='k', lw=.6); ax.set_xlabel('α'); ax.set_ylabel('Δ C'); ax.legend(frameon=False, fontsize=6.5); ax.set_title('L25 steering is not error-specific')
ax = axs[2]
labs3 = ['M', 'C', 'O', 'M×C', 'M×O', 'C×O', 'M×C×O']
for j, (k, c, l) in enumerate([('error', '#b5462a', 'error'), ('clean', '#11808c', 'clean'), ('disc', '#444444', 'error − clean')]):
    m = np.array([fac[k][t]['beta'] for t in terms]); lo = np.array([fac[k][t]['ci'][0] for t in terms]); hi = np.array([fac[k][t]['ci'][1] for t in terms])
    ax.errorbar(m, np.arange(7)[::-1] + (j - 1) * .22, xerr=[m - lo, hi - m], fmt='o', color=c, ms=4, capsize=2, label=l)
ax.axvspan(-DELTA, DELTA, color='#ffe9a8', alpha=.6, label=f'equivalence ±{DELTA}')
ax.set_yticks(np.arange(7)[::-1]); ax.set_yticklabels(labs3); ax.axvline(0, color='k', lw=.6); ax.set_xlabel(f'β (+ = component intact), 95% CI, n={N}'); ax.legend(frameon=False, fontsize=6.5); ax.set_title('Factorial (fp32)')
ax = axs[3]
x = np.arange(3); w = 0.2
for j, (k, c, l) in enumerate([('noising_matched_frac_of_gap', '#b5462a', 'error ← clean donor'), ('denoising_matched_frac_of_gap', '#11808c', 'clean ← error donor')]):
    m = np.array([t[k][0] for t in pt]); lo = np.array([t[k][1] for t in pt]); hi = np.array([t[k][2] for t in pt])
    ax.bar(x + (j - .5) * w * 1.8, m, w * 1.8, yerr=[m - lo, hi - m], color=c, label=l, capsize=2)
ax.axhline(0, color='k', lw=.6); ax.set_xticks(x); ax.set_xticklabels([t['module'] for t in pt]); ax.set_ylabel('fraction of error−clean gap moved'); ax.legend(frameon=False, fontsize=7); ax.set_title('Matched patching (fraction of gap)')
plt.tight_layout(); plt.savefig(f'{F}/figP0_1_specificity.png', dpi=200)
print(json.dumps(out, indent=1, default=float)[:12000])

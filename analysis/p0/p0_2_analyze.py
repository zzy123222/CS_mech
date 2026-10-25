import json, os, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results/p0'
F = 'figures/p0'; os.makedirs(F, exist_ok=True)
CELLS = [('inst_chat', 'Qwen3-4B, chat'), ('inst_raw', 'Qwen3-4B, raw'), ('base_raw', 'Base, raw'), ('base_chat', 'Base, chat')]
COL = {'inst_chat': '#11808c', 'inst_raw': '#7bb8c0', 'base_raw': '#d68a44', 'base_chat': '#3b5b7a'}
GS = 4
rng = np.random.default_rng(0)


def load(tag, task='graph'):
    p = f'{R}/p0_2/{tag}/{task}_main.parquet'
    return pd.read_parquet(p) if os.path.exists(p) else None


def piv(m, kind, cast=None):
    x = m[m.kind == kind].pivot_table(index='g', columns='key', values='v')
    if cast:
        x.columns = [cast(c) for c in x.columns]
    return x


def rm(num, den, B=3000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    r = np.random.default_rng(seed); n = len(num)
    ix = r.integers(0, n, (B, n))
    b = num[ix].mean(1) / den[ix].mean(1)
    return float(num.mean() / den.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def mci(x, B=3000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed); n = len(x)
    b = x[r.integers(0, n, (B, n))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


res = {}
for tag, label in CELLS:
    m = load(tag)
    if m is None:
        continue
    base = piv(m, 'base')
    conf, ro = base['conflict'], base['ronly']
    ko = m[m.kind == 'block_ko'].set_index('g').v
    out = {'n_groups': int(len(conf)), 'conflict_margin': mci(conf), 'ronly_margin': mci(ro)}
    beh = pd.read_parquet(f'{R}/p0_2/{tag}/graph_behavior.parquet').groupby('cond')[['acc', 'margin']].mean()
    out['behavior'] = beh.round(4).to_dict()
    out['block_ko_damage'] = mci(conf - ko)
    out['block_ko_norm'] = rm(conf - ko, ro)
    hk = piv(m, 'head_ko', int)
    dmg = hk.rsub(conf, axis=0)
    out['head_damage_raw'] = {int(h): float(dmg[h].mean()) for h in dmg.columns}
    out['head_damage_norm'] = {int(h): rm(dmg[h], ro)[0] for h in dmg.columns}
    pos = np.clip(dmg.mean(), 0, None)
    grp = np.array([pos[[h for h in dmg.columns if h // GS == k]].sum() for k in range(8)])
    out['group_damage_raw'] = grp.tolist(); out['group7_share_of_positive'] = float(grp[7] / grp.sum())
    out['top4_heads'] = [int(h) for h in dmg.mean().sort_values(ascending=False).index[:4]]
    sk = piv(m, 'set_ko')
    sdmg = sk.rsub(conf, axis=0)
    out['set_ko_norm'] = {c: rm(sdmg[c], ro) for c in sdmg.columns}
    for kind in ('k_ko', 'v_ko', 'kv_ko'):
        x = piv(m, kind, int); out[kind + '_norm'] = {int(c): rm(x[c].rsub(conf), ro)[0] for c in x.columns}
    if (m.kind == 'block_patch').any():
        cons = base['consistent']
        bp = piv(m, 'block_patch')
        out['block_rescue_KO'] = {c: rm(bp[c] - ko, conf - ko) for c in bp.columns}
        out['block_recovery_clean'] = rm(bp['matched'] - conf, cons - conf)
        gain_blk = bp['matched'] - conf
        suff = {}
        for kind in ('set_patch', 'set_patch_ronly', 'set_patch_allpos'):
            if (m.kind == kind).any():
                sp = piv(m, kind)
                gb = {'set_patch': bp['matched'] - conf,
                      'set_patch_ronly': (piv(m, 'block_patch_ronly')['matched'] - conf) if (m.kind == 'block_patch_ronly').any() else None,
                      'set_patch_allpos': (piv(m, 'block_patch_allpos')['matched'] - conf) if (m.kind == 'block_patch_allpos').any() else None}[kind]
                suff[kind] = {c: {'gain': mci(sp[c] - conf), 'frac_of_block_gain': rm(sp[c] - conf, gb)} for c in sp.columns}
        out['sufficiency'] = suff
        for kind in ('k_patch', 'v_patch', 'kv_patch'):
            x = piv(m, kind, int)
            out[kind] = {int(c): rm(x[c] - conf, gain_blk) for c in x.columns}
    sw = f'{R}/p0_2/{tag}/graph_sweep.parquet'
    if os.path.exists(sw):
        s = pd.read_parquet(sw)
        curve = {}
        for l, g in s.groupby('layer'):
            curve[int(l)] = rm(g.patched - g.corrupt, g.clean - g.corrupt)
        out['sweep'] = curve
    ret = load(tag, 'retrieval')
    if ret is not None:
        b2 = piv(ret, 'base'); c2, r2 = b2['conflict'], b2['ronly']
        h2 = piv(ret, 'head_ko', int).rsub(c2, axis=0)
        out['retrieval'] = {'head_damage_norm': {int(h): rm(h2[h], r2)[0] for h in h2.columns},
                            'head_damage_raw': {int(h): float(h2[h].mean()) for h in h2.columns},
                            'group_damage_raw': [float(np.clip(h2.mean(), 0, None)[[h for h in h2.columns if h // GS == k]].sum()) for k in range(8)],
                            'set_ko_norm': {c: rm(piv(ret, 'set_ko')[c].rsub(c2), r2) for c in piv(ret, 'set_ko').columns},
                            'kv_ko_norm': {int(c): rm(piv(ret, 'kv_ko', int)[c].rsub(c2), r2)[0] for c in piv(ret, 'kv_ko', int).columns},
                            'n_groups': int(len(c2)), 'behavior': pd.read_parquet(f'{R}/p0_2/{tag}/retrieval_behavior.parquet').groupby('cond')[['acc', 'margin']].mean().round(4).to_dict()}
    res[tag] = out

if 'inst_chat' in res and 'retrieval' in res['inst_chat']:
    g = np.array(res['inst_chat']['group_damage_raw']); r = np.array(res['inst_chat']['retrieval']['group_damage_raw'])
    perms = list(itertools.permutations(range(8)))
    obs = stats.spearmanr(g, r)[0]
    null = np.array([stats.spearmanr(g, r[list(p)])[0] for p in perms])
    p_sp = float((np.sum(null >= obs - 1e-12)) / len(perms))
    top1_same = int(np.argmax(g) == np.argmax(r))
    hg = np.array([res['inst_chat']['head_damage_raw'][h] for h in (28, 29, 30, 31)]); hr = np.array([res['inst_chat']['retrieval']['head_damage_raw'][h] for h in (28, 29, 30, 31)])
    obs_w = stats.spearmanr(hg, hr)[0]
    null_w = np.array([stats.spearmanr(hg, hr[list(p)])[0] for p in itertools.permutations(range(4))])
    res['gqa_aware'] = {'group_spearman': float(obs), 'group_spearman_exact_p': p_sp, 'top1_group_same': top1_same, 'top1_group_graph': int(np.argmax(g)), 'top1_group_retrieval': int(np.argmax(r)),
                        'p_top1_same_under_null': 1 / 8, 'within_group7_spearman': float(obs_w), 'within_group7_exact_p': float(np.mean(null_w >= obs_w - 1e-12)),
                        'group7_share_graph': float(g[7] / np.clip(g, 0, None).sum()), 'group7_share_retrieval': float(r[7] / np.clip(r, 0, None).sum())}
json.dump(res, open(f'{R}/p0_2_results.json', 'w'), indent=1, default=float)

fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw={'width_ratios': [1.5, 1, 1.15, 1]})
ax = axs[0]
x = np.arange(32); cells = [c for c in CELLS if c[0] in res]; w = 0.8 / len(cells)
for j, (tag, lab) in enumerate(cells):
    ax.bar(x + (j - (len(cells) - 1) / 2) * w, [res[tag]['head_damage_norm'][h] for h in range(32)], w, color=COL[tag], label=lab)
for k in range(8):
    ax.axvspan(4 * k - .5, 4 * k + 3.5, color='#fbe3dc' if k == 7 else ('#f4f4f4' if k % 2 else 'white'), zorder=-1)
ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=5.5); ax.set_xlabel('L24 query head (KV group = 4 consecutive heads)'); ax.set_ylabel('ablation damage / R-only margin')
ax.legend(frameon=False, fontsize=6.5, loc='upper left'); ax.set_title('Necessity is confined to KV group 7')
ax = axs[1]
for tag, lab in cells:
    if 'sweep' in res[tag]:
        ls = sorted(res[tag]['sweep']); ax.plot(ls, [res[tag]['sweep'][l][0] for l in ls], color=COL[tag], label=lab, lw=1.5)
        ax.fill_between(ls, [res[tag]['sweep'][l][1] for l in ls], [res[tag]['sweep'][l][2] for l in ls], color=COL[tag], alpha=.12)
ax.axvline(24, ls=':', color='k', lw=.8); ax.set_xlabel('layer (resid, terminal pos.)'); ax.set_ylabel('patch recovery'); ax.legend(frameon=False, fontsize=6.5); ax.set_title('Patch transition by model × format')
ax = axs[2]
names = ['H28_29_31', 'qgroup7', 'all_but_grp7', 'H30']
labs = ['H28+H29+H31', 'group 7\n(4 heads)', 'other 28\nheads', 'H30\nalone']
cells_c = [c for c in cells if c[0] in ('inst_chat', 'base_raw')]
wc = 0.8 / len(cells_c)
for j, (tag, lab) in enumerate(cells_c):
    s_ = res[tag]['sufficiency']['set_patch_ronly']
    m_ = np.array([s_[n]['frac_of_block_gain'][0] for n in names]); lo = np.array([s_[n]['frac_of_block_gain'][1] for n in names]); hi = np.array([s_[n]['frac_of_block_gain'][2] for n in names])
    ax.bar(np.arange(4) + (j - (len(cells_c) - 1) / 2) * wc, m_, wc, yerr=[m_ - lo, hi - m_], color=COL[tag], capsize=1.5, label=lab)
ax.axhline(1, color='k', lw=.6, ls='--'); ax.set_ylim(-0.1, 1.35); ax.legend(frameon=False, fontsize=6.5, loc='upper right'); ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel('patch gain / full-block patch gain'); ax.set_title('Sufficiency of head sets (R-only donor)')
ax = axs[3]
tg = 'inst_chat'
if tg in res and 'k_patch' in res[tg]:
    kk = np.arange(8)
    ax.bar(kk - .27, [res[tg]['k_patch'][k][0] for k in kk], .27, color='#b5462a', label='K slice patch')
    ax.bar(kk, [res[tg]['v_patch'][k][0] for k in kk], .27, color='#7e57c2', label='V slice patch')
    ax.bar(kk + .27, [res[tg]['kv_patch'][k][0] for k in kk], .27, color='#444444', label='K+V patch')
    ax.axhline(1, color='k', lw=.6, ls='--'); ax.set_xlabel('KV head'); ax.set_ylabel('patch gain / block gain'); ax.legend(frameon=False, fontsize=6.5); ax.set_title('K/V-slice patches (Qwen3-4B, chat)')
plt.tight_layout(); plt.savefig(f'{F}/figP0_2_format_kv.png', dpi=200)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('n_groups', 'block_ko_norm', 'group7_share_of_positive', 'top4_heads', 'behavior', 'block_rescue_KO', 'block_recovery_clean')} for k, v in res.items() if k != 'gqa_aware'}, indent=1, default=float)[:9000])
print(json.dumps(res.get('gqa_aware'), indent=1))

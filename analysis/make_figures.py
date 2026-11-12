import os, json, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from PIL import Image

C = '.'
R = 'results/raw'
V = 'results/p0'
P4 = 'results/posttraining'
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.1, 'lines.markersize': 3, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'legend.frameon': False, 'axes.titlepad': 3, 'savefig.dpi': 600})
W = 6.7
BLUE, ORANGE, VERM, GREEN, SKY, PURPLE, GREY, BLACK, YEL = '#0072B2', '#E69F00', '#D55E00', '#009E73', '#56B4E9', '#CC79A7', '#8C8C8C', '#222222', '#F0E442'
CELL = {'inst_chat': BLUE, 'inst_raw': SKY, 'base_raw': ORANGE, 'base_chat': PURPLE}
CLAB = {'inst_chat': 'Qwen3-4B, chat', 'inst_raw': 'Qwen3-4B, raw', 'base_raw': 'Base, raw', 'base_chat': 'Base, chat'}
ARM = {'general_sft': (GREY, 'Arm A (general)'), 'correction_sft': (VERM, 'Arm B (true repairs)'), 'sham_correction': (BLUE, 'Arm C (sham repairs)')}
ERR, CLN = VERM, GREEN
G7 = '#FBE5C8'


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{OUT}/{name}.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    im = Image.open(f'{OUT}/{name}.png'); im.convert('RGB').save(f'{OUT}/{name}.tif', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)


def lab(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def eb(ax, x, est, color, width, label=None, hatch=None, **kw):
    m = np.array([e[0] for e in est]); lo = np.array([e[1] for e in est]); hi = np.array([e[2] for e in est])
    ax.bar(x, m, width, yerr=[m - lo, hi - m], color=color, label=label, capsize=1.6, error_kw={'elinewidth': 0.6, 'capthick': 0.6}, hatch=hatch, edgecolor='white' if not hatch else 'k', linewidth=0.3, **kw)


def kvshade(ax, ymin=None):
    for g in range(8):
        ax.axvspan(4 * g - 0.5, 4 * g + 3.5, color=G7 if g == 7 else ('#F2F2F2' if g % 2 else 'white'), zorder=-2, lw=0)


P2 = json.load(open(f'{V}/p0_2_results.json'))
P1 = json.load(open(f'{V}/p0_1_results.json'))
P3 = json.load(open(f'{V}/p0_3_summary.json'))
BR = json.load(open(f'{V}/p0_5b/bugrepro.json'))
IC, BRW = P2['inst_chat'], P2['base_raw']

def card(ax, x, y, w, h, fc='#F7F7F7', ec='#BBBBBB'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.02', fc=fc, ec=ec, lw=0.6))


fig = plt.figure(figsize=(W, 4.6))
ax = fig.add_axes([0.0, 0.52, 0.50, 0.46]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); lab(ax, 'a', -0.0, 0.96)
ax.text(0.04, 0.93, 'Evidence arbitration (graph reasoning)', fontsize=7.5, fontweight='bold')
card(ax, 0.03, 0.30, 0.55, 0.58)
lines = [('Follow the arrows: starting at N130, where', BLACK), ('do you arrive after 2 links?', BLACK), ('Graph facts:  N060 → N253', BLUE), ('                     N333 → N230', BLUE),
         ('                     N130 → N060', BLUE), ('A previous quick solver selected option B.', ORANGE), ('Options:  A N230   B N389', BLACK), ('               C N253   D N241', BLACK)]
for i, (t, c) in enumerate(lines):
    ax.text(0.05, 0.83 - i * 0.066, t, fontsize=6.2, color=c, family='Liberation Sans')
ax.text(0.05, 0.22, 'Rule answer: C (via N060)', fontsize=6.4, color=BLUE)
ax.text(0.05, 0.14, 'Shortcut answer: B (cue)', fontsize=6.4, color=ORANGE)
ax.text(0.05, 0.04, r'$M=\log P(\mathrm{rule})-\log P(\mathrm{shortcut})$', fontsize=6.6)
rows = [('R-only', 'yes', '–'), ('S-only', '–', 'cue only'), ('Consistent', 'yes', 'agrees'), ('Conflict', 'yes', 'conflicts'), ('S-matched', 'filler', 'cue only')]
ax.text(0.63, 0.83, 'Condition', fontsize=6.3, fontweight='bold'); ax.text(0.81, 0.83, 'Rule', fontsize=6.3, fontweight='bold'); ax.text(0.89, 0.83, 'Cue', fontsize=6.3, fontweight='bold')
for i, (a, b, c) in enumerate(rows):
    y = 0.74 - i * 0.09
    ax.text(0.63, y, a, fontsize=6.3, color=VERM if a == 'Conflict' else BLACK); ax.text(0.82, y, b, fontsize=6.3, color=BLUE); ax.text(0.89, y, c, fontsize=6.3, color=ORANGE)
ax.plot([0.62, 0.99], [0.80, 0.80], color='#999999', lw=0.5)
ax = fig.add_axes([0.52, 0.52, 0.48, 0.46]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); lab(ax, 'b', -0.0, 0.96)
ax.text(0.04, 0.93, 'Error-conditioned verification', fontsize=7.5, fontweight='bold')
card(ax, 0.03, 0.36, 0.94, 0.52)
vl = [('Start value: 38.', BLACK), ('Step 1: apply ×3; result = 114.', BLACK), ('Step 2: apply −5; result = 111.', VERM), ('Step 3: apply +9; result = 120.', BLACK),
      ('Should the solver CONTINUE, or CHECK the work?', BLACK)]
for i, (t, c) in enumerate(vl):
    ax.text(0.06, 0.82 - i * 0.095, t, fontsize=6.4, color=c)
ax.text(0.64, 0.63, '← error (should be 109)', fontsize=6.0, color=VERM)
ax.text(0.06, 0.27, 'Matched clean trajectory: identical except step 2 = 109 and step 3 = 118.', fontsize=6.2, color=GREEN)
ax.text(0.06, 0.16, r'$C=\log P(\mathrm{CHECK})-\log P(\mathrm{CONTINUE})$, scored on both', fontsize=6.6)
ax.text(0.06, 0.07, 'error and clean trajectories (response criterion vs. error sensitivity)', fontsize=6.2)
ax = fig.add_axes([0.0, 0.0, 0.50, 0.50]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); lab(ax, 'c', -0.0, 0.94)
ax.text(0.04, 0.91, 'Causal assays on layer-24 attention', fontsize=7.5, fontweight='bold')
ax.add_patch(Rectangle((0.06, 0.10), 0.012, 0.72, color='#DDDDDD'))
ax.text(0.02, 0.84, 'residual stream', fontsize=5.8, color='#666666')
card(ax, 0.12, 0.25, 0.60, 0.48, fc='#FFFFFF', ec='#888888')
ax.text(0.14, 0.68, 'Attention block (32 query heads, 8 key/value heads)', fontsize=6.0)
for g in range(8):
    x0 = 0.145 + g * 0.071
    ax.add_patch(Rectangle((x0 - 0.004, 0.31), 0.066, 0.31, color=G7 if g == 7 else '#F0F0F0', lw=0))
    ax.add_patch(Rectangle((x0 + 0.012, 0.52), 0.034, 0.06, color=ORANGE if g == 7 else '#BBBBBB', lw=0))
    ax.text(x0 + 0.029, 0.545, 'K/V', fontsize=4.6, ha='center', va='center', color='white' if g == 7 else BLACK)
    for q in range(4):
        ax.add_patch(Rectangle((x0 + q * 0.015, 0.36), 0.012, 0.11, color=(VERM if g == 7 and q != 2 else (ORANGE if g == 7 else '#9E9E9E')), lw=0))
    ax.text(x0 + 0.029, 0.325, str(g), fontsize=5.2, ha='center')
ax.text(0.66, 0.255, 'group 7 = heads 28–31', fontsize=5.6, color=VERM, ha='right')
ax.annotate('', xy=(0.42, 0.74), xytext=(0.42, 0.82), arrowprops=dict(arrowstyle='->', lw=0.6))
for i, (t, c) in enumerate([('Mean ablation: replace output by its clean mean → necessity', BLACK), ('Patching: copy output from a donor run → sufficiency', BLUE),
                            ('Key/value-slice patching of one key/value head', ORANGE), ('Steering: add α·u along an error − clean direction', PURPLE)]):
    ax.text(0.12, 0.17 - i * 0.055, '• ' + t, fontsize=5.9, color=c)
ax = fig.add_axes([0.52, 0.0, 0.48, 0.50]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); lab(ax, 'd', -0.0, 0.94)
ax.text(0.04, 0.91, 'Controlled post-training from the base model', fontsize=7.5, fontweight='bold')
card(ax, 0.04, 0.47, 0.16, 0.22, fc='#FFF4E0', ec=ORANGE); ax.text(0.12, 0.58, 'Qwen3-4B-\nBase', fontsize=6.2, ha='center', va='center')
for i, (k, t) in enumerate([('general_sft', 'A: general instructions'), ('correction_sft', 'B: + true error→repair'), ('sham_correction', 'C: + sham repairs (same prompts)')]):
    y = 0.76 - i * 0.17
    c = ARM[k][0]
    ax.add_patch(FancyArrowPatch((0.21, 0.58), (0.30, y), arrowstyle='->', mutation_scale=6, lw=0.6, color=c))
    ax.text(0.31, y + 0.02, t, fontsize=6.0, color=c)
    xs = np.array([0, 10, 20, 50, 100, 200, 500, 1000]); xx = 0.31 + 0.62 * np.log10(xs + 1) / np.log10(1001)
    ax.plot(xx, np.full_like(xx, y - 0.035), '-o', color=c, ms=2.2, lw=0.6)
for st_, lb_ in ((0, '0'), (10, '10'), (100, '100'), (1000, '1,000')):
    ax.text(0.31 + 0.62 * np.log10(st_ + 1) / np.log10(1001), 0.33, lb_, fontsize=5.4, ha='center')
ax.text(0.62, 0.285, 'fine-tuning step', fontsize=5.4, ha='center', color='#555555')
ax.text(0.06, 0.20, 'LoRA, 1,000 steps, three independent runs per arm.', fontsize=5.9)
ax.text(0.06, 0.13, 'Each checkpoint: error/clean margins, ablations, L25 steering,', fontsize=5.9)
ax.text(0.06, 0.07, 'linear probes and head tracking.', fontsize=5.9)
save(fig, 'Fig1')

fig, axs = plt.subplots(1, 4, figsize=(W, 1.9), gridspec_kw={'width_ratios': [0.8, 1, 1.25, 1.5], 'wspace': 0.55})
ax = axs[0]
conds = ['R-only', 'S-only', 'R+S-consistent', 'R+S-conflict', 'S-matched']; cl = ['R-only', 'S-only', 'Consist.', 'Conflict', 'S-match.']
acc = [IC['behavior']['acc'][c] for c in conds]
ax.bar(range(5), acc, 0.7, color=[BLUE, ORANGE, GREEN, VERM, GREY])
ax.set_xticks(range(5)); ax.set_xticklabels(cl, rotation=45, ha='right'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.05); ax.axhline(0.25, color='k', ls=':', lw=0.6); lab(ax, 'a', -0.42)
ax = axs[1]
s = IC['sweep']; ls = sorted(int(k) for k in s)
ax.plot(ls, [s[str(l)][0] for l in ls], color=BLUE); ax.fill_between(ls, [s[str(l)][1] for l in ls], [s[str(l)][2] for l in ls], color=BLUE, alpha=0.2, lw=0)
ax.axvline(24, color='k', ls=':', lw=0.6); ax.set_xlabel('Layer (residual, final token)'); ax.set_ylabel('Patch recovery'); ax.set_xlim(0, 35); lab(ax, 'b', -0.3)
ax = axs[2]
nd = pd.read_parquet(f'{V}/p0_6_7/module_null.parquet')
for task, c, mk in (('graph', BLUE, 'o'), ('retrieval', ORANGE, 's')):
    sub = nd[nd.task == task]; base = sub[sub.module == 'base'].set_index('g').v
    for kind, mfc in (('attn', c), ('mlp', 'white')):
        mods = [m for m in sub.module.unique() if m.endswith(':' + kind)]
        Ls = [int(m.split(':')[0][1:]) for m in mods]; eff = [float((base - sub[sub.module == m].set_index('g').v).mean()) for m in mods]
        ax.scatter(Ls, eff, s=7, marker=mk, facecolor=mfc, edgecolor=c, lw=0.5, label=f'{task}, {kind}', zorder=3)
ax.set_xlabel('Layer'); ax.set_ylabel('Ablation damage (nats)'); ax.set_ylim(-1.5, 21); ax.legend(loc='upper left', fontsize=5.4, handletextpad=0.2, ncol=2, columnspacing=0.6)
ax.annotate('L24 attention', xy=(23.5, 15.8), xytext=(8, 12.0), fontsize=6, arrowprops=dict(arrowstyle='-', lw=0.5)); lab(ax, 'c', -0.27)
ax = axs[3]
kvshade(ax)
hn = IC['head_damage_norm']; ax.bar(range(32), [hn[str(h)] for h in range(32)], 0.75, color=[VERM if h // 4 == 7 else '#7F7F7F' for h in range(32)])
ax.set_xticks(range(0, 32, 4)); ax.set_xlabel('L24 query head'); ax.set_ylabel('Damage / R-only margin'); ax.set_xlim(-0.8, 31.8)
for h in (28, 29, 31):
    ax.text(h, hn[str(h)] + 0.006, f'H{h}', fontsize=5.5, ha='center')
ax.set_ylim(-0.025, 0.285); ax.text(29.5, 0.262, 'group 7', fontsize=5.5, ha='center', color=VERM)
lab(ax, 'd', -0.2)
save(fig, 'Fig2')

fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw={'width_ratios': [1.15, 1.35, 1.2], 'wspace': 0.5})
ax = axs[0]
names = [('block', 'Whole block'), ('qgroup7', 'Group 7'), ('H28_29_31', 'H28+H29+H31'), ('H29_31', 'H29+H31'), ('H30', 'H30'), ('all_but_grp7', 'Other 28')]
est = [IC['block_ko_norm'] if k == 'block' else IC['set_ko_norm'][k] for k, _ in names]
eb(ax, np.arange(len(names)), est, [BLACK, VERM, VERM, VERM, ORANGE, GREY], 0.7)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n for _, n in names], rotation=45, ha='right'); ax.set_ylabel('Ablation damage / R-only margin'); ax.axhline(0, color='k', lw=0.5); lab(ax, 'a', -0.38)
ax = axs[1]
sets = [('qgroup7', 'Group 7'), ('H28_29_31', 'H28+29+31'), ('H30', 'H30'), ('all_but_grp7', 'Other 28')]
for j, (kind, l, c) in enumerate([('set_patch_ronly', 'R-only donor, final token', BLUE), ('set_patch', 'Consistent donor, final token', SKY), ('set_patch_allpos', 'Consistent donor, all tokens', GREEN)]):
    sub = IC['sufficiency'][kind]
    xs = np.arange(len(sets)) + (j - 1) * 0.26
    est = [sub[k]['frac_of_block_gain'] if k in sub else [np.nan] * 3 for k, _ in sets]
    keep = [i for i, e in enumerate(est) if not np.isnan(e[0])]
    eb(ax, xs[keep], [est[i] for i in keep], c, 0.25, label=l)
ax.axhline(1, color='k', ls='--', lw=0.5); ax.set_xticks(range(len(sets))); ax.set_xticklabels([n for _, n in sets], rotation=45, ha='right'); ax.set_ylabel('Patch gain / whole-block gain'); ax.set_ylim(-0.05, 1.62); ax.legend(loc='upper right', fontsize=5.2); lab(ax, 'b', -0.3)
ax = axs[2]
kk = np.arange(8)
for j, (k, l, c) in enumerate([('k_patch', 'Keys', VERM), ('v_patch', 'Values', BLUE), ('kv_patch', 'Keys + values', BLACK)]):
    eb(ax, kk + (j - 1) * 0.27, [IC[k][str(i)] for i in kk], c, 0.26, label=l)
ax.axhline(1, color='k', ls='--', lw=0.5); ax.set_xticks(kk); ax.set_xlabel('Key-value head'); ax.set_ylabel('Patch gain / whole-block gain'); ax.legend(loc='upper left', fontsize=5.6); ax.set_ylim(-0.2, 1.15); lab(ax, 'c', -0.3)
save(fig, 'Fig3')

fig, axs = plt.subplots(1, 4, figsize=(W, 1.95), gridspec_kw={'width_ratios': [1.45, 0.95, 1.2, 1.1], 'wspace': 0.48})
ax = axs[0]
kvshade(ax)
rh = IC['retrieval']['head_damage_norm']; ax.bar(range(32), [rh[str(h)] for h in range(32)], 0.75, color=[VERM if h // 4 == 7 else '#7F7F7F' for h in range(32)])
ax.set_xticks(range(0, 32, 4)); ax.set_xlabel('L24 query head'); ax.set_ylabel('Damage / R-only margin'); ax.set_xlim(-0.8, 31.8)
for h in (28, 29, 31):
    ax.text(h, rh[str(h)] + 0.0015, f'H{h}', fontsize=5.5, ha='center')
lab(ax, 'a', -0.22)
ax = axs[1]
g = np.clip(np.array(IC['group_damage_raw']), 0, None); r = np.clip(np.array(IC['retrieval']['group_damage_raw']), 0, None)
ax.bar(np.arange(8) - 0.2, g / g.sum(), 0.38, color=BLUE, label='Graph'); ax.bar(np.arange(8) + 0.2, r / r.sum(), 0.38, color=ORANGE, label='Retrieval')
ax.set_xticks(range(8)); ax.set_xlabel('Key/value group'); ax.set_ylabel('Share of head damage'); ax.legend(loc='upper left', fontsize=5.6); lab(ax, 'b', -0.38)
ax = axs[2]
ho = pd.read_csv(f'{R}/p3_control_hubs/heldout_confirm_summary.csv')
x = 0; ticks = []
for task, base_c in (('Graph', BLUE), ('Retrieval', ORANGE)):
    for sd in ('Seed 1', 'Seed 2', 'Seed 3'):
        v = ho[(ho.task == task) & (ho.seed == sd)].set_index('head').damage
        for j, (h, a) in enumerate((('H29', 1.0), ('H31', 0.65), ('H28', 0.35))):
            ax.bar(x + (j - 1) * 0.27, v[h], 0.26, color=base_c, alpha=a, label=(h if (task == 'Graph' and sd == 'Seed 1') else None))
        ticks.append((x, f'{task[0]}{sd[-1]}')); x += 1
ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks]); ax.set_xlabel('Task (G/R) and dataset seed'); ax.set_ylabel('Held-out ablation damage (nats)'); ax.legend(loc='upper right', fontsize=5.6); lab(ax, 'c', -0.3)
ax = axs[3]
td = pd.read_parquet(f'{V}/p0_6_7/transfer.parquet')
for (s_, t_), c in ((('graph', 'retrieval'), VERM), (('retrieval', 'graph'), BLUE)):
    sub = td[(td.src == s_) & (td.tgt == t_)]
    dd = sub[sub.vec == 'dir'].groupby('alpha').delta.mean(); rr = sub[sub.vec != 'dir'].groupby(['vec', 'alpha']).delta.mean().unstack(0)
    ax.fill_between(rr.index, rr.min(axis=1), rr.max(axis=1), color=c, alpha=0.15, lw=0)
    ax.plot(dd.index, dd.values, '-o', color=c, ms=2.5, label=f'{s_} → {t_}')
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('Steering coefficient α'); ax.set_ylabel('Change in target margin (nats)'); ax.legend(loc='upper left', fontsize=5.6); lab(ax, 'd', -0.32)
save(fig, 'Fig4')

cells = ['inst_chat', 'inst_raw', 'base_raw', 'base_chat']
fig, axs = plt.subplots(1, 4, figsize=(W, 1.95), gridspec_kw={'width_ratios': [1.1, 1.2, 1.1, 1.0], 'wspace': 0.45})
ax = axs[0]
xs = np.arange(4)
ax.bar(xs - 0.18, [P2[c]['behavior']['acc']['R-only'] for c in cells], 0.34, color=BLUE, label='R-only')
ax.bar(xs + 0.18, [P2[c]['behavior']['acc']['R+S-conflict'] for c in cells], 0.34, color=VERM, label='Conflict')
ax.axhline(0.25, color='k', ls=':', lw=0.6); ax.set_xticks(xs); ax.set_xticklabels([CLAB[c] for c in cells], rotation=45, ha='right'); ax.set_ylabel('Accuracy'); ax.legend(loc='upper right', fontsize=5.6); ax.set_ylim(0, 1); lab(ax, 'a', -0.33)
ax = axs[1]
for c in cells:
    s = P2[c]['sweep']; ls = sorted(int(k) for k in s)
    ax.plot(ls, [s[str(l)][0] for l in ls], color=CELL[c], label=CLAB[c])
    ax.fill_between(ls, [s[str(l)][1] for l in ls], [s[str(l)][2] for l in ls], color=CELL[c], alpha=0.15, lw=0)
ax.axvline(24, color='k', ls=':', lw=0.6); ax.set_xlabel('Layer (residual, final token)'); ax.set_ylabel('Patch recovery'); ax.legend(loc='upper left', fontsize=5.4); ax.set_xlim(0, 35); lab(ax, 'b', -0.3)
ax = axs[2]
eb(ax, xs - 0.18, [P2[c]['block_ko_norm'] for c in cells], BLACK, 0.34, label='Whole block')
eb(ax, xs + 0.18, [P2[c]['set_ko_norm']['qgroup7'] for c in cells], VERM, 0.34, label='Group 7')
ax.axhline(0, color='k', lw=0.5); ax.set_xticks(xs); ax.set_xticklabels([CLAB[c] for c in cells], rotation=45, ha='right'); ax.set_ylabel('Ablation damage / R-only margin'); ax.legend(loc='upper right', fontsize=5.6); lab(ax, 'c', -0.33)
ax = axs[3]
hs = [28, 29, 30, 31]
for j, c in enumerate(['inst_chat', 'base_raw']):
    ax.bar(np.arange(4) + (j - 0.5) * 0.36, [P2[c]['head_damage_norm'][str(h)] for h in hs], 0.34, color=CELL[c], label=CLAB[c])
ax.set_xticks(range(4)); ax.set_xticklabels([f'H{h}' for h in hs]); ax.set_xlabel('Group-7 query head'); ax.set_ylabel('Damage / R-only margin'); ax.set_ylim(-0.04, 0.36); ax.legend(loc='upper right', fontsize=5.6); lab(ax, 'd', -0.33)
save(fig, 'Fig5')

models = [('Qwen3-0.6B', 'p2_error_monitor_06b'), ('Qwen3-4B-Base', 'p2_error_monitor_base'), ('Qwen3-4B', 'p2_error_monitor'), ('Qwen3-8B', 'p2_error_monitor_8b'), ('Mistral-7B', 'p2_error_monitor_mistral')]
MC = [GREY, ORANGE, BLUE, GREEN, PURPLE]
fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), gridspec_kw={'width_ratios': [1.6, 1.0, 1.0], 'wspace': 0.42})
ax = axs[0]
stats_ = []
for i, (n, dname) in enumerate(models):
    df = pd.read_parquet(f'{R}/{dname}/timecourse_behavior.parquet'); df = df[df.time_step >= df.error_step]
    e, c = df[df.variant == 'error'].check_margin.values, df[df.variant == 'clean'].check_margin.values
    auc = float(((e[:, None] > c[None, :]).mean() + 0.5 * (e[:, None] == c[None, :]).mean()))
    stats_.append((n, e.mean(), c.mean(), auc, (c > 0).mean()))
    for v, arr, col, off in (('error', e, ERR, -0.17), ('clean', c, CLN, 0.17)):
        bp = ax.boxplot(arr, positions=[i + off], widths=0.28, patch_artist=True, showfliers=False, medianprops={'color': 'k', 'lw': 0.7}, whiskerprops={'lw': 0.6}, capprops={'lw': 0.6}, boxprops={'lw': 0.5})
        for b in bp['boxes']: b.set_facecolor(col); b.set_alpha(0.75)
    ax.text(i, 27.5, f'AUC\n{auc:.2f}', ha='center', fontsize=5.6, linespacing=1.0)
ax.axhline(0, color='k', lw=0.5); ax.set_xticks(range(5)); ax.set_xticklabels([m for m, _ in models], rotation=30, ha='right'); ax.set_ylabel('Verification margin C (nats)'); ax.set_ylim(-20, 33)
ax.plot([], [], 's', color=ERR, label='Error trajectory'); ax.plot([], [], 's', color=CLN, label='Clean trajectory'); ax.legend(loc='lower left', fontsize=5.6); lab(ax, 'a', -0.16)
ax = axs[1]
for (n, me, mc, auc, pc), col in zip(stats_, MC):
    off = {'Qwen3-0.6B': (4, -7), 'Qwen3-4B-Base': (4, -3), 'Qwen3-4B': (-40, 3), 'Qwen3-8B': (-4, 6), 'Mistral-7B': (5, -6)}[n]
    ax.scatter(mc, me, color=col, s=16, zorder=3); ax.annotate(n, (mc, me), xytext=off, textcoords='offset points', fontsize=5.4)
lim = (-5, 16); ax.plot(lim, lim, color='k', lw=0.5, ls='--'); ax.axhline(0, color='#999999', lw=0.5); ax.axvline(0, color='#999999', lw=0.5)
ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xlabel('Mean C, clean'); ax.set_ylabel('Mean C, error'); lab(ax, 'b', -0.3)
ax = axs[2]
d1 = pd.read_parquet(f'{V}/p0_1_specificity_fp32.parquet'); b = d1[d1.int == 'base'].pivot_table(index='i', columns='cond', values='C')
bins = np.linspace(8, 18, 26)
ax.hist(b['clean'], bins=bins, color=CLN, alpha=0.35, lw=0); ax.hist(b['clean'], bins=bins, color=CLN, histtype='step', lw=1.0, label='Clean')
ax.hist(b['error'], bins=bins, color=ERR, alpha=0.30, lw=0); ax.hist(b['error'], bins=bins, color=ERR, histtype='step', lw=1.0, label='Error')
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.set_xlabel('C (nats), Qwen3-4B (n = 128 pairs)'); ax.set_ylabel('Trajectories'); ax.legend(loc='upper left', fontsize=5.6); lab(ax, 'c', -0.3)
save(fig, 'Fig6')

fig, axs = plt.subplots(1, 4, figsize=(W, 2.0), gridspec_kw={'width_ratios': [1.1, 1.0, 1.15, 1.1], 'wspace': 0.68})
ab = {t['intervention']: t for t in P1['ablation']}
ax = axs[0]
mods = [('abl M (L20 resid)', 'L20\nresidual'), ('abl C (L22 mlp)', 'L22\nMLP'), ('abl O (L25 attn)', 'L25\nattention')]
for j, (k, l, c) in enumerate([('dC_err', 'Error', ERR), ('dC_clean', 'Clean', CLN), ('dC_disc', 'Error − clean', BLACK)]):
    eb(ax, np.arange(3) + (j - 1) * 0.26, [ab[m][k] for m, _ in mods], c, 0.25, label=l)
ax.axhline(0, color='k', lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in mods]); ax.set_ylabel('Ablation effect on C (nats)'); ax.set_ylim(-0.8, 1.05); ax.set_yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]); ax.legend(loc='upper center', fontsize=5.2, ncol=3, columnspacing=0.5, handlelength=1.0); lab(ax, 'a', -0.36)
ax = axs[1]
d1['a'] = np.nan
S = d1[d1.int == 'steer_dir']; SR = d1[d1.int == 'steer_rand'].copy(); SR['a'] = SR.param.str.split('|').str[0].astype(float)
base_c = d1[d1.int == 'base'].pivot_table(index='i', columns='cond', values='C')
al = [-2.0, -1.0, 1.0, 2.0]
rlo, rhi = [], []
for a in al:
    v = [(SR[(SR.a == a) & (SR.cond == cond)].groupby('param').C.mean() - base_c[cond].mean()) for cond in ('error', 'clean')]
    rlo.append(min(x.min() for x in v)); rhi.append(max(x.max() for x in v))
ax.fill_between([-2, -1, 0, 1, 2], rlo[:2] + [0] + rlo[2:], rhi[:2] + [0] + rhi[2:], color='#BBBBBB', alpha=0.5, lw=0, label='Random')
for cond, col in (('error', ERR), ('clean', CLN)):
    m = [(S[(S.param == str(a)) & (S.cond == cond)].set_index('i').C - base_c[cond]).mean() for a in al]
    ax.plot([-2, -1, 0, 1, 2], [m[0], m[1], 0, m[2], m[3]], '-o', color=col, ms=2.5, label=cond.capitalize())
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('Steering coefficient α'); ax.set_ylabel('Change in C (nats)'); ax.legend(loc='upper right', fontsize=5.2); lab(ax, 'b', -0.33)
ax = axs[2]
fac = P1['factorial']['disc']; terms = ['M', 'C', 'O', 'MxC', 'MxO', 'CxO', 'MxCxO']; tl = ['L20', 'L22', 'L25', 'L20×L22', 'L20×L25', 'L22×L25', '3-way']
y = np.arange(len(terms))[::-1]
mm = np.array([fac[t]['beta'] for t in terms]); lo = np.array([fac[t]['ci'][0] for t in terms]); hi = np.array([fac[t]['ci'][1] for t in terms])
ax.axvspan(-0.03, 0.03, color='#EEEEEE', lw=0, label='Equivalence bound')
ax.errorbar(mm, y, xerr=[mm - lo, hi - mm], fmt='o', color=BLACK, ms=2.8, elinewidth=0.7, capsize=1.5)
ax.axvline(0, color='k', lw=0.5); ax.set_yticks(y); ax.set_yticklabels(tl); ax.set_xlabel('Effect on error − clean C (β)'); ax.set_ylim(-0.8, len(terms) + 0.6); ax.legend(loc='upper center', fontsize=5.2, bbox_to_anchor=(0.5, 1.02)); lab(ax, 'c', -0.42)
ax = axs[3]
pt = {t['module']: t for t in P1['patch']}
pm = [('L20 resid', 'L20\nresidual'), ('L22 mlp', 'L22\nMLP'), ('L25 attn', 'L25\nattention')]
eb(ax, np.arange(3) - 0.18, [pt[m]['noising_matched_frac_of_gap'] for m, _ in pm], ERR, 0.34, label='Clean donor → error trajectory')
eb(ax, np.arange(3) + 0.18, [pt[m]['denoising_matched_frac_of_gap'] for m, _ in pm], CLN, 0.34, label='Error donor → clean trajectory')
ax.axhline(0, color='k', lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in pm]); ax.set_ylabel('Shift toward other condition\n(fraction of error − clean gap)'); ax.set_ylim(-0.75, 1.0); ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75]); ax.legend(loc='upper center', fontsize=5.0, handlelength=1.0); lab(ax, 'd', -0.4)
save(fig, 'Fig7')

RSF = f'{V}/p0_3_rescan_v2.csv'
if not os.path.exists(RSF):
    print('WARNING: grouped-probe rescan table not available yet; Fig. 8e uses the original probes (draft only)'); RSF = f'{V}/p0_3_rescan.csv'
rs = pd.read_csv(RSF)
ref = rs[rs.model == 'instruct_ref'].iloc[0]; base0 = rs[rs.model.str.endswith('@0')].iloc[0]
d = rs[rs.model.str.contains('@') & ~rs.model.str.endswith('@0')].copy()
d['arm'] = d.model.str.extract(r'^(general_sft|correction_sft|sham_correction)')[0]; d['seed'] = d.model.str.extract(r'seed(\d)')[0].astype(int); d['step'] = d.model.str.extract(r'@(\d+)')[0].astype(int)
fig, axs = plt.subplots(2, 3, figsize=(W, 3.8), gridspec_kw={'wspace': 0.48, 'hspace': 0.62})
ax = axs[0, 0]
xs_ = np.linspace(0, 1, 20); ax.plot(xs_, 1 - xs_, color='k', ls='--', lw=0.5)
rng_ = np.random.default_rng(0)
for arm, (c, l) in ARM.items():
    sub = d[d.arm == arm]
    jx = rng_.normal(0, 0.012, len(sub)) if arm != 'general_sft' else 0; jy = rng_.normal(0, 0.012, len(sub)) if arm != 'general_sft' else 0
    ax.scatter(sub.P_cont_clean + jx, sub.P_check_err + jy, s=7, color=c, alpha=0.75, lw=0)
nbc = int((d.arm != 'general_sft').sum())
ax.annotate(f'all {nbc} checkpoints of arms B and C,\nbase model and reference', xy=(0.04, 1.0), xytext=(0.22, 1.02), fontsize=5.0, va='center', arrowprops=dict(arrowstyle='-', lw=0.5))
ax.text(0.02, 0.02, 'dashed: balanced accuracy = 0.5\ngrey: arm A checkpoints', fontsize=5.2, color='#555555')
ax.set_xlabel('P(CONTINUE | clean)'); ax.set_ylabel('P(CHECK | error)'); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.08); lab(ax, 'a', -0.3)


def traj(ax, key, ylabel, refline=True, basev=None):
    for arm, (c, l) in ARM.items():
        g = d[d.arm == arm].groupby('step')[key].agg(['mean', 'min', 'max'])
        g.loc[0] = [base0[key]] * 3; g = g.sort_index()
        ax.plot(g.index + 1, g['mean'], '-o', color=c, ms=2.2, label=l); ax.fill_between(g.index + 1, g['min'], g['max'], color=c, alpha=0.13, lw=0)
    if refline: ax.axhline(ref[key], color='k', ls='--', lw=0.6, label='Qwen3-4B reference')
    ax.set_xscale('log'); ax.set_xlabel('Fine-tuning step + 1'); ax.set_ylabel(ylabel)


traj(axs[0, 1], 'AUC', 'AUC of C (error vs. clean)'); lab(axs[0, 1], 'b', -0.36); axs[0, 1].axhline(0.5, color='#999999', lw=0.5, ls=':')
traj(axs[0, 2], 'ctl_disc', 'L22 MLP effect,\nerror − clean (nats)'); lab(axs[0, 2], 'c', -0.36)
traj(axs[1, 0], 'monE_disc', 'L20 effect on E,\nerror − clean'); lab(axs[1, 0], 'd', -0.36)
traj(axs[1, 1], 'probe_lr_auc_L24attn', 'L24 attention probe AUC'); lab(axs[1, 1], 'e', -0.36)
ax = axs[1, 2]
hr = pd.read_csv(f'{P4}/analysis/head_rank_similarity_3seed.csv')
hr['arm'] = hr.model.str.extract(r'^(general_sft|correction_sft|sham_correction)')[0]; hr['step'] = hr.model.str.extract(r'@(\d+)')[0].astype(int)
for arm, (c, l) in ARM.items():
    g = hr[(hr.arm == arm) & (hr.task == 'retrieval')].groupby('step').rho.agg(['mean', 'min', 'max'])
    ax.plot(g.index, g['mean'], '-o', color=c, ms=2.2, label=l); ax.fill_between(g.index, g['min'], g['max'], color=c, alpha=0.13, lw=0)
ax.axhline(0, color='#999999', lw=0.5, ls=':'); ax.set_xlabel('Fine-tuning step'); ax.set_ylabel('Six-head damage profile,\nρ with reference (retrieval)'); lab(ax, 'f', -0.36)
h_, l_ = axs[0, 2].get_legend_handles_labels(); fig.legend(h_, l_, loc='upper center', ncol=4, fontsize=6, bbox_to_anchor=(0.5, 0.0))
save(fig, 'Fig8')

LEV = {'S': ('#2B8C5A', 'Supported'), 'P': ('#A8D5B5', 'Partial'), 'N': ('#E9B3A8', 'Tested, not supported'), 'U': ('#E8E8E8', 'Not tested'), 'X': ('#F3D48A', 'Depends on intervention')}
rows = ['L24 attn., key-value\ngroup 7 (heads 28–31)', 'L24 attn.,\nother 28 heads', 'L22 MLP', 'L25 attention', 'L20 residual']
cols = ['Decodable', 'Sufficient', 'Necessary', 'Condition-\nspecific', 'Reused across\ntasks', 'Changed by\npost-training']
cells9 = [
    [('U', 'not tested at\nthe group level'), ('S', '99% of block\npatch; keys 87%'), ('S', '68% of\nrule margin'), ('P', 'format-dependent\nengagement;\nrobust across\ndonor protocols'), ('S', 'retrieval: 96%\nof head damage'), ('P', '≈3× necessity,\naltered weighting\nvs. base model\n(descriptive)')],
    [('U', '–'), ('N', '2% of whole-\nblock patch'), ('N', '−1% of\nrule margin'), ('U', '–'), ('N', 'no retrieval\ndamage'), ('U', '3 control heads:\nno change;\n25 not tracked')],
    [('P', f'probe AUC {ref["probe_lr_auc_L22mlp"]:.2f}\n(base model {base0["probe_lr_auc_L22mlp"]:.2f})'), ('P', 'moves 28–49%\nof error gap'), ('P', 'error − clean\n−0.44 nats'), ('S', 'clean effect\n≈ 0'), ('U', '–'), ('S', 'recruited; fastest\nwith true repairs')],
    [('P', f'probe AUC {ref["probe_lr_auc_L25attn"]:.2f}'), ('P', 'moves 38–50% of\ngap (opposite sign)'), ('P', 'error − clean\n+0.60 nats'), ('X', 'ablation: specific\nsteering: criterion'), ('U', '–'), ('N', 'steering effect\nunchanged')],
    [('P', f'probe AUC {ref["probe_lr_auc_L20"]:.2f}'), ('N', '≤ 14% of gap'), ('X', 'ablation: no effect\non C; projection\nremoval: small'), ('S', 'specific on E;\nsmall specific\neffect on C'), ('U', '–'), ('P', 'E effect recruited\nin arms B and C')]]
fig, ax = plt.subplots(figsize=(W, 2.9))
for i, row in enumerate(cells9):
    for j, (lv, txt) in enumerate(row):
        ax.add_patch(Rectangle((j, -i), 0.97, -0.94, color=LEV[lv][0], lw=0)); ax.text(j + 0.485, -i - 0.47, txt, ha='center', va='center', fontsize=5.4, linespacing=1.15)
for j, c in enumerate(cols): ax.text(j + 0.485, 0.12, c, ha='center', va='bottom', fontsize=6.2, fontweight='bold')
for i, r in enumerate(rows): ax.text(-0.06, -i - 0.47, r, ha='right', va='center', fontsize=6.1, fontweight='bold')
for k, (lv, (col, l)) in enumerate(LEV.items()):
    x0 = [-0.4, 0.95, 2.1, 3.85, 5.05][k]
    ax.add_patch(Rectangle((x0, -5.35), 0.2, -0.22, color=col, lw=0)); ax.text(x0 + 0.27, -5.46, l, va='center', fontsize=5.8)
ax.set_xlim(-1.9, 6.05); ax.set_ylim(-5.75, 0.75); ax.axis('off')
save(fig, 'Fig9')

fig = plt.figure(figsize=(6.64, 2.66))
ax = fig.add_axes([0.0, 0.0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.93, 'Causal profiling of a large language model (Qwen3-4B, its base model and 63 fine-tuning checkpoints)', ha='center', fontsize=8, fontweight='bold')
for x0, title, lines, col in [(0.015, 'One key/value group arbitrates evidence', ['layer-24 attention, heads 28–31', 'necessary: 68% of the rule margin', 'sufficient: 99% of the block patch', 'keys carry the evidence (87%)', 'reused by retrieval (96%)'], VERM),
                              (0.345, 'Post-training re-weights it', ['present and sufficient in base model', 'necessity about 3× higher', 'heads re-weighted within the group', 'engaged only in native prompt format'], BLUE),
                              (0.675, 'Verification is mostly a criterion', ['P(CHECK) = 1 on error and clean', 'small error component (AUC 0.72)', 'L22 MLP and L25 attention oppose', 'tuning keeps policy, not detection'], GREEN)]:
    ax.add_patch(FancyBboxPatch((x0, 0.06), 0.31, 0.78, boxstyle='round,pad=0.005,rounding_size=0.02', fc='white', ec=col, lw=1.2))
    ax.text(x0 + 0.155, 0.76, title, ha='center', fontsize=7.4, fontweight='bold', color=col)
    for i, t in enumerate(lines): ax.text(x0 + 0.02, 0.62 - i * 0.12, '• ' + t, fontsize=6.6)
fig.savefig(f'{OUT}/Graphical_abstract.pdf'); fig.savefig(f'{OUT}/Graphical_abstract.png', dpi=600)
Image.open(f'{OUT}/Graphical_abstract.png').convert('RGB').save(f'{OUT}/Graphical_abstract.tif', compression='tiff_lzw', dpi=(600, 600)); plt.close(fig)
print('done', sorted(os.listdir(OUT)))

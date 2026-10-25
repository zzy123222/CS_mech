import sys, os, json, argparse, time
ROOT = '.'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from src.hooks.core import first_tensor, replace_first_tensor, temporary_hooks
from src.interventions.hooks import make_patch_hook, make_mean_ablation_hook
from src.models.load import load_causal_lm
from src.models.scoring import score_answer_pairs
from src.utils.module_access import ModuleSpec, resolve_module, get_decoder_layers

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--fmt', required=True, choices=['raw', 'chat'])
ap.add_argument('--tag', required=True)
ap.add_argument('--n_groups', type=int, default=96)
ap.add_argument('--n_beh', type=int, default=300)
ap.add_argument('--n_sweep', type=int, default=48)
ap.add_argument('--dtype', default='bfloat16')
ap.add_argument('--retrieval', action='store_true', help='also run head/KV-group ablation on the retrieval task')
ap.add_argument('--out', default='results/p0/p0_2')
args = ap.parse_args()
OUT = os.path.join(args.out, args.tag); os.makedirs(OUT, exist_ok=True)

bundle = load_causal_lm(args.model, dtype=args.dtype, device_map='auto', chat=True)
tok, model = bundle.tokenizer, bundle.model
LAYER = 24
layers = get_decoder_layers(model)
attn = layers[LAYER].self_attn
oproj, kproj, vproj = attn.o_proj, attn.k_proj, attn.v_proj
NH, NKV, HD = model.config.num_attention_heads, model.config.num_key_value_heads, model.config.head_dim
GS = NH // NKV
blk = resolve_module(model, ModuleSpec(LAYER, 'attn'))


def prep(p):
    return bundle.prepare_prompt(p, enable_thinking=False) if args.fmt == 'chat' else p


def plen(p):
    return len(tok(p, add_special_tokens=False)['input_ids'])


def lp(prompt, answers, hooks=None, pre=None):
    hs = []
    if pre:
        hs = [m.register_forward_pre_hook(h) for m, h in pre]
    try:
        return score_answer_pairs(bundle, [prompt] * len(answers), answers, batch_size=len(answers), hooks=hooks)
    finally:
        for h in hs:
            h.remove()


def margin(prompt, r, s, hooks=None, pre=None):
    z = lp(prompt, [str(r), str(s)], hooks, pre)
    return float(z[0] - z[1])


def split_df(df, split, cond):
    return df[(df.split == split) & (df.condition == cond)].set_index('group_id')


def cap_pre(module, prompt, pos=-1):
    box = {}
    h = module.register_forward_pre_hook(lambda m, i: box.__setitem__('x', i[0].detach().float().cpu()))
    try:
        enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        model(**enc, use_cache=False)
    finally:
        h.remove()
    return box['x'][0]


def cap_out(module, prompt):
    box = {}
    h = module.register_forward_hook(lambda m, i, o: box.__setitem__('x', first_tensor(o).detach().float().cpu()))
    try:
        enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        model(**enc, use_cache=False)
    finally:
        h.remove()
    return box['x'][0]


def head_prehook(pos, replace):
    def f(_m, inp):
        x = inp[0].clone()
        xv = x.view(*x.shape[:-1], NH, HD)
        for h, v in replace.items():
            xv[:, pos, h, :] = v.to(x.device, x.dtype)
        return (x,) + tuple(inp[1:])
    return f


def kv_hook(dims, vec_all, pos=None):
    s, e = dims
    def f(_m, _i, out):
        o = first_tensor(out).clone()
        v = vec_all.to(o.device, o.dtype)
        if v.ndim == 1:
            o[:, :, s:e] = v
        else:
            T = v.shape[0]
            if T > o.shape[1]:
                return None
            o[:, :T, s:e] = v[:, s:e]
        return replace_first_tensor(out, o)
    return f


def head_prehook_all(donor_full, hs):
    def f(_m, inp):
        x = inp[0].clone()
        xv = x.view(*x.shape[:-1], NH, HD)
        d = donor_full.view(donor_full.shape[0], NH, HD).to(x.device, x.dtype)
        T = d.shape[0]
        for h in hs:
            xv[:, :T, h, :] = d[:, h, :]
        return (x,) + tuple(inp[1:])
    return f


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    b = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_task(df, task, do_patch=True, do_sweep=True):
    res = {}
    t0 = time.time()
    ev_split = 'test_ood'
    ro_e, cf_e = split_df(df, ev_split, 'R-only'), split_df(df, ev_split, 'R+S-conflict')
    has_cons = (df.condition == 'R+S-consistent').any()
    cs_e = split_df(df, ev_split, 'R+S-consistent') if has_cons else None
    ids = [i for i in cf_e.index if i in ro_e.index and (cs_e is None or i in cs_e.index)][:args.n_groups]
    ctl = split_df(df, 'train', 'R-only').index[:96]
    ro_t = split_df(df, 'train', 'R-only')
    ctl_prompts = [prep(ro_t.loc[g].prompt) for g in ctl]

    beh = []
    for cond in ['R-only', 'S-only', 'R+S-consistent', 'R+S-conflict', 'S-matched']:
        sub = df[(df.split == ev_split) & (df.condition == cond)].head(args.n_beh)
        for _, r in sub.iterrows():
            z = lp(prep(r.prompt), list('ABCD'))
            ci = 'ABCD'.index(str(r.correct_answer))
            ru, sh = 'ABCD'.index(str(r.rule_answer)), 'ABCD'.index(str(r.shortcut_answer))
            beh.append({'cond': cond, 'acc': float(np.argmax(z) == ci), 'margin': float(z[ru] - z[sh])})
    pd.DataFrame(beh).to_parquet(f'{OUT}/{task}_behavior.parquet')
    print(task, 'behaviour', round(time.time() - t0), 's', flush=True)

    blk_mean = torch.stack([cap_out(blk, p)[-1] for p in ctl_prompts]).mean(0)
    head_acts = torch.stack([cap_pre(oproj, p)[-1].view(NH, HD) for p in ctl_prompts])
    head_mean = head_acts.mean(0)
    k_all = torch.cat([cap_out(kproj, p) for p in ctl_prompts]).mean(0)
    v_all = torch.cat([cap_out(vproj, p) for p in ctl_prompts]).mean(0)

    rows = []
    cache = {}
    rng = np.random.default_rng(0)
    for gi, g in enumerate(tqdm(ids, desc=f'{task} eval', leave=False)):
        xp = prep(cf_e.loc[g].prompt); r, s = cf_e.loc[g].rule_answer, cf_e.loc[g].shortcut_answer
        pos = plen(xp) - 1
        rp = prep(ro_e.loc[g].prompt)
        m_base = margin(xp, r, s)
        m_ro = margin(rp, r, s)
        rows.append({'g': g, 'kind': 'base', 'key': 'conflict', 'v': m_base})
        rows.append({'g': g, 'kind': 'base', 'key': 'ronly', 'v': m_ro})
        m_ko = margin(xp, r, s, hooks=[(blk, make_mean_ablation_hook(blk_mean, pos))])
        rows.append({'g': g, 'kind': 'block_ko', 'key': 'L24', 'v': m_ko})
        for h in range(NH):
            m = margin(xp, r, s, pre=[(oproj, head_prehook(pos, {h: head_mean[h]}))])
            rows.append({'g': g, 'kind': 'head_ko', 'key': h, 'v': m})
        sets = {f'qgroup{k}': list(range(k * GS, (k + 1) * GS)) for k in range(NKV)}
        sets['H28_29_31'] = [28, 29, 31]; sets['H29_31'] = [29, 31]; sets['H30'] = [30]
        sets['all_but_grp7'] = [h for h in range(NH) if h // GS != 7]
        sets['all_heads'] = list(range(NH))
        for name, hs in sets.items():
            m = margin(xp, r, s, pre=[(oproj, head_prehook(pos, {h: head_mean[h] for h in hs}))])
            rows.append({'g': g, 'kind': 'set_ko', 'key': name, 'v': m})
        for k in range(NKV):
            sl = (k * HD, (k + 1) * HD)
            mk = margin(xp, r, s, hooks=[(kproj, kv_hook(sl, k_all[sl[0]:sl[1]]))])
            mv = margin(xp, r, s, hooks=[(vproj, kv_hook(sl, v_all[sl[0]:sl[1]]))])
            mkv = margin(xp, r, s, hooks=[(kproj, kv_hook(sl, k_all[sl[0]:sl[1]])), (vproj, kv_hook(sl, v_all[sl[0]:sl[1]]))])
            rows += [{'g': g, 'kind': 'k_ko', 'key': k, 'v': mk}, {'g': g, 'kind': 'v_ko', 'key': k, 'v': mv}, {'g': g, 'kind': 'kv_ko', 'key': k, 'v': mkv}]
        if do_patch and has_cons:
            cp = prep(cs_e.loc[g].prompt)
            assert plen(cp) == plen(xp)
            donor_blk_full = cap_out(blk, cp)
            donor_blk = donor_blk_full[-1]
            donor_head_full = cap_pre(oproj, cp)
            donor_head = donor_head_full[-1].view(NH, HD)
            m_clean = margin(cp, r, s)
            ro_blk = cap_out(blk, rp)[-1]
            ro_head = cap_pre(oproj, rp)[-1].view(NH, HD)
            m = margin(xp, r, s, hooks=[(blk, make_patch_hook(ro_blk, pos))]); rows.append({'g': g, 'kind': 'block_patch_ronly', 'key': 'matched', 'v': m})
            for name in ('H28_29_31', 'H29_31', 'H30', 'qgroup7', 'all_but_grp7', 'all_heads'):
                hs_ = sets[name]
                m = margin(xp, r, s, pre=[(oproj, head_prehook(pos, {h: ro_head[h] for h in hs_}))])
                rows.append({'g': g, 'kind': 'set_patch_ronly', 'key': name, 'v': m})
            m = margin(xp, r, s, hooks=[(blk, make_patch_hook(donor_blk_full, 0, all_positions=True))]); rows.append({'g': g, 'kind': 'block_patch_allpos', 'key': 'matched', 'v': m})
            for name in ('H28_29_31', 'qgroup7', 'all_but_grp7'):
                m = margin(xp, r, s, pre=[(oproj, head_prehook_all(donor_head_full, sets[name]))])
                rows.append({'g': g, 'kind': 'set_patch_allpos', 'key': name, 'v': m})
            rows.append({'g': g, 'kind': 'base', 'key': 'consistent', 'v': m_clean})
            m = margin(xp, r, s, hooks=[(blk, make_patch_hook(donor_blk, pos))]); rows.append({'g': g, 'kind': 'block_patch', 'key': 'matched', 'v': m})
            others = [o for o in ids if o != g]
            same = [o for o in others if cs_e.loc[o].rule_answer == r]
            for name, pool in (('wrong', others), ('same_answer', same or others)):
                o = pool[rng.integers(0, len(pool))]
                dd = cap_out(blk, prep(cs_e.loc[o].prompt))[-1]
                m = margin(xp, r, s, hooks=[(blk, make_patch_hook(dd, pos))]); rows.append({'g': g, 'kind': 'block_patch', 'key': name, 'v': m})
            perm = torch.randperm(donor_blk.numel(), generator=torch.Generator().manual_seed(gi))
            m = margin(xp, r, s, hooks=[(blk, make_patch_hook(donor_blk[perm], pos))]); rows.append({'g': g, 'kind': 'block_patch', 'key': 'shuffled', 'v': m})
            for name, hs in sets.items():
                if name.startswith('qgroup') and name != 'qgroup7':
                    continue
                m = margin(xp, r, s, pre=[(oproj, head_prehook(pos, {h: donor_head[h] for h in hs}))])
                rows.append({'g': g, 'kind': 'set_patch', 'key': name, 'v': m})
            kd, vd = cap_out(kproj, cp), cap_out(vproj, cp)
            for k in range(NKV):
                sl = (k * HD, (k + 1) * HD)
                mk = margin(xp, r, s, hooks=[(kproj, kv_hook(sl, kd))])
                mv = margin(xp, r, s, hooks=[(vproj, kv_hook(sl, vd))])
                mkv = margin(xp, r, s, hooks=[(kproj, kv_hook(sl, kd)), (vproj, kv_hook(sl, vd))])
                rows += [{'g': g, 'kind': 'k_patch', 'key': k, 'v': mk}, {'g': g, 'kind': 'v_patch', 'key': k, 'v': mv}, {'g': g, 'kind': 'kv_patch', 'key': k, 'v': mkv}]
        if gi % 16 == 15:
            pd.DataFrame(rows).assign(key=lambda d: d.key.astype(str)).to_parquet(f'{OUT}/{task}_main.parquet')
    pd.DataFrame(rows).assign(key=lambda d: d.key.astype(str)).to_parquet(f'{OUT}/{task}_main.parquet')
    print(task, 'main', round(time.time() - t0), 's', flush=True)

    if do_sweep:
        sw = []
        for g in tqdm(ids[:args.n_sweep], desc='sweep', leave=False):
            xp = prep(cf_e.loc[g].prompt); rp = prep(ro_e.loc[g].prompt); r, s = cf_e.loc[g].rule_answer, cf_e.loc[g].shortcut_answer
            pos_x = plen(xp) - 1
            caps = {}
            hs = [layers[l].register_forward_hook((lambda l_: (lambda m, i, o: caps.__setitem__(l_, first_tensor(o)[0, -1].detach().float().cpu())))(l)) for l in range(len(layers))]
            enc = tok(rp, return_tensors='pt', add_special_tokens=False).to(model.device); model(**enc, use_cache=False)
            for h in hs: h.remove()
            mc, mx = margin(rp, r, s), margin(xp, r, s)
            for l in range(len(layers)):
                m = margin(xp, r, s, hooks=[(layers[l], make_patch_hook(caps[l], pos_x))])
                sw.append({'g': g, 'layer': l, 'clean': mc, 'corrupt': mx, 'patched': m})
        pd.DataFrame(sw).to_parquet(f'{OUT}/{task}_sweep.parquet')
        print(task, 'sweep', round(time.time() - t0), 's', flush=True)


g_df = pd.read_parquet('data/generated/p1_graph_4b_seed1.parquet')
run_task(g_df, 'graph')
if args.retrieval:
    b = pd.read_parquet('data/generated/p3_task_battery.parquet')
    r_df = b[b.task_family == 'retrieval']
    run_task(r_df, 'retrieval', do_patch=False, do_sweep=False)
json.dump({'model': args.model, 'fmt': args.fmt, 'dtype': args.dtype, 'NH': NH, 'NKV': NKV, 'HD': HD, 'n_groups': args.n_groups}, open(f'{OUT}/meta.json', 'w'))
print('DONE', flush=True)

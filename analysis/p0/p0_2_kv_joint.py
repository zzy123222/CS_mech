import json
import numpy as np, pandas as pd

R = 'results/p0'
m = pd.read_parquet(f'{R}/p0_2/inst_chat/graph_main.parquet')
piv = lambda kind: m[m.kind == kind].pivot_table(index='g', columns='key', values='v')
conf = piv('base')['conflict']
gb = (piv('block_patch')['matched'] - conf).values
k, v, kv = [(piv(kind)['7'] - conf).values for kind in ('k_patch', 'v_patch', 'kv_patch')]
rng = np.random.default_rng(0); n = len(gb); ix = rng.integers(0, n, (3000, n))


def rm(x):
    b = x[ix].mean(1) / gb[ix].mean(1)
    return [float(x.mean() / gb.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


out = {'n_groups': int(n), 'key_only': rm(k), 'value_only': rm(v), 'key_plus_value': rm(kv), 'joint_minus_key_only': rm(kv - k),
       'joint_minus_sum_of_single': rm(kv - k - v)}
others = {f'{kind}_head{h}': float(((piv(kind)[str(h)] - conf).values).mean() / gb.mean()) for kind in ('v_patch', 'kv_patch') for h in range(7)}
top = max(others, key=lambda q: abs(others[q])); out['largest_outside_head7'] = {top: others[top]}
json.dump(out, open(f'{R}/p0_2_kv_joint.json', 'w'), indent=1)
for key, val in out.items(): print(key, val)

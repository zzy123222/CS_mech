from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.probes.linear_probe import fit_binary_probe, centroid_direction, projection_auc
from src.utils.io import save_table


CONTRASTS = {
    "conflict_state": ("R+S-conflict", "R+S-consistent"),
    "matched_s": ("R+S-conflict", "S-matched"),
    "rs_state": ("R-only", "S-only"),
}


def _lowdim_eval(X_tr, y_tr, X_te, y_te, seed):
    full = fit_binary_probe(X_tr, y_tr, X_te, y_te, seed=seed)
    v = centroid_direction(X_tr[y_tr == 1], X_tr[y_tr == 0])
    auc_1d = projection_auc(X_te, y_te, v)
    k = min(2, min(X_tr.shape[0], X_tr.shape[1]))
    pca = PCA(n_components=k, random_state=seed).fit(np.vstack([X_tr, X_te]))
    pca_clf = fit_binary_probe(pca.transform(X_tr), y_tr, pca.transform(X_te), y_te, seed=seed)
    return {"auc": full["auc"], "auc_1d": auc_1d, "auc_pca2": pca_clf["auc"],
            "lowdim_gap": float(full["auc"] - auc_1d)}, v, full["model"]


def _eval_layer(X, m, pos_label, neg_label, seed):
    y = (m.condition == pos_label).astype(int).values
    if "split" in m:
        tr = (m["split"] == "train").values
        te = (m["split"] == "test_ood").values
    else:
        tr = np.arange(len(m)) % 2 == 0
        te = ~tr
    if tr.sum() < 10 or te.sum() < 10:
        return None
    res, v, clf = _lowdim_eval(X[tr], y[tr], X[te], y[te], seed)
    res.update({"n_train": int(tr.sum()), "n_test": int(te.sum())})
    return res, v, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation_dir", required=True)
    ap.add_argument("--contrast", choices=list(CONTRASTS) + ["custom"], default="conflict_state",
                    help="Preset condition pair; 'custom' uses --positive/--negative. "
                         "conflict_state/matched_s are surface-matched and preferred over rs_state.")
    ap.add_argument("--positive", default="R-only")
    ap.add_argument("--negative", default="S-only")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_split", default="test_ood")
    ap.add_argument("--transfer_activation_dir", default=None,
                    help="Second activation dir (e.g., arithmetic battery): report probe transfer.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.contrast == "custom":
        pos_label, neg_label = args.positive, args.negative
    else:
        pos_label, neg_label = CONTRASTS[args.contrast]
    rows = []
    out_dir = Path(args.out or (Path(args.activation_dir).parent / "probe"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.activation_dir).glob("layer_*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        H = obj["H"].float().numpy()
        meta = pd.DataFrame(obj["meta"])
        lid = int(obj["layer"])
        mask = meta.condition.isin([pos_label, neg_label])
        if mask.sum() == 0:
            continue
        X = H[mask.values]
        m = meta.loc[mask].reset_index(drop=True)
        ev = _eval_layer(X, m, pos_label, neg_label, args.seed)
        if ev is None:
            continue
        res, v, clf = ev
        np.save(out_dir / f"{args.contrast}_direction_layer_{lid:03d}.npy", v.astype(np.float32))
        if args.contrast == "rs_state":
            np.save(out_dir / f"reasoning_direction_layer_{lid:03d}.npy", v.astype(np.float32))
        rows.append({"layer": lid, "contrast": args.contrast, "positive": pos_label,
                     "negative": neg_label, **res})
    df = pd.DataFrame(rows).sort_values("layer")
    save_table(df, out_dir / "probe_summary.csv")
    print(df.to_string(index=False))

    if args.transfer_activation_dir:
        trows = []
        for path in sorted(Path(args.transfer_activation_dir).glob("layer_*.pt")):
            obj = torch.load(path, map_location="cpu", weights_only=False)
            H = obj["H"].float().numpy()
            meta = pd.DataFrame(obj["meta"])
            lid = int(obj["layer"])
            mask = meta.condition.isin([pos_label, neg_label])
            if mask.sum() == 0:
                continue
            X = H[mask.values]
            m = meta.loc[mask].reset_index(drop=True)
            y = (m.condition == pos_label).astype(int).values
            v_path = out_dir / f"{args.contrast}_direction_layer_{lid:03d}.npy"
            s_path = sorted(Path(args.activation_dir).glob(f"layer_{lid:03d}.pt"))
            if not v_path.exists() or not s_path:
                continue
            v = np.load(v_path)
            trows.append({"layer": lid, "auc_transfer_1d": projection_auc(X, y, v)})
            src = torch.load(s_path[0], map_location="cpu", weights_only=False)
            sH = src["H"].float().numpy()
            sm = pd.DataFrame(src["meta"])
            smask = sm.condition.isin([pos_label, neg_label])
            sev = _eval_layer(sH[smask.values], sm.loc[smask].reset_index(drop=True),
                              pos_label, neg_label, args.seed)
            if sev is not None:
                _, _, clf = sev
                trows[-1]["auc_transfer_full"] = float(
                    roc_auc_like(clf, X, y))
        if trows:
            tdf = pd.DataFrame(trows).sort_values("layer")
            save_table(tdf, out_dir / "probe_transfer_summary.csv")
            print("\ntransfer:\n", tdf.to_string(index=False))


def roc_auc_like(clf, X, y):
    from sklearn.metrics import roc_auc_score
    prob = clf.predict_proba(X)[:, 1]
    return roc_auc_score(np.asarray(y), prob)


if __name__ == "__main__":
    main()

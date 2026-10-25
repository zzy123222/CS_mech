from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "posttraining/results"
FORMAL = RES / "formal"
OUT = FORMAL / "analysis"

ARMS = [("general_sft_seed1", "A: General"),
        ("correction_sft_seed1", "B: Correction"),
        ("sham_correction_seed1", "C: Sham")]
for _s in (2, 3):
    for _d, _l in [("general_sft", "A"), ("correction_sft", "B"), ("sham_correction", "C")]:
        if (FORMAL / f"{_d}_seed{_s}" / "mechanism_panel.csv").exists():
            ARMS.append((f"{_d}_seed{_s}", f"{_l}{_s}"))

MECHS = [("monitor_d_L20", "monitor", "Monitor (L20)"),
         ("controller_d_L22mlp", "controller", "Controller (L22:mlp)"),
         ("integrator_d_L24attn", "integrator", "Integrator (L24:attn)"),
         ("opponent_d_L25attn", "opponent", "Opponent (L25:attn)")]

GATES = [("gate_graph_ronly_acc", "Graph R-only acc"),
         ("gate_retrieval_ronly_acc", "Retrieval R-only acc"),
         ("gate_p2_error_acc", "Error-detect acc")]


def boot_idx(n, n_boot=10000, seed=777):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def ratio_bootstrap(ckpt_samples, base_samples, ref_samples, n_boot=10000, seed=777):
    ck, ba, rf = (np.asarray(x, float) for x in (ckpt_samples, base_samples, ref_samples))
    i_ck, i_ba, i_rf = boot_idx(len(ck), n_boot, seed), boot_idx(len(ba), n_boot, seed + 1), \
        boot_idx(len(rf), n_boot, seed + 2)
    num = ck[i_ck].mean(1) - ba[i_ba].mean(1)
    den = rf[i_rf].mean(1) - ba[i_ba].mean(1)
    D_dist = np.full(n_boot, np.nan)
    keep = np.abs(den) > 1e-9
    D_dist[keep] = num[keep] / den[keep]
    den_mean = den.mean()
    D = num.mean() / den_mean if abs(den_mean) > 1e-9 else np.nan
    valid = D_dist[~np.isnan(D_dist)]
    lo = float(np.percentile(valid, 2.5)) if len(valid) else np.nan
    hi = float(np.percentile(valid, 97.5)) if len(valid) else np.nan
    return float(D), lo, hi, float(rf.mean() - ba.mean())


def ckpt_samples(npz, step, tag):
    if tag == "integrator":
        return np.concatenate([npz[f"step{step:04d}_integrator_graph_per_sample"],
                               npz[f"step{step:04d}_integrator_retrieval_per_sample"]])
    return npz[f"step{step:04d}_{tag}_per_sample"]


def ref_samples(npz, tag):
    if tag == "integrator":
        return np.concatenate([npz["integrator_graph_per_sample"],
                               npz["integrator_retrieval_per_sample"]])
    return npz[f"{tag}_per_sample"]


def stable_emergence(steps, means, ci_lows, ci_highs, base_level, base_ci):
    n = len(steps)

    def ci_excludes(i):
        return (ci_highs[i] < base_ci[0]) or (ci_lows[i] > base_ci[1])

    for i in range(n):
        if not ci_excludes(i) or np.sign(means[i] - base_level) == 0:
            continue
        direction = np.sign(means[i] - base_level)
        nxt = list(range(i + 1, min(i + 3, n)))
        if len(nxt) >= 2:
            if all(np.sign(means[j] - base_level) == direction for j in nxt):
                return int(steps[i]), "strict"
    for i in range(n - 1):
        if (ci_excludes(i) and ci_excludes(i + 1)
                and np.sign(means[i] - base_level) == np.sign(means[i + 1] - base_level) != 0):
            return int(steps[i]), "fallback-2consec"
    return None, "not-emerged"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.load(open(RES / "reference_metrics.json"))
    ref_npz = np.load(RES / "reference_metrics.npz")

    panels = {}
    for arm_dir, label in ARMS:
        p = FORMAL / arm_dir / "mechanism_panel.csv"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        panels[arm_dir] = (pd.read_csv(p), np.load(FORMAL / arm_dir / "mechanism_panel_persample.npz"))
    if not panels:
        raise SystemExit("no panel results found")

    first_arm = next(iter(panels))
    df0, npz0 = panels[first_arm]
    base_row = df0[df0.step == 0].iloc[0]

    rows = []
    for arm_dir, label in ARMS:
        if arm_dir not in panels:
            continue
        df, npz = panels[arm_dir]
        for key, tag, mech in MECHS:
            rf_s = ref_samples(ref_npz, tag)
            ba_s = ckpt_samples(npz0, 0, tag)
            for _, r in df.iterrows():
                step = int(r.step)
                D, lo, hi, ref_span = ratio_bootstrap(ckpt_samples(npz, step, tag), ba_s, rf_s)
                rows.append({
                    "arm": label, "arm_dir": arm_dir, "mechanism": mech, "key": key,
                    "step": step, "raw": r[key],
                    "raw_ci_low": r[f"{key}_ci_low"], "raw_ci_high": r[f"{key}_ci_high"],
                    "D": D, "D_ci_low": lo, "D_ci_high": hi, "ref_span": ref_span,
                })
        for gkey, gname in GATES:
            for _, r in df.iterrows():
                rows.append({"arm": label, "arm_dir": arm_dir, "mechanism": gname,
                             "key": gkey, "step": int(r.step), "raw": r[gkey]})

    dev = pd.DataFrame(rows)
    dev.to_csv(OUT / "development_index.csv", index=False)

    summary = {
        "reference": {k: ref[k] for k in
                      ["monitor_d_L20", "controller_d_L22mlp", "integrator_d_L24attn",
                       "opponent_d_L25attn", "gate_graph_ronly_acc",
                       "gate_retrieval_ronly_acc", "gate_p2_error_acc"]},
        "reference_note": ("Monitor/Opponent reference CIs cross zero in this panel "
                           "(margin proxy has low sensitivity for these mechanisms); "
                           "their Development Index denominators are unstable - treat as "
                           "descriptive only."),
        "emergence": {}, "arm_contrast": {},
    }

    for arm_dir, label in ARMS:
        if arm_dir not in panels:
            continue
        df, _ = panels[arm_dir]
        sub = df.sort_values("step")
        for key, tag, mech in MECHS:
            t_star, mode = stable_emergence(
                sub.step.values, sub[key].values,
                sub[f"{key}_ci_low"].values, sub[f"{key}_ci_high"].values,
                base_row[key],
                (base_row[f"{key}_ci_low"], base_row[f"{key}_ci_high"]))
            summary["emergence"][f"{label} | {mech}"] = {
                "t_star": t_star, "mode": mode,
                "base": float(base_row[key]), "final": float(sub[key].values[-1]),
                "instruct_ref": float(ref[key]),
            }

    for key, tag, mech in MECHS:
        piv = {}
        for arm_dir, label in ARMS:
            if arm_dir not in panels:
                continue
            piv[label] = panels[arm_dir][0].set_index("step")[key]
        p = pd.DataFrame(piv)
        final = p.iloc[-1]
        summary["arm_contrast"][mech] = {
            "final_step": int(p.index[-1]),
            "final_values": {k: (None if pd.isna(v) else float(v)) for k, v in final.items()},
            "B_minus_A_final": float(final.get("B: Correction", np.nan) - final.get("A: General", np.nan))
                if "B: Correction" in p and "A: General" in p else None,
            "B_minus_C_final": float(final.get("B: Correction", np.nan) - final.get("C: Sham", np.nan))
                if "B: Correction" in p and "C: Sham" in p else None,
        }

    with open(OUT / "emergence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fam = {"A": "#7f8c8d", "B": "#c0392b", "C": "#2980b9"}

        def style(label):
            fam_key = label[0]
            if label in ("A: General", "B: Correction", "C: Sham"):
                return fam[fam_key], "-"
            return fam[fam_key], "--"

        fig, axes = plt.subplots(2, 4, figsize=(19, 9))
        for ax, (key, tag, mech) in zip(axes.flat, MECHS):
            for arm_dir, label in ARMS:
                if arm_dir not in panels:
                    continue
                sub = panels[arm_dir][0].sort_values("step")
                c, ls = style(label)
                ax.plot(sub.step, sub[key], ls, marker="o", ms=3, color=c,
                        label=label, alpha=0.55 if ls == "--" else 1.0)
                ax.fill_between(sub.step, sub[f"{key}_ci_low"], sub[f"{key}_ci_high"],
                                color=c, alpha=0.15 if ls != "--" else 0.08)
            ax.axhline(ref[key], color="black", ls="--", lw=1, label="Instruct ref")
            ax.set_title(mech, fontsize=11)
            ax.set_xlabel("SFT step")
        for ax, (gkey, gname) in zip(axes.flat[4:7], GATES):
            for arm_dir, label in ARMS:
                if arm_dir not in panels:
                    continue
                sub = panels[arm_dir][0].sort_values("step")
                c, ls = style(label)
                ax.plot(sub.step, sub[gkey], ls, marker="o", ms=3, color=c,
                        alpha=0.55 if ls == "--" else 1.0)
            ax.axhline(0.65, color="black", ls=":", lw=1, label="gate threshold")
            ax.axhline(ref.get(gkey, 1.0), color="black", ls="--", lw=1)
            ax.set_title(gname, fontsize=11)
            ax.set_xlabel("SFT step")
            ax.set_ylim(-0.05, 1.05)
        axes.flat[7].axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Mechanism development across 1000 SFT steps "
                     "(Qwen3-4B-Base, three arms, n=64, 95% paired-bootstrap CI)", y=1.00)
        fig.tight_layout()
        fig.savefig(OUT / "development_curves.png", dpi=150, bbox_inches="tight")
        fig.savefig(OUT / "development_curves.pdf", format="pdf", bbox_inches="tight")
        print(f"figure saved: {OUT / 'development_curves.png'}")
    except Exception as e:
        print(f"[warn] figure failed: {e}")


if __name__ == "__main__":
    main()

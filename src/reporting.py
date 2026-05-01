"""
CHITIN · src/reporting.py  (v2)
────────────────────────────────
Generates a plain-text summary report of a completed CHITIN v2 run.
Saved to output/reports/{dataset_name}_CHITIN_report.txt

New in v2:
  - Section 0: Calibration summary (auto vs manual, selected k/n_pcs/metric,
               sweep statistics)
  - Correction mode displayed throughout
  - Section 7: PC decomposition summary (if applicable)
  - Section numbering shifted +1 from v1 (0 is new calibration section)
  - chitin_model parameter required to extract calibration/correction info
  - Sweep results CSV saved alongside report if save_sweep_results: true
"""

import numpy as np
from datetime import datetime
from pathlib import Path


_RHO_STRONG    = 0.80
_RHO_MODERATE  = 0.92
_DISC_STRONG   =  20.0
_DISC_MODERATE =   0.0
_COS_HIGH      =  0.40
_COS_MED       =  0.20


def _rho_verdict(r):
    if r < _RHO_STRONG:
        return "STRONG — rank order substantially reshuffled"
    elif r < _RHO_MODERATE:
        return "MODERATE — partial rank disruption; some tree splits will differ"
    return "WEAK — ranks largely preserved; limited effect on LightGBM"


def _disc_verdict(pct):
    if pct > _DISC_STRONG:
        return "STRONG IMPROVEMENT — perturbation centroids significantly more separable"
    elif pct > _DISC_MODERATE:
        return "MILD IMPROVEMENT — slight increase in perturbation separability"
    return (f"DECREASED ({pct:+.1f}%) — perturbations less separable post-CHITIN. "
            f"Common in homogeneous cell lines where pre-CHITIN separability "
            f"was inflated by systematic variation.")


def _cos_verdict(c):
    if c > _COS_HIGH:
        return f"HIGH ({c:.3f}) — strong systematic variation; correction warranted"
    elif c > _COS_MED:
        return f"MODERATE ({c:.3f}) — some systematic variation"
    return f"LOW ({c:.3f}) — minimal systematic variation"


def generate_report(
    cfg,
    mean_rho,
    dist_pre,
    dist_post,
    cos_ctrl_pre,
    V_pre,
    delta_df,
    delta_norms,
    adata_train_delta,
    adata_val_delta,
    adata_test_delta,
    chitin_model,
):
    dataset   = cfg["dataset"]["name"]
    pert_col  = cfg["dataset"]["perturbation_col"]
    ctrl_lbl  = cfg["dataset"]["control_label"]

    k         = chitin_model.k
    n_pcs     = chitin_model.n_pcs
    metric    = chitin_model.distance_metric
    mode      = chitin_model.correction_mode
    auto_cal  = cfg.get("calibration", {}).get("auto_calibrate", True)
    sel       = chitin_model.selected_params
    sys_pcs   = chitin_model.systematic_pc_indices

    disc_pct  = (dist_post.mean() - dist_pre.mean()) / dist_pre.mean() * 100
    v_mag     = float(np.linalg.norm(V_pre))
    mean_cos  = float(cos_ctrl_pre.mean())
    std_cos   = float(cos_ctrl_pre.std())

    n_train = adata_train_delta.n_obs
    n_val   = adata_val_delta.n_obs
    n_test  = adata_test_delta.n_obs
    n_genes = adata_train_delta.n_vars
    n_total = n_train + n_val + n_test

    pert_obs  = adata_train_delta.obs[pert_col]
    n_perts   = pert_obs[pert_obs != ctrl_lbl].nunique()
    n_ctrl_mc = int((pert_obs == ctrl_lbl).sum())

    mean_d  = float(delta_norms.mean())
    std_d   = float(delta_norms.std())
    min_d   = float(delta_norms.min())
    max_d   = float(delta_norms.max())
    top5    = delta_df.head(5)[["perturbation","mean_delta_norm"]].values.tolist()
    bot5    = delta_df.tail(5)[["perturbation","mean_delta_norm"]].values.tolist()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    W  = 72
    lines = []

    def h(title=""):
        lines.append("═" * W)
        if title:
            lines.append(f"  {title}")
            lines.append("═" * W)

    def s(text=""):
        lines.append(text)

    h(f"CHITIN v2 RUN REPORT — {dataset}")
    s(f"  Generated   : {ts}")
    s(f"  Dataset     : {dataset}")
    s(f"  Pert col    : {pert_col}   |   Control: {ctrl_lbl}")
    s(f"  Correction  : {mode}")
    s()

    # 0. Calibration
    h("0 · CALIBRATION")
    if auto_cal and sel is not None:
        n_sweep  = len(chitin_model.sweep_results)  \
            if chitin_model.sweep_results is not None else "?"
        n_pareto = len(chitin_model.pareto_front) \
            if chitin_model.pareto_front is not None else "?"
        s(f"  Strategy         : AUTO-CALIBRATE (Pareto sweep)")
        s(f"  Combinations     : {n_sweep}")
        s(f"  Pareto front     : {n_pareto} points")
        s(f"  Selected k       : {k}")
        s(f"  Selected n_pcs   : {n_pcs}")
        s(f"  Selected metric  : {metric}")
        if chitin_model.sweep_results is not None:
            df    = chitin_model.sweep_results
            match = df[(df["k"] == k) &
                       (df["n_pcs"] == n_pcs) &
                       (df["metric"] == metric)]
            if len(match) > 0:
                row = match.iloc[0]
                s(f"  At selected point:")
                s(f"    rank_disruption  : {row['rank_disruption']:.4f}")
                s(f"    disc_ratio       : {row['disc_ratio']:.4f}")
                s(f"    signal_stability : {row['signal_stability']:.4f}")
    else:
        s(f"  Strategy  : MANUAL (auto_calibrate: false)")
        s(f"  k         : {k}")
        s(f"  n_pcs     : {n_pcs}")
        s(f"  metric    : {metric}")
    s()

    # 1. Dataset size
    h("1 · DATASET SIZE")
    s(f"  Genes              : {n_genes:,}")
    s(f"  Perturbations      : {n_perts:,}  (excl. controls)")
    s(f"  Control metacells  : {n_ctrl_mc:,}  (train)")
    s()
    s(f"  {'Split':<10}  {'Metacells':>12}")
    s(f"  {'─'*10}  {'─'*12}")
    s(f"  {'Train':<10}  {n_train:>12,}")
    s(f"  {'Val':<10}  {n_val:>12,}")
    s(f"  {'Test':<10}  {n_test:>12,}")
    s(f"  {'TOTAL':<10}  {n_total:>12,}")
    s()

    # 2. Systematic variation
    h("2 · SYSTEMATIC VARIATION (PRE-CHITIN)")
    s(f"  |V|                : {v_mag:.4f}")
    s(f"  Mean cosine to V   : {mean_cos:.4f} ± {std_cos:.4f}")
    s(f"  Verdict : {_cos_verdict(mean_cos)}")
    s()

    # 3. Rank disruption
    h("3 · RANK-ORDER DISRUPTION (KEY METRIC FOR LIGHTGBM)")
    s(f"  Mean Spearman ρ    : {mean_rho:.4f}")
    s(f"  Verdict : {_rho_verdict(mean_rho)}")
    s()
    if mean_rho >= _RHO_MODERATE and mode == "knn":
        s("  ⚠ KNN correction had limited effect on rank order.")
        s("  → Consider: correction.mode: 'pc_decomposition' for homogeneous")
        s("    cell lines (iPSC, hESC) where KNN localisation fails.")
    s()

    # 4. Pairwise discrimination
    h("4 · PAIRWISE PERTURBATION DISCRIMINATION")
    s(f"  Pre-CHITIN  : {dist_pre.mean():.4f} ± {dist_pre.std():.4f}")
    s(f"  Post-CHITIN : {dist_post.mean():.4f} ± {dist_post.std():.4f}")
    s(f"  Change      : {disc_pct:+.1f}%")
    s(f"  Verdict : {_disc_verdict(disc_pct)}")
    s()

    # 5. Delta magnitude
    h("5 · DELTA MAGNITUDE DISTRIBUTION")
    s(f"  Mean |Δ| : {mean_d:.4f}   Std: {std_d:.4f}")
    s(f"  Min  |Δ| : {min_d:.4f}   Max: {max_d:.4f}")
    s()
    s("  Top 5 perturbations by causal signal strength:")
    for p, v in top5:
        s(f"    {str(p):<30}  |Δ| = {v:.4f}")
    s()
    s("  Bottom 5 perturbations (weakest causal signal):")
    for p, v in bot5:
        s(f"    {str(p):<30}  |Δ| = {v:.4f}")
    s()

    # 6. Output files
    h("6 · OUTPUT FILES")
    suffix  = cfg["output"]["suffix"]
    out_dir = cfg["paths"]["_output"]
    for split in ["train", "val", "test"]:
        s(f"  {split:<6} → {out_dir / f'{dataset}_{split}{suffix}.h5ad'}")
    s()

    # 7. PC decomposition (only when applicable)
    if mode in ("pc_decomposition", "hybrid"):
        h("7 · PC DECOMPOSITION SUMMARY")
        if len(sys_pcs) > 0:
            s(f"  PCs projected out : {[f'PC{i+1}' for i in sys_pcs]}")
            if chitin_model.V_systematic_cos is not None:
                for i in sys_pcs:
                    c = float(chitin_model.V_systematic_cos[i])
                    s(f"    PC{i+1}: cos_sim to V = {c:.4f}")
        else:
            s("  No PCs classified as systematic above threshold.")
            s("  Increase pc_systematic_threshold or verify V is non-zero.")
        s()

    # 8. Overall assessment
    h("8 · OVERALL ASSESSMENT")
    score = sum([
        mean_rho  < _RHO_MODERATE,
        disc_pct  > _DISC_MODERATE,
        mean_cos  > _COS_MED,
    ])
    if score == 3:
        verdict = "EFFECTIVE — CHITIN substantially improved the dataset."
    elif score == 2:
        verdict = "PARTIALLY EFFECTIVE — moderate improvement; GRN may benefit."
    elif score == 1:
        verdict = "LIMITED EFFECT — minor changes. Review correction mode."
    else:
        verdict = "MINIMAL EFFECT — CHITIN did not substantially alter this dataset."

    s(f"  Score   : {score}/3")
    s(f"  Verdict : {verdict}")
    s()
    s("  Metric summary:")
    s(f"  {'✓' if mean_rho < _RHO_MODERATE else '⚠'}  "
      f"Rank disruption      ρ={mean_rho:.4f}  (target < {_RHO_MODERATE})")
    s(f"  {'✓' if disc_pct > _DISC_MODERATE else '⚠'}  "
      f"Discrimination       {disc_pct:+.1f}%  (target > 0%)")
    s(f"  {'✓' if mean_cos > _COS_MED else '~'}  "
      f"Systematic variation cos={mean_cos:.4f}  (> {_COS_MED} = needed)")
    s()
    h()

    report_text = "\n".join(lines)

    report_dir  = Path(cfg["paths"]["_output"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{dataset}_CHITIN_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    if cfg.get("output", {}).get("save_sweep_results", True):
        if chitin_model.sweep_results is not None and \
                len(chitin_model.sweep_results) > 0:
            sweep_path = report_dir / f"{dataset}_sweep_results.csv"
            chitin_model.sweep_results.to_csv(sweep_path, index=False)

    print(report_text)
    print(f"\n  Report saved → {report_path}")
    return report_path

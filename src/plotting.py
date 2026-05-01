"""
CHITIN · src/plotting.py  (v2)
───────────────────────────────
Publication-quality diagnostic visualizations for CHITIN.

New in v2:
  plot_pareto_sweep()          — rank_disruption vs disc_ratio scatter,
                                  coloured by signal_stability, selected
                                  point starred. One panel per metric.
  plot_pc_systematic_cosines() — bar chart of cosine similarity to V per PC,
                                  threshold line, systematic PCs coloured red.

All v1 plots preserved with identical signatures.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from .utils import get_theme, save_fig, format_ax


_MAX_SCATTER = 50_000


# ═══════════════════════════════════════════════════════════════════════════
#  PCA & LATENT SPACE  (v1 unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def plot_pca_variance(adata, cfg: dict):
    """Scree plot of PCA variance explained."""
    theme = get_theme(cfg)
    if "chitin_pca_variance_ratio" not in adata.uns:
        return

    var_ratio  = adata.uns["chitin_pca_variance_ratio"]
    cumulative = np.cumsum(var_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(var_ratio)), var_ratio,
            color=theme["accent"], alpha=0.7)
    format_ax(ax1, theme, "Individual Variance Explained",
              xlabel="PC", ylabel="Variance Ratio")

    ax2.plot(range(len(cumulative)), cumulative,
             color=theme["accent"], lw=2)
    ax2.axhline(0.9, color=theme["warn"], ls="--", lw=1, label="90%")
    ax2.legend(fontsize=10)
    format_ax(ax2, theme, "Cumulative Variance Explained",
              xlabel="PC", ylabel="Cumulative Ratio")

    fig.suptitle("CHITIN · PCA Variance (Metacell Latent Space)",
                 fontsize=16, fontweight="bold",
                 color=theme["text"], y=1.02)
    fig.tight_layout()
    save_fig(fig, cfg, "chitin_pca_variance")
    plt.show()


def plot_latent_space(adata, cfg: dict):
    """2D scatter of the first two PCA components, ctrl vs pert."""
    theme    = get_theme(cfg)
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    if "X_pca_chitin" not in adata.obsm:
        return

    pca    = adata.obsm["X_pca_chitin"]
    labels = adata.obs[pert_col].values
    is_ctrl = labels == ctrl_label

    fig, ax = plt.subplots(figsize=(8, 8))

    n = len(labels)
    if n > _MAX_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=_MAX_SCATTER, replace=False)
    else:
        idx = np.arange(n)

    ctrl_idx_ = idx[is_ctrl[idx]]
    pert_idx_ = idx[~is_ctrl[idx]]

    ax.scatter(pca[pert_idx_, 0], pca[pert_idx_, 1],
               c=theme["pert"], s=3, alpha=0.2,
               label="Perturbed", rasterized=True)
    ax.scatter(pca[ctrl_idx_, 0], pca[ctrl_idx_, 1],
               c=theme["ctrl"], s=6, alpha=0.5,
               label="Control", rasterized=True)

    ax.legend(fontsize=11, markerscale=4)
    format_ax(ax, theme, "CHITIN · Metacell Latent Space (PCA)",
              xlabel="PC1", ylabel="PC2")
    fig.tight_layout()
    save_fig(fig, cfg, "chitin_latent_space")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEMA COSINE ANALYSIS  (v1 unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def plot_cosine_distributions(cosines_ctrl, cosines_pert, cfg: dict,
                               label: str = "pre"):
    theme = get_theme(cfg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(cosines_ctrl, bins=50, color=theme["ctrl"],
             alpha=0.7, edgecolor=theme["panel"])
    ax1.axvline(cosines_ctrl.mean(), color=theme["warn"],
                ls="--", lw=2,
                label=f"Mean: {cosines_ctrl.mean():.3f}")
    ax1.legend(fontsize=10)
    format_ax(ax1, theme, f"Control Reference ({label.title()})",
              xlabel="Cosine Similarity", ylabel="Count")

    ax2.hist(cosines_pert, bins=50, color=theme["delta"],
             alpha=0.7, edgecolor=theme["panel"])
    ax2.axvline(cosines_pert.mean(), color=theme["warn"],
                ls="--", lw=2,
                label=f"Mean: {cosines_pert.mean():.3f}")
    ax2.legend(fontsize=10)
    format_ax(ax2, theme, f"Perturbed Reference ({label.title()})",
              xlabel="Cosine Similarity", ylabel="Count")

    fig.suptitle(f"CHITIN · Systematic Variation ({label.title()}-CHITIN)",
                 fontsize=16, fontweight="bold",
                 color=theme["text"], y=1.02)
    fig.tight_layout()
    save_fig(fig, cfg, f"chitin_cosines_{label}")
    plt.show()


def plot_cosine_before_after(cos_ctrl_pre, cos_ctrl_post, cfg: dict):
    theme = get_theme(cfg)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(cos_ctrl_pre, bins=50, color=theme["muted"],
            alpha=0.5,
            label=f"Pre-CHITIN (mean={cos_ctrl_pre.mean():.3f})")
    ax.hist(cos_ctrl_post, bins=50, color=theme["delta"],
            alpha=0.6,
            label=f"Post-CHITIN (mean={cos_ctrl_post.mean():.3f})")
    ax.axvline(cos_ctrl_pre.mean(),  color=theme["muted"],  ls="--", lw=2)
    ax.axvline(cos_ctrl_post.mean(), color=theme["delta"], ls="--", lw=2)

    ax.legend(fontsize=11)
    format_ax(ax, theme,
              "CHITIN · Systematic Variation Reduction",
              xlabel="Cosine Similarity (Control Reference)",
              ylabel="Number of Perturbations")
    fig.tight_layout()
    save_fig(fig, cfg, "chitin_cosine_before_after")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  K-SENSITIVITY  (v1 unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def plot_k_sensitivity(k_values, basal_variances, mean_distances,
                        chosen_k: int, cfg: dict):
    theme = get_theme(cfg)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(k_values, basal_variances, "o-",
                  color=theme["accent"], lw=2, markersize=6,
                  label="Basal Variance (localisation)")
    l2 = ax2.plot(k_values, mean_distances,  "s--",
                  color=theme["highlight"], lw=2, markersize=6,
                  label="Mean KNN Distance")

    if chosen_k in k_values:
        ax1.axvline(chosen_k, color=theme["warn"],
                    ls=":", lw=2, alpha=0.8,
                    label=f"Chosen k={chosen_k}")

    ax1.set_xlabel("k (number of neighbors)", color=theme["text"])
    ax1.set_ylabel("Basal Vector Variance",   color=theme["accent"])
    ax2.set_ylabel("Mean KNN Distance",        color=theme["highlight"])

    lines  = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc="center right")
    ax1.set_title("CHITIN · k-Sensitivity Analysis (diagnostic)",
                  fontsize=14, fontweight="bold",
                  color=theme["text"], pad=12)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, cfg, "chitin_k_sensitivity")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  NEW v2: PARETO SWEEP PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto_sweep(chitin_model, cfg: dict):
    """
    Visualise the full Pareto sweep results.

    Layout: one column per distance_metric (euclidean / cosine).
    Each panel: scatter of rank_disruption (x) vs disc_ratio (y),
    points coloured by signal_stability, size proportional to n_pcs,
    Pareto-front points outlined, selected point starred in gold.
    """
    if chitin_model.sweep_results is None or len(chitin_model.sweep_results) == 0:
        print("No sweep results to plot (auto_calibrate was False).")
        return

    theme   = get_theme(cfg)
    df      = chitin_model.sweep_results.copy()
    pareto  = chitin_model.pareto_front
    sel     = chitin_model.selected_params
    metrics = df["metric"].unique().tolist()

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(7 * len(metrics), 6),
                             squeeze=False)

    # Normalise signal_stability to [0, 1] for colour mapping
    stab_min = df["signal_stability"].min()
    stab_max = df["signal_stability"].max()
    stab_range = max(stab_max - stab_min, 1e-10)

    # n_pcs → marker size
    npcs_vals  = sorted(df["n_pcs"].unique())
    size_map   = {v: 40 + i * 25 for i, v in enumerate(npcs_vals)}

    for col, metric in enumerate(metrics):
        ax  = axes[0][col]
        sub = df[df["metric"] == metric].copy()

        norm_stab = (sub["signal_stability"] - stab_min) / stab_range
        sizes     = [size_map.get(int(v), 60) for v in sub["n_pcs"]]
        colours   = plt.cm.plasma(norm_stab.values)

        sc = ax.scatter(sub["rank_disruption"], sub["disc_ratio"],
                        c=norm_stab, cmap="plasma",
                        s=sizes, alpha=0.7,
                        edgecolors=theme["grid"], linewidths=0.4,
                        zorder=2)

        # Outline Pareto-front points
        if pareto is not None:
            sub_p = pareto[pareto["metric"] == metric]
            ax.scatter(sub_p["rank_disruption"], sub_p["disc_ratio"],
                       s=[size_map.get(int(v), 60) for v in sub_p["n_pcs"]],
                       facecolors="none",
                       edgecolors=theme["good"], linewidths=2.0,
                       zorder=3, label="Pareto front")

        # Star the selected point
        if sel and sel.get("metric") == metric:
            sel_row = sub[
                (sub["k"] == sel["k"]) &
                (sub["n_pcs"] == sel["n_pcs"])
            ]
            if len(sel_row) > 0:
                ax.scatter(sel_row["rank_disruption"],
                           sel_row["disc_ratio"],
                           marker="*", s=350,
                           c=theme["warn"],
                           zorder=5,
                           label=f"Selected k={sel['k']} n_pcs={sel['n_pcs']}")

        # Annotate a few representative k values
        for _, row in sub.iterrows():
            if int(row["n_pcs"]) == npcs_vals[len(npcs_vals) // 2]:
                ax.annotate(f"k={int(row['k'])}",
                            xy=(row["rank_disruption"], row["disc_ratio"]),
                            xytext=(3, 3), textcoords="offset points",
                            fontsize=6, color=theme["muted"], alpha=0.8)

        # Colourbar
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signal Stability (normalised)",
                       color=theme["text"], fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=theme["text"])

        # Legend for n_pcs sizes
        legend_elements = [
            plt.scatter([], [], s=size_map[v], label=f"n_pcs={v}",
                        color=theme["muted"], alpha=0.7)
            for v in npcs_vals
        ]
        ax.legend(handles=legend_elements, fontsize=7,
                  loc="lower right", framealpha=0.5)

        format_ax(ax, theme,
                  f"CHITIN v2 · Pareto Sweep  [{metric}]",
                  xlabel="Rank Disruption  (1 − ρ)  ▶ higher = better",
                  ylabel="Discrimination Ratio  (post/pre)  ▶ higher = better")

    fig.suptitle(
        f"CHITIN v2 · Pareto Sweep Results\n"
        f"★ = selected  |  green outline = Pareto front  |  "
        f"dot size = n_pcs",
        fontsize=13, fontweight="bold",
        color=theme["text"], y=1.02)
    fig.tight_layout()
    save_fig(fig, cfg, "chitin_pareto_sweep")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  NEW v2: PC SYSTEMATIC COSINES PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_pc_systematic_cosines(chitin_model, cfg: dict):
    """
    Bar chart of |cosine similarity to V| for each PC.
    Red bars = classified as systematic and projected out.
    Blue bars = retained.
    Dashed horizontal line = classification threshold.
    """
    if chitin_model.V_systematic_cos is None:
        print("No PC decomposition data available — "
              "run with correction.mode: pc_decomposition or hybrid.")
        return

    theme    = get_theme(cfg)
    cos_sims = chitin_model.V_systematic_cos
    sys_idx  = set(chitin_model.systematic_pc_indices)
    threshold = cfg.get("correction", {}).get(
        "pc_systematic_threshold", 0.30)

    n_pcs  = len(cos_sims)
    x      = np.arange(n_pcs)
    colors = [theme["warn"] if i in sys_idx else theme["accent"]
              for i in range(n_pcs)]

    fig, ax = plt.subplots(figsize=(max(10, n_pcs * 0.5), 5))

    bars = ax.bar(x, cos_sims, color=colors, alpha=0.8,
                  edgecolor=theme["panel"], linewidth=0.5)

    ax.axhline(threshold, color=theme["good"], ls="--", lw=1.5,
               label=f"Threshold = {threshold:.2f}")

    # Label systematic PCs
    for i in sys_idx:
        ax.text(i, cos_sims[i] + 0.005, f"PC{i+1}",
                ha="center", va="bottom",
                fontsize=7, color=theme["warn"], fontweight="bold")

    ax.set_xticks(x[::max(1, n_pcs // 20)])
    ax.set_xticklabels([f"PC{i+1}" for i in x[::max(1, n_pcs // 20)]],
                       rotation=45, ha="right", fontsize=8)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=theme["warn"],   label="Systematic (projected out)"),
        Patch(facecolor=theme["accent"], label="Retained"),
        plt.Line2D([0], [0], color=theme["good"], ls="--",
                   lw=1.5, label=f"Threshold ({threshold:.2f})"),
    ]
    ax.legend(handles=legend_elems, fontsize=9)

    format_ax(ax, theme,
              "CHITIN v2 · PC Systematic Variation Decomposition",
              xlabel="Principal Component",
              ylabel="|Cosine Similarity to V|")

    sys_str = (", ".join([f"PC{i+1}" for i in sorted(sys_idx)])
               if sys_idx else "none")
    ax.text(0.99, 0.97,
            f"Systematic PCs: {sys_str}\n"
            f"V magnitude: {np.linalg.norm(np.zeros(1)):.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=theme["text"],
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=theme["panel"], ec=theme["grid"], alpha=0.9))

    fig.tight_layout()
    save_fig(fig, cfg, "chitin_pc_systematic_cosines")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  DELTA ANALYSIS  (v1 unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def plot_delta_magnitude_distribution(delta_norms, cfg: dict):
    theme = get_theme(cfg)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(delta_norms, bins=80, color=theme["delta"],
            alpha=0.7, edgecolor=theme["panel"])
    ax.axvline(delta_norms.mean(), color=theme["warn"],
               ls="--", lw=2,
               label=f"Mean |Δ|: {delta_norms.mean():.3f}")
    ax.legend(fontsize=11)
    format_ax(ax, theme, "CHITIN · Delta Magnitude Distribution",
              xlabel="|ΔX_i| (L2 norm)", ylabel="Number of Metacells")
    fig.tight_layout()
    save_fig(fig, cfg, "chitin_delta_magnitudes")
    plt.show()


def plot_top_perturbation_deltas(delta_df: pd.DataFrame, cfg: dict,
                                  n_top: int = 30):
    theme = get_theme(cfg)
    df    = delta_df.head(n_top)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(df)), df["mean_delta_norm"].values,
            color=theme["delta"], height=0.6,
            edgecolor=theme["grid"], linewidth=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["perturbation"].values, fontsize=8)
    ax.invert_yaxis()

    for i, val in enumerate(df["mean_delta_norm"].values):
        ax.text(val + 0.01, i, f"{val:.3f}",
                va="center", fontsize=8, color=theme["text"])

    format_ax(ax, theme,
              f"CHITIN · Top {n_top} Perturbations by |Δ|",
              xlabel="Mean Delta Magnitude")
    fig.tight_layout()
    save_fig(fig, cfg, "chitin_top_deltas")
    plt.show()


def plot_rank_disruption(rank_corrs, sampled_genes, cfg: dict):
    theme = get_theme(cfg)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(rank_corrs, bins=50, color=theme["accent"],
            alpha=0.7, edgecolor=theme["panel"])
    ax.axvline(rank_corrs.mean(), color=theme["warn"],
               ls="--", lw=2,
               label=f"Mean ρ = {rank_corrs.mean():.3f}")
    ax.axvline(1.0, color=theme["muted"], ls=":", lw=1,
               alpha=0.5, label="No disruption (ρ=1)")
    ax.legend(fontsize=11)

    format_ax(ax, theme,
              "CHITIN · Rank-Order Disruption (Spearman ρ, Pre vs Post)",
              xlabel="Spearman Correlation", ylabel="Number of Genes")

    ax.text(0.02, 0.95,
            f"ρ < 0.5: {(rank_corrs < 0.5).sum()} genes\n"
            f"ρ < 0.8: {(rank_corrs < 0.8).sum()} genes\n"
            f"ρ > 0.95: {(rank_corrs > 0.95).sum()} genes",
            transform=ax.transAxes, fontsize=10, color=theme["text"],
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=theme["panel"], ec=theme["grid"], alpha=0.9))

    fig.tight_layout()
    save_fig(fig, cfg, "chitin_rank_disruption")
    plt.show()


def plot_pairwise_discrimination(dist_pre, dist_post, cfg: dict):
    theme = get_theme(cfg)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(dist_pre, bins=60, color=theme["muted"],
            alpha=0.5,
            label=f"Pre-CHITIN (mean={dist_pre.mean():.4f})")
    ax.hist(dist_post, bins=60, color=theme["delta"],
            alpha=0.6,
            label=f"Post-CHITIN (mean={dist_post.mean():.4f})")
    ax.axvline(dist_pre.mean(),  color=theme["muted"],  ls="--", lw=2)
    ax.axvline(dist_post.mean(), color=theme["delta"], ls="--", lw=2)

    ax.legend(fontsize=11)
    format_ax(ax, theme,
              "CHITIN · Pairwise Perturbation Discrimination",
              xlabel="Cosine Distance Between Perturbation Centroids",
              ylabel="Number of Pairs")

    improvement = ((dist_post.mean() - dist_pre.mean())
                   / dist_pre.mean() * 100)
    ax.text(0.97, 0.95,
            f"Δ discrimination: {improvement:+.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12,
            color=theme["delta"] if improvement > 0 else theme["warn"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=theme["panel"], ec=theme["grid"], alpha=0.9))

    fig.tight_layout()
    save_fig(fig, cfg, "chitin_pairwise_discrimination")
    plt.show()


def plot_expression_shift_comparison(adata_pre, adata_post,
                                      gene: str, cfg: dict):
    """Before/after violin for a single gene."""
    theme      = get_theme(cfg)
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, adata, title in [(ax1, adata_pre,  "Pre-CHITIN"),
                              (ax2, adata_post, "Post-CHITIN (Δ)")]:
        labels  = adata.obs[pert_col].values
        is_ctrl = labels == ctrl_label

        if gene not in adata.var_names:
            continue

        import scipy.sparse as sp as _sp
        gidx = list(adata.var_names).index(gene)
        X = adata.X
        if _sp.issparse(X):
            vals = X[:, gidx].toarray().flatten()
        else:
            vals = X[:, gidx].flatten()

        ctrl_vals = vals[is_ctrl]
        pert_vals = vals[~is_ctrl]

        vp = ax.violinplot([ctrl_vals, pert_vals],
                           positions=[0, 1],
                           showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(theme["ctrl"] if i == 0 else theme["pert"])
            body.set_alpha(0.6)
        vp["cmedians"].set_color(theme["highlight"])
        vp["cmedians"].set_linewidth(2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Perturbed"], fontsize=11)
        format_ax(ax, theme, title, ylabel=f"{gene} Expression")

    fig.suptitle(f"CHITIN · {gene} Expression Shift",
                 fontsize=16, fontweight="bold",
                 color=theme["text"], y=1.02)
    fig.tight_layout()
    save_fig(fig, cfg, f"chitin_gene_{gene}")
    plt.show()

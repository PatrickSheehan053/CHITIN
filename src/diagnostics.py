"""
CHITIN · src/diagnostics.py  (v2)
──────────────────────────────────
All diagnostic and evaluation functions.

New in v2:
  - sweep_results_summary()   : tabular summary of Pareto sweep results
  - pc_decomposition_report() : which PCs were identified as systematic,
                                 their cos-sim to V, and their top gene loadings
  - compute_rank_disruption() : unchanged from v1
  - compute_pairwise_discrimination() : unchanged from v1
  - All v1 functions preserved with identical signatures.

CRITICAL DESIGN NOTE (unchanged from v1):
  Post-CHITIN, control cells are all zeros, which collapses the baseline to
  the origin and makes the Systema cosine metric measure something different.
  Always compute Systema metrics on pre-CHITIN data.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from .utils import log_phase, log_memory


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEMA CENTROID ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_systema_centroids(adata, cfg: dict, logger):
    """
    Compute Systema global centroids from the expression matrix.

    Returns: C_ctrl, O_pert, V  (all in gene expression space)
    """
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels    = adata.obs[pert_col].values
    ctrl_mask = labels == ctrl_label

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    C_ctrl = X[ctrl_mask].mean(axis=0)
    unique_perts = [p for p in np.unique(labels) if p != ctrl_label]
    pert_centroids = []
    for p in unique_perts:
        mask = labels == p
        pert_centroids.append(X[mask].mean(axis=0))
    O_pert = np.array(pert_centroids).mean(axis=0)
    V = O_pert - C_ctrl

    logger.info(f"  Systema centroids:")
    logger.info(f"    C_ctrl norm : {np.linalg.norm(C_ctrl):.4f}")
    logger.info(f"    O_pert norm : {np.linalg.norm(O_pert):.4f}")
    logger.info(f"    |V|         : {np.linalg.norm(V):.4f}")
    return C_ctrl, O_pert, V


def compute_perturbation_cosines(adata, C_ctrl, O_pert, cfg: dict, logger):
    """
    Per-perturbation cosine similarities between perturbation-specific
    shifts and the systematic variation vector V.

    Returns: (cosines_ctrl_ref, cosines_pert_ref, perturbation_names)
    """
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    labels     = adata.obs[pert_col].values

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    V      = O_pert - C_ctrl
    V_norm = np.linalg.norm(V)
    unique_perts = sorted([p for p in np.unique(labels) if p != ctrl_label])

    cosines_ctrl_ref = []
    cosines_pert_ref = []
    for p in unique_perts:
        mask      = labels == p
        centroid  = X[mask].mean(axis=0)

        shift_ctrl = centroid - C_ctrl
        n_ctrl     = np.linalg.norm(shift_ctrl)
        cos_ctrl   = (float(np.dot(shift_ctrl, V) / (n_ctrl * V_norm + 1e-12))
                      if n_ctrl > 1e-12 else 0.0)
        cosines_ctrl_ref.append(cos_ctrl)

        shift_pert = centroid - O_pert
        n_pert     = np.linalg.norm(shift_pert)
        cos_pert   = (float(np.dot(shift_pert, V) / (n_pert * V_norm + 1e-12))
                      if n_pert > 1e-12 else 0.0)
        cosines_pert_ref.append(cos_pert)

    cosines_ctrl_ref = np.array(cosines_ctrl_ref)
    cosines_pert_ref = np.array(cosines_pert_ref)
    logger.info(f"  Cosine (ctrl ref):  {cosines_ctrl_ref.mean():.3f} "
                f"± {cosines_ctrl_ref.std():.3f}")
    logger.info(f"  Cosine (pert ref):  {cosines_pert_ref.mean():.3f} "
                f"± {cosines_pert_ref.std():.3f}")
    return cosines_ctrl_ref, cosines_pert_ref, unique_perts


# ═══════════════════════════════════════════════════════════════════════════
#  RANK-ORDER DISRUPTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_rank_disruption(adata_pre, adata_post, cfg: dict, logger,
                             n_genes_sample: int = 200):
    """
    Measure how much CHITIN disrupted gene expression rank orders.
    This is the primary metric for LightGBM effectiveness.

    Returns: mean_rank_corr, rank_corrs, sampled_genes
    """
    log_phase(logger, "CHITIN · Rank-Order Disruption")

    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels    = adata_pre.obs[pert_col].values
    pert_mask = labels != ctrl_label
    pert_idx  = np.where(pert_mask)[0]

    X_pre = adata_pre.X
    if sp.issparse(X_pre):
        X_pre = X_pre.toarray()
    X_post = adata_post.X
    if sp.issparse(X_post):
        X_post = X_post.toarray()

    n_genes  = adata_pre.n_vars
    rng      = np.random.default_rng(42)
    n_sample = min(n_genes_sample, n_genes)
    gene_idx = rng.choice(n_genes, size=n_sample, replace=False)
    sampled_genes = adata_pre.var_names[gene_idx].tolist()

    rank_corrs = []
    for gi in gene_idx:
        pre  = X_pre[pert_idx,  gi]
        post = X_post[pert_idx, gi]
        if np.std(pre) < 1e-12 or np.std(post) < 1e-12:
            continue
        r, _ = spearmanr(pre, post)
        rank_corrs.append(r)

    rank_corrs = np.array(rank_corrs)
    mean_corr  = rank_corrs.mean()

    logger.info(f"  Rank disruption ({n_sample} genes):")
    logger.info(f"    Mean ρ : {mean_corr:.4f}   Std: {rank_corrs.std():.4f}")
    logger.info(f"    Min: {rank_corrs.min():.4f}   Max: {rank_corrs.max():.4f}")

    if mean_corr > 0.95:
        logger.warning("    ⚠ Very high rank preservation — limited LightGBM impact")
    elif mean_corr < 0.80:
        logger.info("    ✓ Strong disruption")
    else:
        logger.info("    Moderate disruption")

    return mean_corr, rank_corrs, sampled_genes


# ═══════════════════════════════════════════════════════════════════════════
#  PAIRWISE DISCRIMINATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_pairwise_discrimination(adata_pre, adata_post, cfg: dict, logger,
                                     n_pairs: int = 5000):
    """
    Measure whether CHITIN improved perturbation separability.
    Returns: dist_pre, dist_post
    """
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels = adata_pre.obs[pert_col].values

    X_pre  = adata_pre.X
    if sp.issparse(X_pre):
        X_pre = X_pre.toarray()
    X_post = adata_post.X
    if sp.issparse(X_post):
        X_post = X_post.toarray()

    unique_perts  = [p for p in np.unique(labels) if p != ctrl_label]
    centroids_pre  = {p: X_pre[labels == p].mean(axis=0)  for p in unique_perts}
    centroids_post = {p: X_post[labels == p].mean(axis=0) for p in unique_perts}

    rng      = np.random.default_rng(42)
    n        = len(unique_perts)
    n_actual = min(n_pairs, n * (n - 1) // 2)
    dist_pre, dist_post = [], []
    pairs_seen = set()

    while len(dist_pre) < n_actual:
        i, j = rng.integers(0, n, size=2)
        if i == j or (i, j) in pairs_seen:
            continue
        pairs_seen.add((i, j))
        p1, p2 = unique_perts[i], unique_perts[j]

        for dist_list, cents in [(dist_pre,  centroids_pre),
                                  (dist_post, centroids_post)]:
            c1, c2 = cents[p1], cents[p2]
            n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
            if n1 > 1e-12 and n2 > 1e-12:
                dist_list.append(1.0 - float(np.dot(c1 / n1, c2 / n2)))
            else:
                dist_list.append(1.0)

    dist_pre  = np.array(dist_pre)
    dist_post = np.array(dist_post)

    improvement = (dist_post.mean() - dist_pre.mean()) / dist_pre.mean() * 100
    logger.info(f"  Pairwise discrimination ({n_actual} pairs):")
    logger.info(f"    Pre : {dist_pre.mean():.4f} ± {dist_pre.std():.4f}")
    logger.info(f"    Post: {dist_post.mean():.4f} ± {dist_post.std():.4f}")
    logger.info(f"    Δ   : {improvement:+.1f}%")

    return dist_pre, dist_post


# ═══════════════════════════════════════════════════════════════════════════
#  K-SENSITIVITY SWEEP  (v1 API preserved — now complemented by Pareto sweep)
# ═══════════════════════════════════════════════════════════════════════════

def k_sensitivity_sweep(adata, cfg: dict, logger):
    """
    Sweep k values and measure localisation of basal vectors.
    Preserved from v1 for diagnostic/plotting use.
    The auto-calibration Pareto sweep in engine.py supersedes this for
    parameter selection — this function is now a diagnostic visualisation tool.

    Returns: k_range, basal_variances, mean_distances
    """
    log_phase(logger, "CHITIN · k-Sensitivity Sweep (diagnostic)")

    k_range    = cfg["diagnostics"]["k_sweep_range"]
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    metric     = cfg["knn"]["distance_metric"]

    labels    = adata.obs[pert_col].values
    ctrl_mask = labels == ctrl_label
    ctrl_idx  = np.where(ctrl_mask)[0]
    pert_idx  = np.where(~ctrl_mask)[0]

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X_ctrl = X[ctrl_idx]

    pca_ctrl = adata.obsm["X_pca_chitin"][ctrl_idx]
    pca_pert = adata.obsm["X_pca_chitin"][pert_idx]

    max_k   = len(ctrl_idx) - 1
    k_range = [k for k in k_range if k <= max_k]

    basal_variances = []
    mean_distances  = []
    n_sample        = min(5000, len(pert_idx))
    rng             = np.random.default_rng(42)
    sample_idx      = rng.choice(len(pert_idx), size=n_sample, replace=False)
    pca_pert_sample = pca_pert[sample_idx]

    for k in k_range:
        nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
        nn.fit(pca_ctrl)
        dists, nbr_idx = nn.kneighbors(pca_pert_sample)

        N_i = np.zeros((n_sample, X.shape[1]), dtype=np.float32)
        for i in range(n_sample):
            N_i[i] = X_ctrl[nbr_idx[i]].mean(axis=0)

        var_N = float(np.var(N_i, axis=0).mean())
        basal_variances.append(var_N)
        mean_distances.append(float(dists.mean()))
        logger.info(f"  k={k:4d}  basal_var={var_N:.6f}  "
                    f"mean_dist={dists.mean():.4f}")

    return k_range, basal_variances, mean_distances


# ═══════════════════════════════════════════════════════════════════════════
#  NEW v2: SWEEP RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def sweep_results_summary(chitin_model, logger):
    """
    Summarise the Pareto sweep results from a fitted ChitinModel.

    Returns: summary_df (full results), pareto_df (Pareto front only)
    """
    if chitin_model.sweep_results is None or len(chitin_model.sweep_results) == 0:
        logger.info("  No sweep results available "
                    "(auto_calibrate was False or sweep not run)")
        return None, None

    df     = chitin_model.sweep_results.copy()
    pareto = chitin_model.pareto_front.copy() if chitin_model.pareto_front is not None else df

    logger.info(f"  Sweep: {len(df)} combinations evaluated, "
                f"{len(pareto)} on Pareto front")
    logger.info(f"  Selected: {chitin_model.selected_params}")

    # Summary statistics
    logger.info("\n  Top 10 by rank_disruption:")
    top10 = df.nlargest(10, "rank_disruption")[
        ["k","n_pcs","metric","rank_disruption","disc_ratio","signal_stability"]]
    for _, row in top10.iterrows():
        star = " ★" if (int(row["k"]) == chitin_model.k and
                        int(row["n_pcs"]) == chitin_model.n_pcs and
                        row["metric"] == chitin_model.distance_metric) else ""
        logger.info(f"    k={int(row['k']):<3} n_pcs={int(row['n_pcs']):<3} "
                    f"metric={row['metric']:<10} "
                    f"disrupt={row['rank_disruption']:.4f}  "
                    f"disc={row['disc_ratio']:.4f}  "
                    f"stab={row['signal_stability']:.4f}{star}")

    return df, pareto


# ═══════════════════════════════════════════════════════════════════════════
#  NEW v2: PC DECOMPOSITION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def pc_decomposition_report(chitin_model, adata, cfg, logger, top_n_genes=10):
    """
    Report which PCs were classified as systematic, their cosine similarities
    to V, and their top gene loadings (most influential genes per PC).

    Returns: report_df with one row per PC, including top gene names.
    """
    if chitin_model.correction_mode not in ("pc_decomposition", "hybrid"):
        logger.info("  PC decomposition not active for this model.")
        return None

    if chitin_model.V_systematic_cos is None:
        logger.info("  No PC decomposition data available.")
        return None

    cos_sims = chitin_model.V_systematic_cos
    n_pcs    = len(cos_sims)
    var_names = list(adata.var_names)
    pca_loadings = chitin_model.pca_components  # (n_genes × n_pcs)

    log_phase(logger, "CHITIN v2 · PC Decomposition Report")
    logger.info(f"  Total PCs computed: {n_pcs}")
    logger.info(f"  Systematic PCs (cos_sim to V > threshold): "
                f"{chitin_model.systematic_pc_indices}")

    records = []
    for i in range(n_pcs):
        cos = float(cos_sims[i])
        is_systematic = i in chitin_model.systematic_pc_indices

        top_genes = []
        if pca_loadings is not None and i < pca_loadings.shape[1]:
            loadings = pca_loadings[:, i]
            top_idx  = np.argsort(np.abs(loadings))[::-1][:top_n_genes]
            top_genes = [(var_names[j], float(loadings[j])) for j in top_idx]

        record = {
            "pc_index":      i + 1,
            "cos_sim_to_V":  cos,
            "is_systematic": is_systematic,
            "top_genes":     ", ".join([f"{g}({w:+.3f})"
                                        for g, w in top_genes[:5]]),
        }
        records.append(record)

        flag = " ← SYSTEMATIC" if is_systematic else ""
        logger.info(f"  PC{i+1:02d}: cos_sim={cos:.4f}{flag}")
        if is_systematic and top_genes:
            logger.info(f"         Top genes: "
                        f"{', '.join([g for g, _ in top_genes[:5]])}")

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
#  DELTA MAGNITUDE ANALYSIS (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_magnitudes(adata_delta, cfg: dict, logger):
    """Compute L2 norm of each perturbed metacell's delta vector."""
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels    = adata_delta.obs[pert_col].values
    pert_mask = labels != ctrl_label

    X = adata_delta.X
    if sp.issparse(X):
        X = X.toarray()

    delta_norms  = np.linalg.norm(X[pert_mask], axis=1)
    pert_labels  = labels[pert_mask]
    unique_perts = np.unique(pert_labels)

    records = []
    for p in unique_perts:
        mask  = pert_labels == p
        norms = delta_norms[mask]
        records.append({
            "perturbation":    p,
            "mean_delta_norm": float(norms.mean()),
            "std_delta_norm":  float(norms.std()),
            "n_metacells":     int(mask.sum()),
        })

    df = pd.DataFrame(records).sort_values("mean_delta_norm", ascending=False)
    logger.info(f"  Delta magnitudes: mean={delta_norms.mean():.4f}, "
                f"std={delta_norms.std():.4f}")
    logger.info("  Top 5:")
    for _, row in df.head(5).iterrows():
        logger.info(f"    {row['perturbation']}: "
                    f"|Δ|={row['mean_delta_norm']:.4f} "
                    f"(n={row['n_metacells']})")

    return df, delta_norms

"""
CHITIN · src/diagnostics.py
────────────────────────────
Diagnostic and evaluation tools.

CRITICAL DESIGN NOTE:
  The Systema cosine metric measures alignment of perturbation-specific
  shifts with the average perturbation effect, relative to a baseline centroid.

  Post-CHITIN, control cells are all zeros, which collapses the baseline to
  the origin and makes the cosine metric measure something entirely different
  (raw delta alignment from origin, which is trivially high).

  CORRECT approach: To evaluate whether CHITIN reduced systematic variation,
  we must compare the ORIGINAL expression space geometry before and after
  the transformation. We do this by:
    1. Computing Systema metrics on the pre-CHITIN data (standard)
    2. Computing the "effective" post-CHITIN perturbation centroids by
       adding back the control baseline: X_eff = delta + C_ctrl_pre
    OR (simpler and more interpretable):
    3. Measuring rank-order disruption directly — this is what actually
       matters for LightGBM.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from .utils import log_phase, log_memory


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEMA CENTROID ANALYSIS (on original expression data)
# ═══════════════════════════════════════════════════════════════════════════

def compute_systema_centroids(adata, cfg: dict, logger):
    """
    Compute Systema global centroids from the expression matrix.

    Returns:
        C_ctrl: centroid of control metacells (R^G)
        O_pert: average of perturbation-specific centroids (R^G)
        V: systematic variation vector (O_pert - C_ctrl)
    """
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels = adata.obs[pert_col].values
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

    logger.info(f"  Systema centroids computed:")
    logger.info(f"    C_ctrl norm: {np.linalg.norm(C_ctrl):.4f}")
    logger.info(f"    O_pert norm: {np.linalg.norm(O_pert):.4f}")
    logger.info(f"    |V| (systematic variation magnitude): {np.linalg.norm(V):.4f}")

    return C_ctrl, O_pert, V


def compute_perturbation_cosines(adata, C_ctrl, O_pert, cfg: dict, logger):
    """
    Per-perturbation cosine similarities between perturbation-specific
    shifts and the systematic variation vector V.

    Returns: (cosines_ctrl_ref, cosines_pert_ref, perturbation_names)
    """
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels = adata.obs[pert_col].values

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    V = O_pert - C_ctrl
    V_norm = np.linalg.norm(V)

    unique_perts = sorted([p for p in np.unique(labels) if p != ctrl_label])

    cosines_ctrl_ref = []
    cosines_pert_ref = []

    for p in unique_perts:
        mask = labels == p
        centroid_p = X[mask].mean(axis=0)

        shift_ctrl = centroid_p - C_ctrl
        norm_ctrl = np.linalg.norm(shift_ctrl)
        cos_ctrl = (np.dot(shift_ctrl, V) / (norm_ctrl * V_norm + 1e-12)
                    if norm_ctrl > 1e-12 else 0.0)
        cosines_ctrl_ref.append(cos_ctrl)

        shift_pert = centroid_p - O_pert
        norm_pert = np.linalg.norm(shift_pert)
        cos_pert = (np.dot(shift_pert, V) / (norm_pert * V_norm + 1e-12)
                    if norm_pert > 1e-12 else 0.0)
        cosines_pert_ref.append(cos_pert)

    cosines_ctrl_ref = np.array(cosines_ctrl_ref)
    cosines_pert_ref = np.array(cosines_pert_ref)

    logger.info(f"  Cosine similarities (control ref): "
                f"mean={cosines_ctrl_ref.mean():.3f} ± {cosines_ctrl_ref.std():.3f}")
    logger.info(f"  Cosine similarities (perturbed ref): "
                f"mean={cosines_pert_ref.mean():.3f} ± {cosines_pert_ref.std():.3f}")

    return cosines_ctrl_ref, cosines_pert_ref, unique_perts


# ═══════════════════════════════════════════════════════════════════════════
#  CHITIN EFFECTIVENESS METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_rank_disruption(adata_pre, adata_post, cfg: dict, logger,
                             n_genes_sample: int = 200):
    """
    Measure how much CHITIN disrupted the rank order of gene expression.
    This is the metric that ACTUALLY matters for LightGBM — if ranks don't
    change, information gain doesn't change, and the GRN is identical.

    For a sample of genes, compute the Spearman rank correlation between
    pre-CHITIN and post-CHITIN expression across all perturbed metacells.
    Low correlation = high disruption = CHITIN is working.

    Returns:
        mean_rank_corr: average Spearman correlation across sampled genes
        rank_corrs: per-gene Spearman correlations
        sampled_genes: gene names that were sampled
    """
    from scipy.stats import spearmanr

    log_phase(logger, "CHITIN · Rank-Order Disruption Analysis")

    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels = adata_pre.obs[pert_col].values
    pert_mask = labels != ctrl_label
    pert_idx = np.where(pert_mask)[0]

    X_pre = adata_pre.X
    if sp.issparse(X_pre):
        X_pre = X_pre.toarray()

    X_post = adata_post.X
    if sp.issparse(X_post):
        X_post = X_post.toarray()

    # Sample genes for speed
    n_genes = adata_pre.n_vars
    rng = np.random.default_rng(42)
    n_sample = min(n_genes_sample, n_genes)
    gene_idx = rng.choice(n_genes, size=n_sample, replace=False)
    sampled_genes = adata_pre.var_names[gene_idx].tolist()

    rank_corrs = []
    for gi in gene_idx:
        pre_vals = X_pre[pert_idx, gi]
        post_vals = X_post[pert_idx, gi]

        # Skip constant vectors
        if np.std(pre_vals) < 1e-12 or np.std(post_vals) < 1e-12:
            continue

        corr, _ = spearmanr(pre_vals, post_vals)
        rank_corrs.append(corr)

    rank_corrs = np.array(rank_corrs)
    mean_corr = rank_corrs.mean()

    logger.info(f"  Rank-order disruption ({n_sample} genes sampled):")
    logger.info(f"    Mean Spearman correlation: {mean_corr:.4f}")
    logger.info(f"    Std:  {rank_corrs.std():.4f}")
    logger.info(f"    Min:  {rank_corrs.min():.4f}")
    logger.info(f"    Max:  {rank_corrs.max():.4f}")

    if mean_corr > 0.95:
        logger.warning(f"    ⚠ Very high rank preservation — CHITIN may not be "
                        f"providing meaningful disruption for tree-based GRNs")
    elif mean_corr < 0.5:
        logger.info(f"    ✓ Strong rank disruption — information gain landscape "
                     f"has been substantially reshaped")
    else:
        logger.info(f"    Moderate rank disruption — partial reshaping of the "
                     f"information gain landscape")

    return mean_corr, rank_corrs, sampled_genes


def compute_pairwise_discrimination(adata_pre, adata_post, cfg: dict, logger,
                                      n_pairs: int = 5000):
    """
    Measure whether CHITIN improved perturbation DISCRIMINATION.

    For random pairs of perturbations, compute the cosine distance between
    their centroids, before and after CHITIN. If CHITIN is working, perturbation
    centroids should be more SPREAD OUT (less dominated by the shared
    systematic direction), meaning pairwise distances should increase or
    the variance of pairwise distances should increase.

    Returns:
        dist_pre: pairwise distances before CHITIN
        dist_post: pairwise distances after CHITIN
    """
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels_pre = adata_pre.obs[pert_col].values
    labels_post = adata_post.obs[pert_col].values

    X_pre = adata_pre.X
    if sp.issparse(X_pre):
        X_pre = X_pre.toarray()
    X_post = adata_post.X
    if sp.issparse(X_post):
        X_post = X_post.toarray()

    unique_perts = [p for p in np.unique(labels_pre) if p != ctrl_label]

    # Compute centroids
    centroids_pre = {}
    centroids_post = {}
    for p in unique_perts:
        mask = labels_pre == p
        centroids_pre[p] = X_pre[mask].mean(axis=0)
        centroids_post[p] = X_post[mask].mean(axis=0)

    # Sample random pairs
    rng = np.random.default_rng(42)
    n = len(unique_perts)
    n_actual = min(n_pairs, n * (n - 1) // 2)

    dist_pre = []
    dist_post = []

    pairs_seen = set()
    while len(dist_pre) < n_actual:
        i, j = rng.integers(0, n, size=2)
        if i == j or (i, j) in pairs_seen:
            continue
        pairs_seen.add((i, j))

        p1, p2 = unique_perts[i], unique_perts[j]

        # Cosine distance = 1 - cosine_similarity
        c_pre_1, c_pre_2 = centroids_pre[p1], centroids_pre[p2]
        norm_product = np.linalg.norm(c_pre_1) * np.linalg.norm(c_pre_2)
        if norm_product > 1e-12:
            cos_pre = np.dot(c_pre_1, c_pre_2) / norm_product
            dist_pre.append(1 - cos_pre)
        else:
            dist_pre.append(1.0)

        c_post_1, c_post_2 = centroids_post[p1], centroids_post[p2]
        norm_product = np.linalg.norm(c_post_1) * np.linalg.norm(c_post_2)
        if norm_product > 1e-12:
            cos_post = np.dot(c_post_1, c_post_2) / norm_product
            dist_post.append(1 - cos_post)
        else:
            dist_post.append(1.0)

    dist_pre = np.array(dist_pre)
    dist_post = np.array(dist_post)

    logger.info(f"  Pairwise perturbation discrimination ({n_actual} pairs):")
    logger.info(f"    Pre-CHITIN  → mean cosine dist: {dist_pre.mean():.4f} ± {dist_pre.std():.4f}")
    logger.info(f"    Post-CHITIN → mean cosine dist: {dist_post.mean():.4f} ± {dist_post.std():.4f}")

    improvement = (dist_post.mean() - dist_pre.mean()) / dist_pre.mean() * 100
    if improvement > 0:
        logger.info(f"    ✓ Discrimination improved by {improvement:.1f}% "
                     f"(perturbations are more separable)")
    else:
        logger.info(f"    Discrimination changed by {improvement:.1f}%")

    return dist_pre, dist_post


# ═══════════════════════════════════════════════════════════════════════════
#  K-SENSITIVITY SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def k_sensitivity_sweep(adata, cfg: dict, logger):
    """
    Sweep across k values and measure localization of basal vectors.
    """
    log_phase(logger, "CHITIN · k-Sensitivity Sweep")

    k_range = cfg["diagnostics"]["k_sweep_range"]
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    metric = cfg["knn"]["distance_metric"]

    labels = adata.obs[pert_col].values
    ctrl_mask = labels == ctrl_label
    pert_mask = ~ctrl_mask

    ctrl_idx = np.where(ctrl_mask)[0]
    pert_idx = np.where(pert_mask)[0]

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    X_ctrl = X[ctrl_idx]

    pca_ctrl = adata.obsm["X_pca_chitin"][ctrl_idx]
    pca_pert = adata.obsm["X_pca_chitin"][pert_idx]

    max_k = len(ctrl_idx) - 1
    k_range = [k for k in k_range if k <= max_k]

    basal_variances = []
    mean_distances = []
    n_sample = min(5000, len(pert_idx))

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(pert_idx), size=n_sample, replace=False)
    pca_pert_sample = pca_pert[sample_idx]

    for k in k_range:
        nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
        nn.fit(pca_ctrl)
        dists, nbr_idx = nn.kneighbors(pca_pert_sample)

        N_i = np.zeros((n_sample, X.shape[1]), dtype=np.float32)
        for i in range(n_sample):
            N_i[i] = X_ctrl[nbr_idx[i]].mean(axis=0)

        var_N = np.var(N_i, axis=0).mean()
        basal_variances.append(var_N)
        mean_distances.append(float(dists.mean()))

        logger.info(f"  k={k:4d} → basal variance: {var_N:.6f}, "
                    f"mean dist: {dists.mean():.4f}")

    return k_range, basal_variances, mean_distances


# ═══════════════════════════════════════════════════════════════════════════
#  DELTA MAGNITUDE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_magnitudes(adata_delta, cfg: dict, logger):
    """Compute L2 norm of each perturbed metacell's delta vector."""
    import pandas as pd

    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    labels = adata_delta.obs[pert_col].values
    pert_mask = labels != ctrl_label

    X = adata_delta.X
    if sp.issparse(X):
        X = X.toarray()

    delta_norms = np.linalg.norm(X[pert_mask], axis=1)

    pert_labels = labels[pert_mask]
    unique_perts = np.unique(pert_labels)
    records = []
    for p in unique_perts:
        mask = pert_labels == p
        norms = delta_norms[mask]
        records.append({
            "perturbation": p,
            "mean_delta_norm": norms.mean(),
            "std_delta_norm": norms.std(),
            "n_metacells": mask.sum(),
        })

    df = pd.DataFrame(records).sort_values("mean_delta_norm", ascending=False)

    logger.info(f"  Delta magnitudes: mean={delta_norms.mean():.4f}, "
                f"std={delta_norms.std():.4f}")
    logger.info(f"  Top 5 strongest perturbations:")
    for _, row in df.head(5).iterrows():
        logger.info(f"    {row['perturbation']}: |Δ|={row['mean_delta_norm']:.4f} "
                     f"(n={row['n_metacells']})")

    return df, delta_norms

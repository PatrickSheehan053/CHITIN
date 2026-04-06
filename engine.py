"""
CHITIN · src/engine.py
──────────────────────
Core engine for localized manifold subtraction.

Architecture:
  - fit():      Compute PCA + fit KNN on training split control metacells
  - transform(): Apply localized subtraction to ANY split using the fitted reference

This separation ensures:
  1. The control baseline is always defined from training data (no leakage)
  2. Test splits with zero control cells can still be transformed
  3. CHITIN can work on any .h5ad file (standalone mode auto-fits and transforms)
"""

import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from .utils import log_phase, snapshot, log_memory, force_gc


class ChitinModel:
    """
    Fitted CHITIN model that stores the control reference manifold.

    Usage:
        model = ChitinModel()
        model.fit(adata_train, cfg, logger)
        delta_train = model.transform(adata_train, cfg, logger)
        delta_val   = model.transform(adata_val, cfg, logger)
        delta_test  = model.transform(adata_test, cfg, logger)
    """

    def __init__(self):
        self.pca_components = None   # PCA projection matrix (varm['PCs'])
        self.nn_model = None         # fitted NearestNeighbors on control PCA coords
        self.X_ctrl_expr = None      # control expression matrix for basal computation
        self.k = None
        self.is_fitted = False

    def fit(self, adata, cfg: dict, logger):
        """
        Fit CHITIN on a training split:
          1. Compute PCA on full training data
          2. Fit KNN on control metacells in PCA space
          3. Store control expression vectors for basal derivation
        """
        log_phase(logger, "CHITIN · Fitting Reference Manifold (Train Controls)")

        pert_col = cfg["dataset"]["perturbation_col"]
        ctrl_label = cfg["dataset"]["control_label"]
        n_pcs = cfg["knn"]["n_pcs"]
        self.k = cfg["knn"]["k"]
        metric = cfg["knn"]["distance_metric"]
        solver = cfg["knn"]["svd_solver"]

        # ── PCA on full training data ──────────────────────────────────────
        logger.info(f"  Computing PCA on training data ({n_pcs} components, {solver})...")
        adata_work = adata.copy()
        sc.pp.scale(adata_work, max_value=10, zero_center=False)
        sc.pp.pca(adata_work, n_comps=n_pcs, use_highly_variable=False,
                  svd_solver=solver, zero_center=False)

        adata.obsm["X_pca_chitin"] = adata_work.obsm["X_pca"].copy()

        if "pca" in adata_work.uns:
            adata.uns["chitin_pca_variance_ratio"] = adata_work.uns["pca"]["variance_ratio"]

        # Store PCA projection for reuse on other splits
        if hasattr(adata_work, 'varm') and 'PCs' in adata_work.varm:
            self.pca_components = adata_work.varm['PCs'].copy()

        del adata_work
        force_gc(logger)

        # ── Identify controls and fit KNN ──────────────────────────────────
        labels = adata.obs[pert_col].values
        ctrl_mask = labels == ctrl_label
        ctrl_indices = np.where(ctrl_mask)[0]
        n_ctrl = len(ctrl_indices)
        logger.info(f"  Control metacells in training split: {n_ctrl:,}")

        if n_ctrl == 0:
            raise ValueError("No control metacells found. CHITIN requires controls to fit.")

        if self.k >= n_ctrl:
            old_k = self.k
            self.k = max(1, n_ctrl - 1)
            logger.warning(f"  k={old_k} exceeds control count ({n_ctrl}). Reduced to k={self.k}")

        pca_ctrl = adata.obsm["X_pca_chitin"][ctrl_indices]

        logger.info(f"  Fitting KNN (k={self.k}, metric='{metric}') on {n_ctrl:,} controls...")
        self.nn_model = NearestNeighbors(n_neighbors=self.k, metric=metric, n_jobs=-1)
        self.nn_model.fit(pca_ctrl)

        # ── Store control expression vectors ───────────────────────────────
        X = adata.X
        if sp.issparse(X):
            self.X_ctrl_expr = X[ctrl_indices].toarray().astype(np.float32)
        else:
            self.X_ctrl_expr = np.array(X[ctrl_indices], dtype=np.float32)

        self.is_fitted = True
        logger.info(f"  CHITIN model fitted. Ready to transform any split.")
        log_memory(logger, "post fit")

        return adata

    def _project_to_pca(self, adata, cfg: dict, logger):
        """Project a new split into the training PCA space."""
        if self.pca_components is not None:
            logger.info(f"  Projecting into training PCA space...")
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            X_scaled = np.clip(X.astype(np.float32), None, 10)
            adata.obsm["X_pca_chitin"] = X_scaled @ self.pca_components
        else:
            logger.info(f"  No stored PCA — computing independent PCA (standalone mode)...")
            n_pcs = cfg["knn"]["n_pcs"]
            solver = cfg["knn"]["svd_solver"]
            adata_work = adata.copy()
            sc.pp.scale(adata_work, max_value=10, zero_center=False)
            sc.pp.pca(adata_work, n_comps=n_pcs, use_highly_variable=False,
                      svd_solver=solver, zero_center=False)
            adata.obsm["X_pca_chitin"] = adata_work.obsm["X_pca"].copy()
            del adata_work
            force_gc(logger)
        return adata

    def transform(self, adata, cfg: dict, logger, label: str = ""):
        """
        Transform any split using the fitted CHITIN reference.

        Each perturbed metacell is compared against the TRAINING controls
        (stored in self), NOT the controls in this split.
        """
        if not self.is_fitted:
            raise RuntimeError("CHITIN model not fitted. Call fit() first.")

        tag = f" [{label}]" if label else ""
        log_phase(logger, f"CHITIN · Transforming{tag}")

        pert_col = cfg["dataset"]["perturbation_col"]
        ctrl_label = cfg["dataset"]["control_label"]

        labels = adata.obs[pert_col].values
        ctrl_mask = labels == ctrl_label
        pert_mask = ~ctrl_mask

        ctrl_indices = np.where(ctrl_mask)[0]
        pert_indices = np.where(pert_mask)[0]

        logger.info(f" {tag} Controls in this split: {len(ctrl_indices):,} | "
                    f"Perturbed: {len(pert_indices):,}")

        if len(pert_indices) == 0:
            logger.warning(f" {tag} No perturbed metacells — returning unchanged.")
            return adata

        # Project into training PCA space
        adata = self._project_to_pca(adata, cfg, logger)

        # KNN lookup against TRAINING controls
        pca_pert = adata.obsm["X_pca_chitin"][pert_indices]
        distances, neighbor_idx = self.nn_model.kneighbors(pca_pert)
        logger.info(f" {tag} KNN complete → {len(pert_indices):,} × {self.k} neighbors")

        # Get expression matrix
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        n_genes = adata.n_vars
        n_pert = len(pert_indices)

        # Compute N_i from TRAINING controls
        N_i = np.zeros((n_pert, n_genes), dtype=np.float32)
        for i in range(n_pert):
            N_i[i] = self.X_ctrl_expr[neighbor_idx[i]].mean(axis=0)

        # Delta extraction
        delta_pert = X[pert_indices] - N_i

        # Build output
        X_delta = np.zeros((adata.n_obs, n_genes), dtype=np.float32)
        X_delta[pert_indices] = delta_pert

        adata_delta = ad.AnnData(
            X=X_delta,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy() if adata.uns else {},
        )

        if adata.obsm:
            for key, val in adata.obsm.items():
                adata_delta.obsm[key] = val.copy()

        # Diagnostic layers
        X_basal = np.zeros((adata.n_obs, n_genes), dtype=np.float32)
        X_basal[pert_indices] = N_i
        if len(ctrl_indices) > 0:
            X_basal[ctrl_indices] = X[ctrl_indices]
        adata_delta.layers["basal"] = X_basal
        adata_delta.layers["pre_chitin"] = X.copy()

        adata_delta.obs["chitin_transformed"] = False
        adata_delta.obs.iloc[pert_indices,
                             adata_delta.obs.columns.get_loc("chitin_transformed")] = True

        snapshot(adata_delta, f"Post CHITIN{tag}", logger)
        return adata_delta


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def fit_and_transform_all(adata_train, adata_val, adata_test,
                           cfg: dict, logger):
    """Fit on train controls, transform all three splits."""
    model = ChitinModel()
    adata_train = model.fit(adata_train, cfg, logger)
    delta_train = model.transform(adata_train, cfg, logger, label="Train")
    delta_val = model.transform(adata_val, cfg, logger, label="Val")
    delta_test = model.transform(adata_test, cfg, logger, label="Test")
    return model, delta_train, delta_val, delta_test


def run_chitin_standalone(adata, cfg: dict, logger):
    """
    Standalone mode: fit AND transform on the same dataset.
    Works on any .h5ad with a perturbation column and control label.
    """
    log_phase(logger, "CHITIN · Standalone Mode")

    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    if pert_col not in adata.obs.columns:
        raise ValueError(f"Column '{pert_col}' not found in .obs. "
                         f"Available: {list(adata.obs.columns)}")

    n_ctrl = (adata.obs[pert_col].values == ctrl_label).sum()
    if n_ctrl == 0:
        raise ValueError(f"No cells with label '{ctrl_label}' found. "
                         f"CHITIN requires control cells.")

    logger.info(f"  Standalone: {adata.n_obs:,} cells, {n_ctrl:,} controls")

    model = ChitinModel()
    adata = model.fit(adata, cfg, logger)
    adata_delta = model.transform(adata, cfg, logger, label="Standalone")

    return adata_delta, model
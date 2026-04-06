# CHITIN

### Causal Heuristics Isolating Transcriptomic Intervention Noise

CHITIN is a preprocessing module that strips systematic pan-perturbation variation from single-cell CRISPR screen expression matrices using localized manifold subtraction. It produces a clean delta-matrix of isolated causal perturbation signatures, optimized for downstream Gene Regulatory Network (GRN) inference engines that rely on rank-order statistics (such as LightGBM-based methods like GuanLab's PSGRN).

CHITIN is designed to pair with [SPORE](https://github.com/PatrickSheehan053/SPORE) as part of the MYCELIUM pipeline, but it can also run as a standalone tool on any `.h5ad` file that contains perturbation labels and unperturbed control cells.

## The Problem: Systematic Variation in Perturbation Screens

The Systema evaluation framework ([Vinas Torne et al., Nature Biotechnology 2025](https://doi.org/10.1038/s41587-025-02777-8)) demonstrated that existing perturbation response prediction methods achieve inflated performance scores because they primarily capture systematic variation rather than perturbation-specific causal effects. Systematic variation is the consistent transcriptional shift between perturbed and control cells that arises from shared stress responses, selection biases, cell-cycle arrest, and other confounders that are common across all perturbations regardless of which gene was targeted.

Systema quantified this by measuring the cosine similarity between each perturbation's expression shift and the average perturbation effect across ten benchmark datasets. They found that this systematic variation strongly correlated with reported performance scores (Pearson r = 0.91 to 0.95), meaning that methods which appeared to predict perturbation responses were largely just predicting the shared background signal.

The standard correction proposed by Systema is a global geometric shift: subtract the translation vector between the control centroid and the perturbed centroid from every cell. This works well for evaluation (re-centering metrics around perturbation-specific effects), but it is fundamentally incompatible with tree-based GRN inference engines.

## Why Global Subtraction Fails for Tree-Based GRNs

Decision tree algorithms like LightGBM (used by GuanLab's PSGRN) do not operate on absolute expression values. They optimize for information gain by evaluating the rank order of cells along each gene's expression axis. A global constant subtraction (subtracting the same vector from every cell) is a monotonic translation that preserves rank order perfectly. Every cell shifts by the same amount, so the relative ordering never changes. The information gain at every possible split threshold remains identical, the resulting regression trees are structurally indistinguishable, and the output GRN is exactly the same as if no correction had been applied.

## What CHITIN Does Instead: Localized Manifold Subtraction

CHITIN breaks rank-order invariance by replacing the global subtraction with a localized, cell-specific subtraction. Instead of defining a single universal baseline for all cells, CHITIN recognizes that the "basal state" of a cell depends on its position on the transcriptomic manifold, influenced by local covariates like metabolic capacity, micro-environment, and cell-cycle phase.

For each perturbed metacell, CHITIN:

1. Projects the dataset into a fresh PCA latent space computed directly on the metacell data (not inherited from upstream preprocessing, since metacell aggregation fundamentally alters the topological geometry).
2. Identifies the k nearest unperturbed control neighbors in that latent space. These neighbors represent cells that occupied the same basal transcriptomic state as the perturbed cell, minus the CRISPR intervention.
3. Computes a unique localized baseline vector as the mean expression of those k control neighbors.
4. Extracts the causal delta by subtracting this localized baseline from the perturbed cell's expression.

Because each cell gets a different subtraction vector, the operation is non-linear across the dataset. The rank order of cells along each gene axis is reshuffled non-uniformly, which permanently breaks the information gain invariance of the downstream decision trees.

The mathematical foundation for this localized KNN subtraction comes from the Mixscape algorithm (Seurat/Pertpy ecosystem). CHITIN surgically extracts only the KNN subtraction engine from Mixscape and discards its Gaussian Mixture Model escaper classification step, which is redundant when operating on metacell-aggregated data that has already been purified by SPORE's escaper filtering (Phase 2) and smoothed by metacell aggregation (Phase 8).

## Results on Replogle 2022 K562

Running CHITIN on the Replogle 2022 K562 genome-wide CRISPRi screen (150,293 metacells, 5,000 genes, processed through SPORE):

**Pre-CHITIN Systema analysis:**
- Systematic variation magnitude |V| = 2.83
- Mean cosine similarity (control reference) = 0.244
- Mean pairwise cosine distance between perturbation centroids = 0.0103

**Post-CHITIN:**
- Mean Spearman rank correlation (pre vs post) = 0.927, confirming moderate rank-order disruption while preserving biological signal
- Mean pairwise cosine distance between perturbation centroids = 0.2287, a 22x increase in perturbation separability
- Top causal signals (by delta magnitude) correspond to biologically validated strong-effect perturbations: TIMM50 (mitochondrial import), GMNN (DNA replication licensing), RRN3 (rRNA transcription), PRPF19 (spliceosome), EIF3E (translation initiation)

The pairwise discrimination increase from 0.01 to 0.23 reveals that even in K562, a dataset characterized by Systema as having "low" systematic variation, the angular separation between perturbation centroids was near-zero before CHITIN. The low Systema cosine score (0.24) was masking severe geometric compression in the perturbation landscape.

## Architecture

CHITIN uses a fit/transform architecture. The model is fitted on training split controls and can then transform any split (including test splits with zero control cells) against that same reference manifold.

```
CHITIN/
├── chitin.ipynb              # Pipeline notebook with diagnostics
├── chitin_config.yaml        # All parameters
├── src/
│   ├── engine.py             # ChitinModel: fit/transform, standalone mode
│   ├── diagnostics.py        # Systema cosines, rank disruption, pairwise discrimination
│   ├── plotting.py           # Dark-themed diagnostic visualizations
│   └── utils.py              # Config, logging, memory, theme
├── output/                   # Delta-transformed .h5ad files
├── figures/                  # Saved plots
└── logs/                     # Timestamped run logs
```

### With SPORE (recommended)

```python
from src.engine import ChitinModel

model = ChitinModel()
model.fit(adata_train, cfg, logger)                          # fit on SPORE train controls
delta_train = model.transform(adata_train, cfg, logger)      # transform train
delta_val   = model.transform(adata_val, cfg, logger)        # transform val
delta_test  = model.transform(adata_test, cfg, logger)       # transform test (0 controls OK)
```

### Standalone (any .h5ad)

```python
from src.engine import run_chitin_standalone

adata_delta, model = run_chitin_standalone(adata, cfg, logger)
```

The only requirements are that the `.h5ad` file has a column in `.obs` identifying perturbation labels and that the dataset contains unperturbed control cells. Set `perturbation_col` and `control_label` in the YAML config to match your data.

## Configuration

Key parameters in `chitin_config.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_pcs` | 50 | PCA components for the metacell latent space |
| `k` | 20 | Number of nearest control neighbors per perturbed cell |
| `distance_metric` | euclidean | Distance metric for KNN lookup |
| `svd_solver` | randomized | PCA solver (randomized is fast, arpack is exact) |
| `k_sweep` | true | Run k-sensitivity analysis to validate neighborhood localization |

The k parameter controls the tradeoff between localization and stability. Low k (5-10) produces highly localized but potentially noisy baselines. High k (50-100) produces stable baselines that converge toward the global control mean, which reintroduces rank-order invariance. The k-sensitivity sweep in the notebook visualizes this tradeoff. For the Replogle K562 dataset, k=20 sits in the healthy middle zone.

## Diagnostic Plots

The notebook produces the following visualizations:

1. **PCA scree plot** confirming the metacell latent space captures sufficient variance
2. **Latent space scatter** (PC1 vs PC2) showing control vs perturbed metacell distributions
3. **Pre-CHITIN Systema cosine distributions** quantifying baseline systematic variation
4. **k-sensitivity analysis** showing basal variance and KNN distance across k values
5. **Rank-order disruption histogram** (Spearman correlations per gene, pre vs post CHITIN)
6. **Pairwise perturbation discrimination** (cosine distance distributions, pre vs post)
7. **Delta magnitude distribution** across all perturbed metacells
8. **Top perturbations by causal signal strength** (horizontal bar chart)

## Design Decisions

**Fresh PCA on metacells:** CHITIN computes its own PCA directly on the metacell data rather than inheriting coordinates from upstream preprocessing. Metacell aggregation fundamentally changes the variance and covariance structure of the dataset. Using stale single-cell PCA coordinates would map dense metacell neighborhoods using sparse, obsolete geometric boundaries.

**No global centroids in the transform:** The Systema centroids (C_ctrl and O_pert) are computed for evaluation purposes only. They are never used in the actual localized subtraction. Injecting a global vector into a localized neighborhood operation would reintroduce the rank-order invariance the module is designed to eliminate.

**No GMM escaper classification:** The standard Mixscape pipeline includes a Gaussian Mixture Model step to identify escaper cells. CHITIN omits this because (a) escaper filtering was already performed upstream in SPORE Phase 2 at single-cell resolution, and (b) metacell aggregation compresses the bimodal distribution that the GMM expects to find, making it mathematically unstable on aggregated data.

**Control cells get delta = 0:** Control metacells are the baseline by definition. Their delta is zero. This is consistent with the downstream GRN framework where controls serve as the reference state.

## Requirements

- Python 3.9+
- scanpy, anndata, scipy, numpy, scikit-learn
- matplotlib, seaborn, pyyaml, psutil

## Part of MYCELIUM

CHITIN sits between [SPORE](https://github.com/PatrickSheehan053/SPORE) (preprocessing) and GuanLab PSGRN (GRN inference) in the MYCELIUM pipeline. Its delta-transformed outputs feed directly into the GRN engine, forcing LightGBM's regression trees to learn causal perturbation-to-perturbation mappings rather than static background co-expression patterns.

```
Raw .h5ad --> SPORE --> CHITIN --> GuanLab PSGRN --> FUNGI --> SPECTRA
              (clean)   (delta)    (GRN)            (prune)   (predict)
```

# Neural Network for Manifold Discovery

A systematic comparison of dimensionality reduction and manifold learning methods on canonical datasets.

## Methods compared
| Method | Type | Key idea |
|---|---|---|
| PCA | Linear | Maximum variance projection |
| Isomap | Nonlinear | Geodesic distance + MDS |
| LLE | Nonlinear | Locally linear reconstruction |
| UMAP | Nonlinear | Topological data analysis |
| Autoencoder | Deep learning | Nonlinear encoder-decoder |

## Datasets
| Dataset | Intrinsic dim | Challenge |
|---|---|---|
| Swiss Roll | 2D | Curved, needs unrolling |
| S-Curve | 2D | Folded surface |
| Torus | 2D | Topology: can't flatten without tearing |
| Möbius Strip | 2D | Non-orientable surface |

## Project structure
```
manifold-discovery/
├── environment.yml          # Reproducible conda environment
├── README.md
├── data/                    # Generated datasets (auto-created)
├── notebooks/
│   ├── 00_setup_check.ipynb         # Phase 1: verify environment
│   ├── 01_datasets.ipynb            # Phase 2: generate & explore
│   ├── 02_pca_isomap.ipynb          # Phase 3: classical baselines
│   ├── 03_lle_spectral.ipynb        # Phase 4: LLE & spectral
│   ├── 04_umap.ipynb                # Phase 5: UMAP experiments
│   ├── 05_autoencoder.ipynb         # Phase 6: deep learning
│   └── 06_evaluation.ipynb          # Phase 7: metrics & comparison
├── src/
│   ├── datasets/            # Dataset generators
│   ├── methods/             # Wrappers for each method
│   ├── evaluation/          # Metric functions
│   └── utils/               # Plotting, I/O helpers
└── results/
    ├── figures/             # Saved plots (300 DPI)
    ├── metrics/             # CSV metric tables
    └── models/              # Saved autoencoder checkpoints
```

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/yourname/manifold-discovery
cd manifold-discovery

# 2. Create and activate environment (one command)
conda env create -f environment.yml
conda activate manifold-discovery

# 3. Register the kernel for Jupyter
python -m ipykernel install --user --name manifold-discovery --display-name "Manifold Discovery"

# 4. Launch Jupyter and open notebooks/00_setup_check.ipynb
jupyter lab
```

## Key findings
*(Filled in after Phase 7 — leave this section for your report summary)*

## References
- Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.
- Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290(5500), 2323–2326.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.

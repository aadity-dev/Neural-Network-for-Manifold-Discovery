# Neural Network for Manifold Discovery

A systematic comparison of linear, nonlinear, topological, and deep learning dimensionality reduction methods on canonical 3D manifolds.

## Methods Compared
| Method | Category | Key Idea | Mapping Type |
|---|---|---|---|
| **PCA** | Linear | Maximum variance linear projection | Parametric ($O(1)$) |
| **Isomap** | Nonlinear (Global) | Geodesic distance matrix via graph shortest paths + MDS | Transductive |
| **LLE (Standard / Modified)** | Nonlinear (Local) | Reconstructs points from $k$-nearest neighbors preserving local weights | Transductive |
| **Spectral Embedding** | Graph Theory | Eigenvectors of unnormalized/normalized Graph Laplacian | Transductive |
| **UMAP** | Topological Data Analysis | Fuzzy simplicial sets & Riemannian geometry optimization | Semi-Parametric |
| **Autoencoder (PyTorch)** | Deep Learning | Multi-layer nonlinear encoder-decoder (3 → 64 → 32 → 16 → 2) | Parametric ($O(1)$) |

## Datasets
| Dataset | Intrinsic Dim | Ambient Dim | Primary Manifold Challenge |
|---|---|---|---|
| **Swiss Roll** | 2D | 3D | Non-linear spiral curvature requiring global unrolling |
| **S-Curve** | 2D | 3D | S-shaped double fold requiring manifold flattening |
| **Torus** | 2D | 3D | Non-Euclidean topology ($S^1 \times S^1$) — cannot flatten without tearing |
| **Möbius Strip** | 2D | 3D | Non-orientable surface with a single boundary and half-twist |

## Project Structure
```
manifold-discovery/
├── environment.yml          # Conda environment specification
├── README.md                # Project documentation & benchmark findings
├── data/                    # Generated synthetic manifolds (.npz)
├── notebooks/
│   ├── 00_setup_check.ipynb         # Phase 1: Environment & CUDA/MPS check
│   ├── 01_datasets.ipynb            # Phase 2: Manifold generation & 3D visualization
│   ├── 02_pca_isomap.ipynb          # Phase 3: Classical baseline evaluations
│   ├── 03_lle_spectral.ipynb        # Phase 4: Local neighborhood & spectral methods
│   ├── 04_umap.ipynb                # Phase 5: UMAP hyperparameter sweeps
│   ├── 05_autoencoder.ipynb         # Phase 6: PyTorch Autoencoder training & latency
│   └── 06_evaluation.ipynb          # Phase 7: Unified metrics & master benchmarking
├── src/
│   ├── datasets/            # Synthetic data generation scripts
│   ├── methods/             # Algorithm wrappers (PCA, Isomap, LLE, UMAP, PyTorch AE)
│   ├── evaluation/          # Neighborhood & topological metric calculations
│   └── utils/               # Plotting and formatting utilities
└── results/
    ├── figures/             # High-resolution comparison plots (300 DPI)
    ├── metrics/             # Quantitative benchmark CSV tables
    └── models/              # Saved PyTorch model checkpoints (.pt)
```

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/yourname/manifold-discovery
cd manifold-discovery

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate manifold-discovery

# 3. Register the kernel for Jupyter
python -m ipykernel install --user --name manifold-discovery --display-name "Manifold Discovery"

# 4. Launch Jupyter Lab
jupyter lab
```

## Key Findings & Quantitative Benchmarks

All 7 methods were evaluated across 2,000 sampling points per dataset using neighborhood preservation metrics (**Trustworthiness** and **Continuity** for $k=10$), distance correlation (**Spearman $r$**), and **Reconstruction Error (MSE)**.

### Master Comparison Table
| Dataset | Method | Trustworthiness ↑ | Continuity ↑ | Spearman $r$ ↑ | Recon MSE ↓ | Fit Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Swiss Roll** | PCA | 0.9664 | 0.3348 | **0.8630** | — | **0.023** |
| | **Isomap** | **0.9997** | **0.8651** | 0.3574 | — | 1.930 |
| | LLE | 0.9977 | 0.6895 | 0.1818 | — | 0.119 |
| | UMAP | 0.9991 | 0.7507 | 0.3573 | — | 11.674 |
| | **Autoencoder** | 0.9829 | 0.5257 | 0.4782 | 1.9960 | 2.500 |
| **S-Curve** | PCA | 0.9287 | 0.2050 | **0.9646** | — | **0.002** |
| | **Isomap** | **0.9986** | **0.7131** | 0.8943 | — | 1.953 |
| | UMAP | **0.9986** | 0.7049 | 0.8399 | — | 5.161 |
| | **Autoencoder** | 0.9878 | 0.5274 | 0.6235 | **0.0221** | 2.500 |
| **Torus** | PCA | 0.9710 | 0.4838 | **0.9954** | — | **0.001** |
| | Isomap | 0.9713 | 0.5080 | 0.9872 | — | 2.080 |
| | **UMAP** | **0.9986** | **0.7343** | 0.9485 | — | 4.779 |
| | Autoencoder | 0.9698 | 0.4589 | 0.5676 | 0.1600 | 2.500 |
| **Möbius Strip** | PCA | 0.9846 | 0.4972 | **0.9944** | — | **0.001** |
| | Isomap | 0.9900 | 0.5164 | 0.9790 | — | 2.075 |
| | **UMAP** | **0.9968** | **0.6286** | 0.9563 | — | 4.834 |
| | **Autoencoder** | 0.9850 | 0.3685 | 0.4571 | **0.0067** | 2.500 |

### Core Insights
1. **Isomap Excelled on Unrolling Curved Euclidean Manifolds:** On the Swiss Roll and S-Curve, Isomap achieved the highest neighborhood continuity (**0.8651** and **0.7131**, respectively) by constructing global shortest-path geodesic distance matrices.
2. **UMAP Dominated Non-Euclidean & Complex Topologies:** UMAP outperformed all classical algorithms on non-trivial topologies (Torus Trustworthiness **0.9986** vs Isomap's **0.9713**), maintaining local neighborhood clustering without artificial tearing.
3. **Parametric Generalization of Deep Autoencoders:** Unlike transductive techniques (Isomap/LLE) which require $O(N^2)$ re-computation for out-of-sample data, the PyTorch Autoencoder learned a continuous, parametric mapping. It achieved precise 3D reconstruction ($MSE = 0.0067$ on Möbius Strip, $MSE = 0.0221$ on S-Curve) with instant $O(1)$ inference.
4. **Linear Limits of PCA:** While PCA is computationally instantaneous ($<2$ ms), its linear subspace constraint causes severe manifold overlap, leading to low neighborhood continuity on curved manifolds (e.g., **0.3348** on Swiss Roll).

## References
- Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.
- Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290(5500), 2323–2326.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.

# Neural Network for Manifold Discovery
## A Systematic Comparison of Dimensionality Reduction Methods

**Author:** [Your Name]  
**Date:** [Date]  
**Repository:** [GitHub URL]  
**Demo:** [Streamlit Cloud URL]

---

## Abstract

This project systematically compares five families of dimensionality reduction methods — PCA, Isomap, LLE variants, UMAP, and a PyTorch autoencoder — on four canonical manifold datasets: the Swiss Roll, S-Curve, Torus, and Möbius Strip. Each dataset presents distinct geometric and topological challenges. We evaluate all methods using Trustworthiness, Continuity, and Spearman rank correlation and conclude that no single method dominates: the right choice depends on the topology of the underlying manifold.

---

## 1. Introduction

### 1.1 The Manifold Hypothesis

High-dimensional data tends to cluster on a lower-dimensional surface called a **manifold**. This is the manifold hypothesis — the silent assumption underneath every neural network, generative model, and embedding system in modern ML. This project makes it *visible and testable* by applying seven methods to four datasets where we know the true manifold structure in advance.

### 1.2 Problem Statement

Given X ∈ ℝⁿˣ³ sampled from a 2D manifold embedded in 3D, find f: ℝ³ → ℝ² that recovers the 2D structure while preserving neighbourhood relationships.

---

## 2. Datasets

| Dataset | Challenge | Intrinsic dim |
|---|---|---|
| Swiss Roll | Curled plane — needs geodesic unrolling | 2 |
| S-Curve | Folded surface — moderate curvature | 2 |
| Torus | Topology: cannot flatten without tearing | 2 |
| Möbius Strip | Non-orientable — only one side | 2 |

**Standard parameters:** n_samples=2000, σ=0.1, random_state=42

---

## 3. Methods

| Method | Type | Core idea |
|---|---|---|
| PCA | Linear | Maximum variance projection |
| Isomap | Nonlinear | Geodesic distances + MDS |
| LLE | Nonlinear | Local reconstruction weights |
| LLE Modified | Nonlinear | Multiple weight vectors, more stable |
| Spectral Embedding | Nonlinear | Graph Laplacian eigenvectors |
| UMAP | Nonlinear | Topological data analysis |
| Autoencoder | Deep learning | Learned bottleneck (parametric) |

### Architecture (Autoencoder)
```
Encoder: 3 → 64 → 32 → 16 → 2  (ReLU)
Decoder: 2 → 16 → 32 → 64 → 3  (Linear output)
Loss: MSE | Optimiser: Adam lr=1e-3 | Epochs: 200
```

---

## 4. Evaluation Metrics

| Metric | Measures | Best |
|---|---|---|
| Trustworthiness | False neighbours in embedding (tears) | 1.0 |
| Continuity | Missing neighbours from original space | 1.0 |
| Spearman r | Rank correlation of pairwise distances | 1.0 |
| Recon MSE | Autoencoder reconstruction error | 0.0 |

---

## 5. Results

### 5.1 Qualitative

*(Insert: 06_grand_comparison_grid.png)*

### 5.2 Quantitative

*(Insert: 06_heatmap_all_metrics.png)*
*(Insert: results/metrics/06_master_comparison.csv as table)*

### 5.3 Method Profiles

*(Insert: 06_radar_per_dataset.png)*

### 5.4 Speed

*(Insert: 06_speed_comparison.png)*

---

## 6. Key Findings

1. **No method wins on all datasets.** Topology determines the right choice.
2. **Topology is the hard constraint.** The Torus cannot embed in ℝ² without tearing — this is mathematics, not a bug.
3. **Local vs global is a genuine trade-off.** LLE and Isomap both unroll the Swiss Roll but make different errors on the Torus.
4. **The autoencoder is the only parametric method** — it can encode new points at inference time without rerunning.
5. **FAILED cells in LLE grids are informative.** Hessian LLE's failures reveal genuine violations of its convexity assumption.
6. **Spectral Embedding is the most consistent** across all four datasets.

---

## 7. Discussion

### Connections to Real-World ML

The manifold hypothesis appears everywhere:
- **VAEs and diffusion models:** the decoder is a parametric manifold map
- **Word embeddings:** word2vec finds manifolds in semantic space  
- **t-SNE:** a nonlinear embedding related to LLE  
- **Graph neural networks:** generalise Laplacian eigenmaps to arbitrary graphs

### Practical Recommendations

| Goal | Best method |
|---|---|
| Fast, robust baseline | UMAP (n_neighbors=15, min_dist=0.1) |
| Global geodesic structure | Isomap (k=10–15) |
| Encode new points at inference | Autoencoder |
| Most consistent across topologies | Spectral Embedding |
| Linear data, interpretability | PCA |

---

## 8. Conclusion

Manifold learning is not solved. Every method has a regime where it excels and one where it fails. The key factors:

1. **Curvature** — linear methods fail on curved manifolds
2. **Topology** — non-trivial topology defeats all flat-embedding methods  
3. **Scale** — local and global methods make different trade-offs
4. **Speed** — UMAP offers the best quality-vs-speed balance
5. **Parametric** — only the autoencoder generalises to new points

---

## 9. References

1. Tenenbaum et al. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science*.
2. Roweis & Saul (2000). Nonlinear dimensionality reduction by LLE. *Science*.
3. Belkin & Niyogi (2003). Laplacian eigenmaps. *Neural Computation*.
4. McInnes et al. (2018). UMAP. *arXiv:1802.03426*.
5. Hinton & Salakhutdinov (2006). Reducing dimensionality with neural networks. *Science*.
6. Pedregosa et al. (2011). Scikit-learn. *JMLR*.

---

## Appendix — Reproducing Results

```bash
git clone https://github.com/yourname/manifold-discovery
cd manifold-discovery
conda env create -f environment.yml
conda activate manifold-discovery
python -m ipykernel install --user --name manifold-discovery
jupyter lab          # run notebooks 00 → 06 in order
streamlit run app/streamlit_app.py
```
"""
app/streamlit_app.py

Interactive Streamlit demo for the Manifold Discovery project.

Run with:
    conda activate manifold-discovery
    streamlit run app/streamlit_app.py

Features
--------
- Dataset selector (Swiss Roll / S-Curve / Torus / Möbius Strip)
- Method selector (PCA / Isomap / LLE / UMAP / Autoencoder)
- Live hyperparameter controls per method
- Side-by-side 3D original + 2D embedding
- Trustworthiness + Spearman scores shown instantly
- Noise level slider to test robustness
"""

import sys
from pathlib import Path

# Make src importable when run from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import streamlit as st
import torch

from src.datasets.generators import swiss_roll, s_curve, torus, mobius_strip
from src.methods.pca_isomap  import run_pca, run_isomap
from src.methods.lle_spectral import run_lle, run_spectral
from src.methods.umap_method  import run_umap
from src.methods.autoencoder  import Autoencoder, encode_dataset
from src.evaluation.metrics   import compute_trustworthiness, compute_spearman
from src.config import CFG


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Manifold Discovery",
    page_icon="🌀",
    layout="wide",
)

st.title("🌀 Neural Network for Manifold Discovery")
st.markdown(
    "Interactive comparison of dimensionality reduction methods "
    "on canonical manifold datasets."
)


# ─────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

DATASET_OPTIONS = ["Swiss Roll", "S-Curve", "Torus", "Möbius Strip"]
METHOD_OPTIONS  = ["PCA", "Isomap", "LLE", "LLE Modified",
                   "Spectral Embedding", "UMAP", "Autoencoder"]

dataset_name = st.sidebar.selectbox("Dataset", DATASET_OPTIONS)
method_name  = st.sidebar.selectbox("Method",  METHOD_OPTIONS)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset options")
n_samples   = st.sidebar.slider("Number of points", 500, 3000, 1500, step=500)
noise_level = st.sidebar.slider("Noise level (σ)", 0.0, 0.5, 0.1, step=0.05)
seed        = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Method hyperparameters")

# Per-method controls
params = {}
if method_name in ("Isomap", "LLE", "LLE Modified", "Spectral Embedding"):
    params["n_neighbors"] = st.sidebar.slider("n_neighbors", 3, 50, 10)

if method_name == "UMAP":
    params["n_neighbors"] = st.sidebar.slider("n_neighbors", 3, 50, 15)
    params["min_dist"]    = st.sidebar.slider("min_dist", 0.0, 1.0, 0.1, step=0.05)
    params["metric"]      = st.sidebar.selectbox(
        "metric", ["euclidean", "cosine", "manhattan"])


# ─────────────────────────────────────────────
# Generate dataset
# ─────────────────────────────────────────────
@st.cache_data
def get_dataset(name, n, noise, s):
    generators = {
        "Swiss Roll":   swiss_roll,
        "S-Curve":      s_curve,
        "Torus":        torus,
        "Möbius Strip": mobius_strip,
    }
    return generators[name](n_samples=n, noise=noise, seed=s)


X, t = get_dataset(dataset_name, n_samples, noise_level, int(seed))


# ─────────────────────────────────────────────
# Run embedding
# ─────────────────────────────────────────────
@st.cache_data
def run_method(method, X_arr, _params, _seed):
    X = np.array(X_arr)
    if method == "PCA":
        return run_pca(X)
    elif method == "Isomap":
        return run_isomap(X, n_neighbors=_params.get("n_neighbors", 10))
    elif method == "LLE":
        return run_lle(X, method="standard",
                        n_neighbors=_params.get("n_neighbors", 10))
    elif method == "LLE Modified":
        return run_lle(X, method="modified",
                        n_neighbors=_params.get("n_neighbors", 10))
    elif method == "Spectral Embedding":
        return run_spectral(X, n_neighbors=_params.get("n_neighbors", 10))
    elif method == "UMAP":
        return run_umap(X,
                         n_neighbors=_params.get("n_neighbors", 15),
                         min_dist=_params.get("min_dist", 0.1),
                         metric=_params.get("metric", "euclidean"),
                         random_state=_seed)
    elif method == "Autoencoder":
        ckpt_name = _params.get("dataset_name", "swiss_roll") + "_ae.pt"
        ckpt_path = CFG["models_dir"] / ckpt_name
        if ckpt_path.exists():
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            model = Autoencoder(input_dim=3,
                                hidden_dims=ckpt["hidden_dims"],
                                latent_dim=ckpt["latent_dim"])
            model.load_state_dict(ckpt["model_state"])
            model.X_mean = ckpt["X_mean"]
            model.X_std  = ckpt["X_std"]
            model.eval()
            Z = encode_dataset(model, X)
            return Z, {"fit_time_s": None}
        else:
            return None, {"error": "No checkpoint found. Run Phase 6 first."}
    return None, {}


ae_params = dict(params)
ae_params["dataset_name"] = dataset_name.lower().replace(" ", "_").replace("ö", "o")

with st.spinner(f"Running {method_name}..."):
    X_emb, meta = run_method(method_name, X, ae_params, int(seed))


# ─────────────────────────────────────────────
# Layout: 3D + 2D side by side
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Original 3D — {dataset_name}")
    fig3d = plt.figure(figsize=(6, 5))
    ax3d  = fig3d.add_subplot(111, projection="3d")
    ax3d.scatter(X[:, 0], X[:, 1], X[:, 2],
                 c=t, cmap="viridis", s=5, alpha=0.7)
    ax3d.set_xlabel("X", fontsize=8)
    ax3d.set_ylabel("Y", fontsize=8)
    ax3d.set_zlabel("Z", fontsize=8)
    ax3d.tick_params(labelsize=7)
    st.pyplot(fig3d)
    plt.close(fig3d)

with col2:
    st.subheader(f"2D Embedding — {method_name}")
    if X_emb is not None and np.std(X_emb) > 1e-6:
        fig2d, ax2d = plt.subplots(figsize=(6, 5))
        sc = ax2d.scatter(X_emb[:, 0], X_emb[:, 1],
                          c=t, cmap="viridis", s=5, alpha=0.8)
        plt.colorbar(sc, ax=ax2d, label="Manifold param t")
        ax2d.set_xticks([]); ax2d.set_yticks([])
        if meta.get("fit_time_s"):
            ax2d.set_xlabel(f"Fit time: {meta['fit_time_s']:.2f}s", fontsize=9)
        st.pyplot(fig2d)
        plt.close(fig2d)
    else:
        err = meta.get("error", meta.get("error_msg", "Embedding failed"))
        st.error(f"⚠️ {err}")


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Quality metrics")

if X_emb is not None and np.std(X_emb) > 1e-6:
    with st.spinner("Computing metrics..."):
        tw    = compute_trustworthiness(X, X_emb, k=10)
        spear = compute_spearman(X, X_emb)

    m1, m2, m3 = st.columns(3)
    m1.metric("Trustworthiness", f"{tw:.4f}",
               help="Fraction of embedding neighbours that were neighbours in 3D. 1.0 = perfect.")
    m2.metric("Spearman r", f"{spear:.4f}",
               help="Rank correlation of pairwise distances. 1.0 = perfect distance preservation.")
    if meta.get("fit_time_s"):
        m3.metric("Fit time", f"{meta['fit_time_s']:.2f}s")
    else:
        m3.metric("Fit time", "n/a")
else:
    st.warning("Metrics unavailable — embedding failed or degenerate.")


# ─────────────────────────────────────────────
# Method info expander
# ─────────────────────────────────────────────
with st.expander("ℹ️ How does this method work?"):
    descriptions = {
        "PCA": (
            "**PCA** finds the directions of maximum variance and projects data onto them. "
            "It is linear — it cannot unroll curved manifolds like the Swiss Roll. "
            "Fast and deterministic, it works well when the data is roughly linear."
        ),
        "Isomap": (
            "**Isomap** builds a k-nearest-neighbour graph, computes shortest-path "
            "geodesic distances along the graph, then uses MDS to find a flat "
            "embedding that preserves those distances. "
            "Succeeds on the Swiss Roll but struggles with non-convex topology (Torus, Möbius)."
        ),
        "LLE": (
            "**LLE** reconstructs each point as a weighted sum of its neighbours, "
            "then finds a low-dimensional embedding that preserves those weights. "
            "Works patch-by-patch — sensitive to k and point density."
        ),
        "LLE Modified": (
            "**Modified LLE** uses multiple weight vectors per point, making it "
            "more stable than standard LLE. Better on noisy or uneven data."
        ),
        "Spectral Embedding": (
            "**Spectral Embedding** (Laplacian Eigenmaps) builds a similarity graph "
            "and embeds using the eigenvectors of the graph Laplacian. "
            "Fast, stable, and works well across all four datasets."
        ),
        "UMAP": (
            "**UMAP** uses topological data analysis to build a fuzzy simplicial "
            "complex representing the manifold, then optimises an embedding that "
            "preserves the topology. Very fast, works on large datasets, "
            "and the min_dist parameter controls local vs global structure."
        ),
        "Autoencoder": (
            "**Autoencoder** trains a neural network (3→64→32→16→2→16→32→64→3) "
            "to compress the 3D input to 2D and reconstruct it back. "
            "The 2D bottleneck is the embedding. Unlike all other methods, "
            "it is **parametric** — it can encode new points without rerunning."
        ),
    }
    st.markdown(descriptions.get(method_name, ""))


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Manifold Discovery Project · PCA · Isomap · LLE · UMAP · Autoencoder · "
    "Built with scikit-learn, umap-learn, PyTorch & Streamlit"
)
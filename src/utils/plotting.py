"""
src/utils/plotting.py

Shared plotting helpers used across all notebooks.

Functions
---------
plot_3d(X, t, title, ax, cmap)      — single 3D scatter on a given Axes3D
plot_all_3d(datasets, save_path)    — 2x2 grid of all four 3D datasets
plot_2d(X_emb, t, title, ax, cmap) — single 2D embedding scatter
savefig(fig, path, dpi)             — save with sensible defaults
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


CMAP       = "viridis"
POINT_SIZE = 6
ALPHA      = 0.75


def plot_3d(X: np.ndarray, t: np.ndarray, title: str = "",
            ax=None, cmap: str = CMAP):
    """
    3D scatter plot of a single dataset.

    Parameters
    ----------
    X     : (n, 3) array
    t     : (n,)   colour parameter, normalised [0, 1]
    title : axes title
    ax    : existing Axes3D (created if None)
    cmap  : matplotlib colormap name
    """
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(6, 5))
        ax  = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                    c=t, cmap=cmap, s=POINT_SIZE, alpha=ALPHA)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z", fontsize=9)
    ax.tick_params(labelsize=7)

    if standalone:
        plt.colorbar(sc, ax=ax, label="Manifold param t", pad=0.1)
        plt.tight_layout()
    return ax


def plot_all_3d(datasets: dict, save_path=None, dpi: int = 150):
    """
    2x2 grid of 3D scatter plots for all four datasets.

    Parameters
    ----------
    datasets  : dict returned by load_all_datasets()
    save_path : optional path to save PNG
    dpi       : output resolution
    """
    names = list(datasets.keys())
    fig   = plt.figure(figsize=(13, 10))
    fig.suptitle("Manifold Datasets — 3D Views", fontsize=14, y=0.98)

    for i, name in enumerate(names):
        X, t = datasets[name]
        ax   = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                   c=t, cmap=CMAP, s=POINT_SIZE, alpha=ALPHA)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save_path is not None:
        savefig(fig, save_path, dpi=dpi)
    return fig


def plot_2d(X_emb: np.ndarray, t: np.ndarray, title: str = "",
            ax=None, cmap: str = CMAP):
    """
    2D scatter of an embedding result (used in Phases 3-6).

    Parameters
    ----------
    X_emb : (n, 2) array — embedding coordinates
    t     : (n,)   colour parameter
    title : axes title
    ax    : existing Axes (created if None)
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))

    sc = ax.scatter(X_emb[:, 0], X_emb[:, 1],
                    c=t, cmap=cmap, s=POINT_SIZE, alpha=ALPHA)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    if standalone:
        plt.colorbar(sc, ax=ax, label="t")
        plt.tight_layout()
    return ax


def savefig(fig, path, dpi: int = 300):
    """Save figure to disk with tight layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved -> {path}")
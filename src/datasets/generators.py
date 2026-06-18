"""
src/datasets/generators.py

Four canonical manifold dataset generators:
  - Swiss Roll       (sklearn)
  - S-Curve          (sklearn)
  - Torus            (parametric)
  - Möbius Strip     (parametric)

All return (X, t) where:
  X : np.ndarray of shape (n_samples, 3)  — 3D point cloud
  t : np.ndarray of shape (n_samples,)    — ground-truth manifold parameter
      used for colouring; values in [0, 1]

Usage:
    from src.datasets.generators import load_all_datasets
    datasets = load_all_datasets(n_samples=2000, noise=0.1, seed=42)
"""

import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve


def swiss_roll(n_samples: int = 2000, noise: float = 0.1, seed: int = 42):
    """Sklearn Swiss Roll. Intrinsic dim = 2."""
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    t_norm = (t - t.min()) / (t.max() - t.min())
    return X, t_norm


def s_curve(n_samples: int = 2000, noise: float = 0.1, seed: int = 42):
    """Sklearn S-Curve. Intrinsic dim = 2."""
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
    t_norm = (t - t.min()) / (t.max() - t.min())
    return X, t_norm


def torus(n_samples: int = 2000, noise: float = 0.1, seed: int = 42,
          R: float = 3.0, r: float = 1.0):
    """
    Torus parametrised by angles (u, v):
        x = (R + r*cos v)*cos u
        y = (R + r*cos v)*sin u
        z = r*sin v

    Returns t = u (the major angle), normalised to [0, 1].
    Topology challenge: cannot be flattened without tearing.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n_samples)
    v = rng.uniform(0, 2 * np.pi, n_samples)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    X = np.column_stack([x, y, z])
    X += rng.normal(0, noise, X.shape)

    t_norm = (u - u.min()) / (u.max() - u.min())
    return X, t_norm


def mobius_strip(n_samples: int = 2000, noise: float = 0.05, seed: int = 42,
                 R: float = 3.0, w: float = 1.0):
    """
    Möbius strip parametrised by (t_param, s):
        t_param in [0, 2pi]    — position along the central circle
        s       in [-w/2, w/2] — position across the width

        x = (R + s*cos(t/2))*cos t
        y = (R + s*cos(t/2))*sin t
        z = s*sin(t/2)

    Non-orientable — the strip has only one side and one edge.
    """
    rng = np.random.default_rng(seed)
    t_param = rng.uniform(0, 2 * np.pi, n_samples)
    s_param = rng.uniform(-w / 2, w / 2, n_samples)

    x = (R + s_param * np.cos(t_param / 2)) * np.cos(t_param)
    y = (R + s_param * np.cos(t_param / 2)) * np.sin(t_param)
    z = s_param * np.sin(t_param / 2)

    X = np.column_stack([x, y, z])
    X += rng.normal(0, noise, X.shape)

    t_norm = (t_param - t_param.min()) / (t_param.max() - t_param.min())
    return X, t_norm


def load_all_datasets(n_samples: int = 2000, noise: float = 0.1,
                      seed: int = 42) -> dict:
    """
    Convenience loader — returns all four datasets as a dict.

    Returns
    -------
    {
      "Swiss Roll":   (X, t),
      "S-Curve":      (X, t),
      "Torus":        (X, t),
      "Mobius Strip": (X, t),
    }
    """
    return {
        "Swiss Roll":   swiss_roll(n_samples, noise, seed),
        "S-Curve":      s_curve(n_samples, noise, seed),
        "Torus":        torus(n_samples, noise, seed),
        "Mobius Strip": mobius_strip(n_samples, noise, seed),
    }
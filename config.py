"""
Global project configuration.
Import anywhere with:  from src.config import CFG
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CFG = {
    # Paths
    "root":         ROOT,
    "data_dir":     ROOT / "data",
    "figures_dir":  ROOT / "results" / "figures",
    "metrics_dir":  ROOT / "results" / "metrics",
    "models_dir":   ROOT / "results" / "models",

    # Reproducibility
    "random_seed":  42,

    # Dataset defaults
    "n_samples":    2000,
    "noise_levels": [0.0, 0.1, 0.3],

    # Figure output
    "figsize":      (8, 6),
    "dpi":          300,

    # Training device — overridden in notebook 05 at runtime
    "device":       "cpu",
}

# python3 setup_project.py"""
# Run once after cloning to create all required directories
# and write empty __init__.py files for the src package.

# Usage:
#     python src/utils/setup_project.py
# """
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

dirs = [
    ROOT / "data",
    ROOT / "notebooks",
    ROOT / "src" / "datasets",
    ROOT / "src" / "methods",
    ROOT / "src" / "evaluation",
    ROOT / "src" / "utils",
    ROOT / "results" / "figures",
    ROOT / "results" / "metrics",
    ROOT / "results" / "models",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"✅  {d.relative_to(ROOT)}")

# Write __init__.py for each src sub-package
for pkg in ["src", "src/datasets", "src/methods", "src/evaluation", "src/utils"]:
    init = ROOT / pkg / "__init__.py"
    if not init.exists():
        init.write_text("")

# Write global config
config = ROOT / "src" / "config.py"
config.write_text('''\
"""
Global project configuration.
Import anywhere with:  from src.config import CFG
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CFG = {
    "root":         ROOT,
    "data_dir":     ROOT / "data",
    "figures_dir":  ROOT / "results" / "figures",
    "metrics_dir":  ROOT / "results" / "metrics",
    "models_dir":   ROOT / "results" / "models",
    "random_seed":  42,
    "n_samples":    2000,
    "noise_levels": [0.0, 0.1, 0.3],
    "figsize":      (8, 6),
    "dpi":          300,
    "device":       "cpu",   # updated at runtime in notebook 05
}
''')
print("✅  src/config.py")
print("\n🎉  Project structure ready. Open notebooks/00_setup_check.ipynb to verify.")

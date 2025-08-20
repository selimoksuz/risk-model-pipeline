"""Entry point for running the CLI without installing the package.

This script ensures that the ``src`` directory is added to ``sys.path`` so
that ``risk_pipeline`` can be imported even when the project hasn't been
installed. It mirrors the behaviour of the ``risk-pipeline`` console script
defined in ``pyproject.toml``.
"""

from pathlib import Path
import sys

# Add ``src`` to ``sys.path`` to allow running without ``PYTHONPATH``.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from risk_pipeline.cli import app

if __name__ == "__main__":
    app()

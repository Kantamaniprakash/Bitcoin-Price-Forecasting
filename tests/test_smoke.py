"""
Smoke tests
===========
Trivially-passing tests that require no network access, API keys, or heavy
model fitting. They verify that:

  1. Every source module under ``src/`` is syntactically valid (parses with
     ``ast``) — this catches broken commits without importing heavy deps.
  2. The key project files are present.

Running ``main()``/``backtest`` is intentionally avoided because those hit
Yahoo Finance over the network.
"""

import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")

SRC_MODULES = [
    "__init__.py",
    "data_collection.py",
    "preprocessing.py",
    "granger_causality.py",
    "arima_model.py",
    "var_model.py",
    "backtesting.py",
    "visualization.py",
]


def test_src_modules_parse():
    """Every src module should parse as valid Python."""
    for name in SRC_MODULES:
        path = os.path.join(SRC_DIR, name)
        assert os.path.isfile(path), f"missing source module: {name}"
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        # Raises SyntaxError if the module is not valid Python.
        ast.parse(source, filename=path)


def test_entrypoint_scripts_parse():
    """Top-level runner scripts should parse as valid Python."""
    for name in ("main.py", "backtest.py"):
        path = os.path.join(REPO_ROOT, name)
        assert os.path.isfile(path), f"missing entrypoint: {name}"
        with open(path, "r", encoding="utf-8") as fh:
            ast.parse(fh.read(), filename=path)


def test_key_files_exist():
    """Key project files should be present."""
    for rel in ("README.md", "requirements.txt", "LICENSE"):
        assert os.path.isfile(os.path.join(REPO_ROOT, rel)), f"missing {rel}"

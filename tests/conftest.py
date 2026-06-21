"""Shared fixtures for tau-storm tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure tau core is importable
ROOT = Path(__file__).resolve().parent.parent
TAU_ROOT = ROOT.parent / "tau"
sys.path.insert(0, str(TAU_ROOT))

# Make extensions/storm_research importable as a top-level package
EXTENSIONS_DIR = ROOT / "extensions"
sys.path.insert(0, str(EXTENSIONS_DIR))

"""scripts/run_observers.py uses the G6 cell schema; make_anchors takes --ensemble."""

from __future__ import annotations

import io
import subprocess
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _names(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    return {t.string for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.NAME}


def _help(module: str) -> str:
    out = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_run_observers_cli():
    text = _help("scripts.run_observers")
    for flag in ("--config", "--workers", "--resume"):
        assert flag in text


def test_run_observers_uses_the_cell_schema():
    names = _names(REPO / "scripts" / "run_observers.py")
    for name in ("load_cell_config", "prepare_cell_inputs", "run_dir_name", "observer_model_name",
                 "numerics_summary", "read_frozen_q"):
        assert name in names, "scripts/run_observers.py does not call cell_inputs.%s" % name


def test_make_anchors_takes_an_ensemble_path():
    assert "--ensemble" in _help("scripts.make_anchors")

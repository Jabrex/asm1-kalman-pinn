"""Test helper: write anchor files in the layout of scripts/make_anchors.py (G3).

Tests may read ground truth; this helper uses it only to build anchor files,
before any poisoning, exactly as make_anchors' A0 anchor does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MEASURED = ("S_O", "S_NH", "S_NO")


def write_anchor(
    path: Path,
    data_dir: Path,
    sigma_tag: str = "0p05",
    rel_std: str = "uniform",
    name: str = "A0_test",
) -> Path:
    """A0-style anchor: dry truth at t = 0, nominal steady state from the constant data.

    ``rel_std="uniform"`` writes 0.01 everywhere (make_anchors' A0 rule), so the
    IC weights are all one. ``"graded"`` writes 0.05 on the measured components
    and 0.5 elsewhere, which exercises the weighted IC path.
    """
    from src.asm1.vault_loader import vault
    from src.data.sensors import ObservationDataset

    dry = ObservationDataset.load(Path(data_dir) / ("obs_dry_sigma%s.npz" % sigma_tag))
    const = ObservationDataset.load(Path(data_dir) / ("obs_constant_sigma%s.npz" % sigma_tag))
    z0 = np.array(dry.truth_reactor[0], dtype=float)
    if rel_std == "uniform":
        rel = np.full_like(z0, 0.01)
    elif rel_std == "graded":
        rel = np.full_like(z0, 0.5)
        rel[:, vault().indices(MEASURED)] = 0.05
    else:
        raise ValueError("rel_std must be 'uniform' or 'graded', got %r" % (rel_std,))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z0_mean=z0,
        z0_rel_std=rel,
        nominal_ss=np.array(const.truth_reactor[0], dtype=float),
        settler_init=np.array(dry.truth_y[0], dtype=float),
        meta=json.dumps({"name": name, "source": "tests/anchor_utils.py", "rel_std": rel_std}),
    )
    return path

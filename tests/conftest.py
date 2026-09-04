"""Shared fixtures. Nothing here trains or runs the 100-day warm-up."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.model import Asm1Kinetics  # noqa: E402
from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.asm1.vault_loader import vault  # noqa: E402
from src.data.influent import BSM1_TABLE5_MEAN  # noqa: E402


@pytest.fixture(scope="session")
def v():
    return vault()


@pytest.fixture(scope="session")
def kinetics(v):
    return Asm1Kinetics(v)


@pytest.fixture(scope="session")
def plant():
    return Bsm1Plant()


@pytest.fixture(scope="session")
def sample_state(v):
    """A well-conditioned probe state, built only from sourced numbers.

    Composition is the BSM1 Table 5 influent with an autotroph inoculum. Oxygen
    and nitrate are set to their own vault half-saturation constants, so every
    Monod switch sits at exactly one half and none of the eight rates is
    degenerate. No value here is invented.
    """
    z = np.array([BSM1_TABLE5_MEAN[name] for name in v.components], dtype=float)
    z[v.index("X_B_A")] = BSM1_TABLE5_MEAN["X_B_H"]
    z[v.index("S_O")] = v.p("KO_H")
    z[v.index("S_NO")] = v.p("KNO")
    return z

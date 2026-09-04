"""Print the vault's process rates and stoichiometric rows for the paper appendix.

Nothing here is used by the pipeline; it exists so that the formulas typed
into ``paper/main.tex`` are transcribed from the audited source rather than
from memory.

Usage: python -m scripts.dump_kinetics
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asm1.plant import Bsm1Plant  # noqa: E402
from src.asm1.vault_loader import vault  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    v = vault()
    plant = Bsm1Plant()
    print("components:", v.components)
    print("tss_factor:", plant.cfg.tss_factor, "| so_sat:", plant.cfg.so_sat, "| kla:", plant.cfg.kla)
    print("volumes:", plant.cfg.volumes, "| q_int:", plant.cfg.q_int, "| q_r:", plant.cfg.q_r)
    for i, (name, expr) in enumerate(zip(v.processes, v.rate_expressions), start=1):
        print(f"rho_{i}: {name}\n    {expr}")
    np.set_printoptions(linewidth=220, precision=4, suppress=True)
    print("nu (8 x 14), columns =", v.components)
    print(np.asarray(v.nu))
    print("composition C (14 x 3, COD / N / charge):")
    print(np.asarray(v.composition))
    print("parameters:", dict(v.parameters))


if __name__ == "__main__":
    main()

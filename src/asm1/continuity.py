"""Continuity and mass-balance diagnostics.

Three independent levels of checking, from cheapest to strongest:

1. ``matrix_residual`` - the vault's own check: ``nu @ composition`` must vanish.
   The audited value is 5.5511151231257827e-17 against a 1e-15 tolerance.
2. ``conversion_residual`` - the pointwise consequence: because the matrix
   product vanishes, ``r(Z) @ composition`` must vanish for *every* state Z.
   This catches errors in the rate evaluation path, not just the matrix.
3. ``tracer_components`` / ``system_inventory`` - plant-level closure. The
   components whose stoichiometry column is identically zero are exact passive
   tracers; their total system mass must obey in - out = accumulation with no
   reaction term at all.
"""

from __future__ import annotations

import numpy as np

from .model import Asm1Kinetics
from .plant import Bsm1Plant
from .vault_loader import Asm1Vault, vault


def matrix_residual(source: Asm1Vault | None = None) -> np.ndarray:
    """``nu @ composition``; shape (8 processes, 3 conserved quantities)."""
    v = source if source is not None else vault()
    return v.continuity_residual()


def conversion_residual(kinetics: Asm1Kinetics, Z: np.ndarray) -> np.ndarray:
    """COD / N / Charge production implied by the conversion rates at state ``Z``."""
    return kinetics.continuity_of_conversion(np.asarray(Z, dtype=float))


def tracer_components(source: Asm1Vault | None = None) -> tuple[str, ...]:
    """Components with an all-zero stoichiometry column - exact passive tracers.

    In ASM1 these are ``S_I`` (soluble inert) and ``X_I`` (particulate inert).
    ``S_I`` exercises only the hydraulics; ``X_I`` additionally exercises the
    settler solids split, which makes the pair a strong plant-level check.
    """
    v = source if source is not None else vault()
    zero_cols = np.all(v.nu == 0.0, axis=0)
    return tuple(name for name, is_zero in zip(v.components, zero_cols) if is_zero)


def system_inventory(plant: Bsm1Plant, y: np.ndarray, component: str) -> float:
    """Total mass [g] of one component held in reactors plus clarifier.

    Reactor holdup uses the tank volumes; clarifier holdup uses layer volume
    (area x layer height). Particulate components in the clarifier are
    reconstructed from the layer solids via the BSM1 eq. 46 fraction rule.
    """
    idx = plant.vault.index(component)
    reactor, solids, solubles = plant.unpack(y)
    volumes = np.asarray(plant.cfg.volumes, dtype=float)
    total = float(np.sum(reactor[:, idx] * volumes))

    layer_volume = plant.cfg.settler_area * plant.cfg.settler_layer_height
    if idx in set(plant.i_soluble.tolist()):
        col = int(np.where(plant.i_soluble == idx)[0][0])
        total += float(np.sum(solubles[:, col]) * layer_volume)
    else:
        x_f = float(plant.tss(reactor[-1]))
        share = reactor[-1, idx] / x_f if x_f > 0.0 else 0.0
        total += float(np.sum(share * solids) * layer_volume)
    return total


def total_cod_and_n(plant: Bsm1Plant, y: np.ndarray) -> tuple[float, float]:
    """System-wide COD and N inventory [g], using the vault composition matrix."""
    comp = plant.vault.composition  # (14, 3): COD, N, Charge
    cod = 0.0
    nitrogen = 0.0
    for i, name in enumerate(plant.vault.components):
        mass = system_inventory(plant, y, name)
        cod += mass * comp[i, 0]
        nitrogen += mass * comp[i, 1]
    return cod, nitrogen


def stream_load(z: np.ndarray, q: float, composition_column: np.ndarray) -> float:
    """Conserved-quantity load [g/d] carried by a stream of composition ``z``."""
    return float(q * np.dot(np.asarray(z, dtype=float), composition_column))

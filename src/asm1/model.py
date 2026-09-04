"""ASM1 process rates and conversion rates, evaluated straight from the vault.

The eight rate expressions are NOT re-typed here. They are compiled from the
vault's ``code_expression`` fields, so the process kinetics used by the ODE
ground-truth generator and by the PINN physics loss are literally the same
strings that were audited against ``asm1.xlsx``.

Two vault-specific features of this ASM1 variant are carried through unchanged:

* rho_1 and rho_2 contain an ammonium Monod term ``S_NH/(KNH_H+S_NH)`` with
  ``KNH_H = 0.05``. The vault records this as taken from ASM2d
  (source anomaly ``knh_h_special_value``); it is not in the original ASM1.
* ``S_N2`` is carried as the 14th component (source anomaly ``matrix_only_sn2``).

The vault also flags cells ``X82``/``X84`` as missing alkalinity kinetic terms
that the workbook deliberately leaves uncorrected. No term is invented here
either; ``S_ALK`` uses the vault stoichiometry exactly as published.

The same class serves NumPy (ground-truth ODE) and PyTorch (PINN physics loss).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .vault_loader import Asm1Vault, vault

# Concentrations are clamped to this floor before entering the rate
# expressions. Purpose is purely numerical: rho_7 and rho_8 divide by X_B_H and
# X_S, which can reach zero during transients or during PINN warm-up. The clamp
# is applied to the values fed to the expressions only, never to the state
# itself, and is far below any physically meaningful concentration.
RATE_FLOOR = 1e-12


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


class Asm1Kinetics:
    """Evaluate ASM1 process rates rho(Z) and conversion rates r = nu^T rho."""

    def __init__(self, source: Asm1Vault | None = None, floor: float = RATE_FLOOR) -> None:
        self.vault = source if source is not None else vault()
        self.floor = float(floor)
        self.components = self.vault.components
        self.n_components = len(self.components)
        self.n_processes = len(self.vault.processes)
        self._params = dict(self.vault.parameters)
        self._codes = tuple(
            compile(expr, "<vault:rho_%d>" % (i + 1,), "eval")
            for i, expr in enumerate(self.vault.rate_expressions)
        )
        self._torch_nu: dict[tuple[Any, Any], Any] = {}

    # -- internals ---------------------------------------------------------
    def _clamped(self, Z: Any) -> Any:
        if _is_torch(Z):
            import torch

            return torch.clamp(Z, min=self.floor)
        return np.maximum(Z, self.floor)

    def _namespace(self, Z: Any) -> dict[str, Any]:
        """Map component code ids and parameter code ids onto the last axis of Z."""
        safe = self._clamped(Z)
        ns: dict[str, Any] = dict(self._params)
        for i, name in enumerate(self.components):
            ns[name] = safe[..., i]
        return ns

    def _nu_for(self, like: Any) -> Any:
        if not _is_torch(like):
            return self.vault.nu
        import torch

        key = (like.device, like.dtype)
        cached = self._torch_nu.get(key)
        if cached is None:
            cached = torch.as_tensor(self.vault.nu, device=like.device, dtype=like.dtype)
            self._torch_nu[key] = cached
        return cached

    # -- public API --------------------------------------------------------
    def rates(self, Z: Any) -> Any:
        """Process rates rho for state ``Z`` of shape ``(..., 14)`` -> ``(..., 8)``."""
        ns = self._namespace(Z)
        env = {"__builtins__": {}}
        values = [eval(code, env, ns) for code in self._codes]  # noqa: S307 - vault-sourced
        if _is_torch(Z):
            import torch

            return torch.stack(values, dim=-1)
        return np.stack(np.broadcast_arrays(*values), axis=-1)

    def conversion(self, Z: Any) -> Any:
        """Conversion rates r = nu^T rho for state ``Z``; ``(..., 14)`` -> ``(..., 14)``."""
        rho = self.rates(Z)
        nu = self._nu_for(Z)
        if _is_torch(Z):
            import torch

            return torch.matmul(rho, nu)
        return rho @ nu

    # -- diagnostics -------------------------------------------------------
    def continuity_of_conversion(self, Z: Any) -> Any:
        """COD / N / Charge production implied by ``r``; must be ~0 for any Z.

        This is the pointwise form of the vault continuity check: because
        ``nu @ composition == 0``, ``r @ composition`` must also vanish for
        every state, independently of the rate values.
        """
        r = self.conversion(Z)
        if _is_torch(Z):
            import torch

            comp = torch.as_tensor(self.vault.composition, device=r.device, dtype=r.dtype)
            return torch.matmul(r, comp)
        return r @ self.vault.composition

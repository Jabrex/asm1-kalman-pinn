"""BSM1 plant layout: five-reactor activated sludge train + Takacs settler.

The biology is the vault's ASM1 (see :mod:`src.asm1.model`). Everything in this
module that is NOT biology - tank volumes, flow rates, aeration coefficients,
settler geometry and settling parameters - comes from the BSM1 technical report:

    J. Alex, L. Benedetti, J. Copp, K.V. Gernaey, U. Jeppsson, I. Nopens,
    M.N. Pons, J.P. Steyer, P. Vanrolleghem,
    "Benchmark Simulation Model no. 1 (BSM1)", IWA Task Group on Benchmarking
    of Control Strategies for WWTPs, 2018.

Equation numbers in the comments below refer to that report.

IMPORTANT - what is NOT taken from BSM1: its kinetic and stoichiometric
parameters (a 15 degrees C set). This project uses the vault's 20 degrees C set
everywhere. The plant geometry, flows and influent composition are temperature
independent, which is why they can be combined. A direct consequence is that the
steady state produced here will NOT match BSM1 Table 6 - faster kinetics settle
on a different operating point. That is expected, not an error.

Aeration follows the BSM1 *open-loop* default case (report p.14):
``KLa_3 = KLa_4 = 240 /d`` and ``KLa_5 = 84 /d``, with ``Q_int`` constant. The
closed-loop DO controller is deliberately not used because the report does not
publish its PI gains, and inventing gains would violate the project rule that
no unsourced number may enter the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .model import Asm1Kinetics
from .vault_loader import Asm1Vault, vault

# Component groupings, by vault code id ------------------------------------
SOLUBLE_COMPONENTS = ("S_I", "S_S", "S_O", "S_NO", "S_NH", "S_ND", "S_ALK", "S_N2")
PARTICULATE_COMPONENTS = ("X_I", "X_S", "X_B_H", "X_B_A", "X_P", "X_ND")
# BSM1 eq. 45: TSS = (1/fr_COD_SS) * (X_S + X_P + X_I + X_BH + X_BA). X_ND is a
# nitrogen component and is deliberately excluded from the solids sum.
TSS_COMPONENTS = ("X_S", "X_P", "X_I", "X_B_H", "X_B_A")


@dataclass(frozen=True)
class Bsm1Config:
    """Plant geometry and operating point. Every field cites its BSM1 source."""

    # Bioreactor - report section 2.3.1
    volumes: tuple[float, ...] = (1000.0, 1000.0, 1333.0, 1333.0, 1333.0)  # m3
    kla: tuple[float, ...] = (0.0, 0.0, 240.0, 240.0, 84.0)                # 1/d, open-loop default
    so_sat: float = 8.0                                                    # g/m3, below eq. 26

    # Flows - report section 2.1 and Table 5
    q_int: float = 55338.0   # m3/d internal recycle, tank 5 -> tank 1
    q_r: float = 18446.0     # m3/d sludge recycle (= Q_i,stab)
    q_w: float = 385.0       # m3/d wastage

    # Secondary clarifier - report section 2.3.3 and Table 4
    settler_area: float = 1500.0      # m2
    settler_layer_height: float = 0.4  # m
    n_layers: int = 10
    feed_layer: int = 6               # 1-based, counted from the bottom
    v0_prime: float = 250.0           # m/d maximum settling velocity
    v0: float = 474.0                 # m/d maximum Vesilind settling velocity
    r_h: float = 0.000576             # m3/g hindered zone parameter
    r_p: float = 0.00286              # m3/g flocculant zone parameter
    f_ns: float = 0.00228             # non-settleable fraction
    x_t: float = 3000.0               # g/m3 threshold concentration, eq. 38/40

    # Solids conversion - report eq. 45
    fr_cod_ss: float = 4.0 / 3.0

    # Verification switch. 1.0 = normal operation. Set to 0.0 for the
    # zero-reaction hydraulic test (RUNBOOK step 3c); the plant then reduces to
    # a pure CSTR-in-series plus settler, with an analytic step response.
    reaction_scale: float = 1.0

    @property
    def tss_factor(self) -> float:
        return 1.0 / self.fr_cod_ss  # 0.75

    @property
    def n_tanks(self) -> int:
        return len(self.volumes)


# Influent provider: t (days) -> (Q_in [m3/d], Z_in [14])
InfluentFn = Callable[[float], tuple[float, np.ndarray]]


@dataclass
class PlantIndex:
    """Flat state-vector layout of the coupled reactor + settler system."""

    n_tanks: int
    n_components: int
    n_layers: int
    n_solubles: int

    reactor: slice = field(init=False)
    settler_solids: slice = field(init=False)
    settler_solubles: slice = field(init=False)
    size: int = field(init=False)

    def __post_init__(self) -> None:
        n_reactor = self.n_tanks * self.n_components
        n_solids = self.n_layers
        n_sol = self.n_layers * self.n_solubles
        self.reactor = slice(0, n_reactor)
        self.settler_solids = slice(n_reactor, n_reactor + n_solids)
        self.settler_solubles = slice(n_reactor + n_solids, n_reactor + n_solids + n_sol)
        self.size = n_reactor + n_solids + n_sol


class Bsm1Plant:
    """Right-hand side of the BSM1 plant ODE, with vault ASM1 biology."""

    def __init__(
        self,
        config: Bsm1Config | None = None,
        source: Asm1Vault | None = None,
    ) -> None:
        self.cfg = config if config is not None else Bsm1Config()
        self.vault = source if source is not None else vault()
        self.kinetics = Asm1Kinetics(self.vault)

        self.n_components = len(self.vault.components)
        self.i_soluble = np.array(self.vault.indices(SOLUBLE_COMPONENTS))
        self.i_particulate = np.array(self.vault.indices(PARTICULATE_COMPONENTS))
        self.i_tss = np.array(self.vault.indices(TSS_COMPONENTS))
        self.i_so = self.vault.index("S_O")
        if len(self.i_soluble) + len(self.i_particulate) != self.n_components:
            raise ValueError("Soluble/particulate split does not cover all 14 components")

        self.idx = PlantIndex(
            n_tanks=self.cfg.n_tanks,
            n_components=self.n_components,
            n_layers=self.cfg.n_layers,
            n_solubles=len(self.i_soluble),
        )
        self._volumes = np.asarray(self.cfg.volumes, dtype=float)
        self._kla = np.asarray(self.cfg.kla, dtype=float)

    # -- state packing -----------------------------------------------------
    @property
    def state_size(self) -> int:
        return self.idx.size

    def unpack(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split flat state into (reactor [5,14], settler solids [10], settler solubles [10,8])."""
        reactor = y[self.idx.reactor].reshape(self.cfg.n_tanks, self.n_components)
        solids = y[self.idx.settler_solids]
        solubles = y[self.idx.settler_solubles].reshape(self.cfg.n_layers, len(self.i_soluble))
        return reactor, solids, solubles

    def pack(self, reactor: np.ndarray, solids: np.ndarray, solubles: np.ndarray) -> np.ndarray:
        return np.concatenate([reactor.reshape(-1), solids.reshape(-1), solubles.reshape(-1)])

    # -- derived quantities -----------------------------------------------
    def tss(self, Z: np.ndarray) -> np.ndarray:
        """BSM1 eq. 45: total suspended solids [g SS/m3] from a state vector."""
        return self.cfg.tss_factor * Z[..., self.i_tss].sum(axis=-1)

    def settling_velocity(self, x_sc: np.ndarray, x_min: float) -> np.ndarray:
        """Takacs double-exponential settling velocity, BSM1 eq. 31."""
        d = x_sc - x_min
        v = self.cfg.v0 * (np.exp(-self.cfg.r_h * d) - np.exp(-self.cfg.r_p * d))
        return np.clip(v, 0.0, self.cfg.v0_prime)

    def _settler_streams(
        self, reactor: np.ndarray, solids: np.ndarray, solubles: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Reconstruct full 14-component underflow and effluent compositions.

        BSM1 eq. 46 assumes particulate fractions are preserved across the
        clarifier, so each particulate component in a layer is the layer solids
        concentration scaled by that component's share of the feed solids.
        """
        z5 = reactor[-1]
        x_f = self.tss(z5)
        ratio = np.where(x_f > 0.0, z5[self.i_particulate] / max(x_f, 1e-30), 0.0)

        underflow = np.zeros(self.n_components)
        underflow[self.i_soluble] = solubles[0]
        underflow[self.i_particulate] = ratio * solids[0]

        effluent = np.zeros(self.n_components)
        effluent[self.i_soluble] = solubles[-1]
        effluent[self.i_particulate] = ratio * solids[-1]

        return {"underflow": underflow, "effluent": effluent, "x_f": np.asarray(x_f)}

    # -- right-hand side ---------------------------------------------------
    def rhs(self, t: float, y: np.ndarray, influent: InfluentFn) -> np.ndarray:
        cfg = self.cfg
        reactor, solids, solubles = self.unpack(y)
        q_in, z_in = influent(t)

        streams = self._settler_streams(reactor, solids, solubles)
        z_r = streams["underflow"]
        x_f = float(streams["x_f"])

        # --- reactors, BSM1 eq. 22-26 ------------------------------------
        q1 = q_in + cfg.q_r + cfg.q_int
        r = self.kinetics.conversion(reactor) * cfg.reaction_scale

        d_reactor = np.empty_like(reactor)
        load_in = cfg.q_int * reactor[-1] + cfg.q_r * z_r + q_in * z_in
        d_reactor[0] = (load_in - q1 * reactor[0]) / self._volumes[0] + r[0]
        for k in range(1, cfg.n_tanks):
            d_reactor[k] = (
                q1 * (reactor[k - 1] - reactor[k]) / self._volumes[k] + r[k]
            )
        # oxygen transfer, eq. 26
        d_reactor[:, self.i_so] += self._kla * (cfg.so_sat - reactor[:, self.i_so])

        # --- settler hydraulics ------------------------------------------
        q_f = q_in + cfg.q_r          # = q1 - q_int, feed to the clarifier
        q_u = cfg.q_r + cfg.q_w       # underflow
        q_e = q_f - q_u               # = q_in - q_w, effluent, eq. 30
        v_dn = q_u / cfg.settler_area   # eq. 32
        v_up = q_e / cfg.settler_area   # eq. 33
        z_m = cfg.settler_layer_height
        f = cfg.feed_layer - 1          # 0-based feed layer index
        n = cfg.n_layers

        x_min = cfg.f_ns * x_f
        v_s = self.settling_velocity(solids, x_min)
        j_s = v_s * solids  # gravity flux, eq. below 30

        # clarification flux, eq. 38 / 40 (layers above the feed layer)
        j_sc = np.zeros(n)
        for j in range(f + 1, n):
            direct = j_s[j]
            j_sc[j] = min(direct, j_s[j - 1]) if solids[j - 1] > cfg.x_t else direct

        d_solids = np.zeros(n)
        d_solids[0] = (v_dn * (solids[1] - solids[0]) + min(j_s[1], j_s[0])) / z_m  # eq. 36
        for j in range(1, f):  # eq. 35
            d_solids[j] = (
                v_dn * (solids[j + 1] - solids[j])
                + min(j_s[j], j_s[j + 1])
                - min(j_s[j], j_s[j - 1])
            ) / z_m
        d_solids[f] = (  # eq. 34
            q_f * x_f / cfg.settler_area
            + j_sc[f + 1]
            - (v_up + v_dn) * solids[f]
            - min(j_s[f], j_s[f - 1])
        ) / z_m
        for j in range(f + 1, n - 1):  # eq. 37
            d_solids[j] = (v_up * (solids[j - 1] - solids[j]) + j_sc[j + 1] - j_sc[j]) / z_m
        d_solids[n - 1] = (v_up * (solids[n - 2] - solids[n - 1]) - j_sc[n - 1]) / z_m  # eq. 39

        # solubles in the clarifier, eq. 41-43
        s_f = reactor[-1][self.i_soluble]
        d_solubles = np.zeros_like(solubles)
        for j in range(0, f):  # eq. 42
            d_solubles[j] = v_dn * (solubles[j + 1] - solubles[j]) / z_m
        d_solubles[f] = (q_f * s_f / cfg.settler_area - (v_dn + v_up) * solubles[f]) / z_m
        for j in range(f + 1, n):  # eq. 43
            d_solubles[j] = v_up * (solubles[j - 1] - solubles[j]) / z_m

        return self.pack(d_reactor, d_solids, d_solubles)

    # -- reporting ---------------------------------------------------------
    def outputs(self, t: float, y: np.ndarray, influent: InfluentFn) -> dict[str, np.ndarray]:
        """Derived signals used by sensors, metrics and the effluent quality index."""
        reactor, solids, solubles = self.unpack(y)
        q_in, z_in = influent(t)
        streams = self._settler_streams(reactor, solids, solubles)
        return {
            "t": np.asarray(t),
            "reactor": reactor,
            "settler_solids": solids,
            "settler_solubles": solubles,
            "influent": z_in,
            "q_in": np.asarray(q_in),
            "q_int": np.asarray(self.cfg.q_int),
            "q_r": np.asarray(self.cfg.q_r),
            "q_w": np.asarray(self.cfg.q_w),
            "q_e": np.asarray(q_in - self.cfg.q_w),
            "effluent": streams["effluent"],
            "underflow": streams["underflow"],
            "tss_reactor": self.tss(reactor),
            # settler layer states are already solids concentrations [g SS/m3]
            "tss_underflow": np.asarray(solids[0]),
            "tss_effluent": np.asarray(solids[-1]),
            "kla": self._kla.copy(),
        }

    # -- initial conditions ------------------------------------------------
    def seed_state(self, z_seed: np.ndarray, solids_seed: float) -> np.ndarray:
        """Uniform starting point for the warm-up integration.

        The initial condition is a numerical starting point, not a model
        parameter: the ``WARMUP_DAYS`` constant-input warm-up washes it out. RUNBOOK
        step 3e verifies the steady state is reached, and the IC-independence
        test verifies two different seeds converge to the same steady state.
        """
        reactor = np.tile(np.asarray(z_seed, dtype=float), (self.cfg.n_tanks, 1))
        solids = np.full(self.cfg.n_layers, float(solids_seed))
        solubles = np.tile(reactor[-1][self.i_soluble], (self.cfg.n_layers, 1))
        return self.pack(reactor, solids, solubles)


def constant_influent(q_in: float, z_in: Sequence[float]) -> InfluentFn:
    """Time-invariant influent provider (warm-up and verification runs)."""
    z = np.asarray(z_in, dtype=float)

    def provider(t: float) -> tuple[float, np.ndarray]:
        return float(q_in), z

    return provider

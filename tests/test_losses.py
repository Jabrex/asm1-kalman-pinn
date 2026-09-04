"""Loss terms, the recycle reconstruction, and the physics residual.

The key test here is ``test_physics_residual_vanishes_on_the_true_dynamics``:
if a state and its exact ASM1 derivative produce a non-zero residual, the
physics term is wrong and every downstream result is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.asm1.plant import Bsm1Config, Bsm1Plant  # noqa: E402
from src.data.influent import BSM1_TABLE5_FLOW, stabilisation_influent  # noqa: E402
from src.data.sensors import SENSOR_SET  # noqa: E402
from src.models.losses import Asm1Loss, LossWeights, ObservationOperator  # noqa: E402

TARGETS = tuple(c for c in SENSOR_SET if c.kind != "tss_underflow")


@pytest.fixture(scope="module")
def rig():
    plant = Bsm1Plant()
    operator = ObservationOperator(plant, TARGETS)
    scale = np.ones(plant.n_components)
    loss = Asm1Loss(
        plant=plant,
        operator=operator,
        state_scale=scale,
        target_scale=np.ones(len(TARGETS)),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    return plant, operator, loss


def _batch(plant, n=6):
    q, z = stabilisation_influent()
    rng = np.random.default_rng(0)
    reactor = np.stack([np.tile(z, (plant.cfg.n_tanks, 1)) * (1.0 + 0.1 * rng.normal(
        size=(plant.cfg.n_tanks, plant.n_components))) for _ in range(n)])
    reactor = np.abs(reactor)
    t = np.linspace(0.0, 1.0, n)
    return {
        "t": torch.as_tensor(t, dtype=torch.float64).view(-1, 1),
        "z": torch.as_tensor(reactor, dtype=torch.float64),
        "q_in": torch.full((n, 1), float(q), dtype=torch.float64),
        "z_in": torch.as_tensor(np.tile(z, (n, 1)), dtype=torch.float64),
        "tss_ras": torch.as_tensor(
            plant.tss(reactor[:, -1]) * 3.0, dtype=torch.float64
        ).view(-1, 1),
    }


def test_observation_operator_rejects_settler_channels(rig):
    plant, _, _ = rig
    with pytest.raises(ValueError, match="measured input"):
        ObservationOperator(plant, SENSOR_SET)


def test_observation_operator_matches_the_numpy_sensor_definition(rig):
    plant, operator, _ = rig
    batch = _batch(plant)
    pred = operator(batch["z"]).numpy()
    z = batch["z"].numpy()
    assert pred[:, 0] == pytest.approx(z[:, 2, plant.vault.index("S_O")])
    assert pred[:, -1] == pytest.approx(plant.tss(z[:, 4]))


def test_recycle_reconstruction_uses_only_measurable_quantities(rig):
    """Solubles come from tank 5; particulates are rescaled by the measured RAS TSS."""
    plant, _, loss = rig
    batch = _batch(plant)
    z5 = batch["z"][:, -1, :]
    zr = loss.recycle_composition(z5, batch["tss_ras"])

    sol = plant.i_soluble
    np.testing.assert_allclose(zr[:, sol].numpy(), z5[:, sol].numpy(), rtol=1e-12)

    thickening = (batch["tss_ras"].numpy().ravel() / plant.tss(z5.numpy()))
    part = plant.i_particulate
    np.testing.assert_allclose(
        zr[:, part].numpy(), z5[:, part].numpy() * thickening[:, None], rtol=1e-12
    )


def test_physics_residual_vanishes_on_the_true_dynamics(rig):
    """Feed the torch RHS its own output as dZ/dt; the residual must be zero.

    This also cross-checks the torch plant model against the NumPy one used to
    generate the ground truth: both must agree to machine precision.
    """
    plant, _, loss = rig
    batch = _batch(plant)
    rhs = loss.plant_rhs(batch["z"], batch["q_in"], batch["z_in"], batch["tss_ras"])
    residual = loss.physics_residual(
        batch["z"], rhs, batch["q_in"], batch["z_in"], batch["tss_ras"]
    )
    assert float(torch.max(torch.abs(residual))) < 1e-12


def test_torch_rhs_matches_the_numpy_reactor_equations(rig):
    plant, _, loss = rig
    batch = _batch(plant, n=1)
    z = batch["z"][0].numpy()
    tss_ras = float(batch["tss_ras"][0])

    zr = np.zeros(plant.n_components)
    z5 = z[-1]
    zr[plant.i_soluble] = z5[plant.i_soluble]
    zr[plant.i_particulate] = z5[plant.i_particulate] * (tss_ras / plant.tss(z5))

    q_in = BSM1_TABLE5_FLOW
    q1 = q_in + plant.cfg.q_r + plant.cfg.q_int
    volumes = np.asarray(plant.cfg.volumes)
    _, z_in = stabilisation_influent()

    expected = np.empty_like(z)
    expected[0] = (
        plant.cfg.q_int * z5 + plant.cfg.q_r * zr + q_in * z_in - q1 * z[0]
    ) / volumes[0]
    for k in range(1, plant.cfg.n_tanks):
        expected[k] = q1 * (z[k - 1] - z[k]) / volumes[k]
    expected += plant.kinetics.conversion(z)
    expected[:, plant.i_so] += np.asarray(plant.cfg.kla) * (plant.cfg.so_sat - z[:, plant.i_so])

    actual = loss.plant_rhs(batch["z"], batch["q_in"], batch["z_in"], batch["tss_ras"])
    np.testing.assert_allclose(actual[0].numpy(), expected, rtol=1e-10, atol=1e-10)


def _loss_for(plant):
    return Asm1Loss(
        plant=plant,
        operator=ObservationOperator(plant, TARGETS),
        state_scale=np.ones(plant.n_components),
        target_scale=np.ones(len(TARGETS)),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0])
def test_plant_rhs_honours_reaction_scale(scale):
    """The torch RHS must mirror ``Bsm1Plant.rhs``, which scales reactions.

    ``Bsm1Plant.rhs`` multiplies the conversion rates by ``cfg.reaction_scale``
    (the zero-reaction verification switch). If the torch path ignored that
    factor, the physics term would supervise dynamics the data was never
    generated with - silently, and only in configurations that turn reactions
    off. The pair of RHS evaluations below isolates the reaction contribution,
    which the transport and aeration terms leave untouched.
    """
    inert = Bsm1Plant(Bsm1Config(reaction_scale=0.0))
    scaled = Bsm1Plant(Bsm1Config(reaction_scale=scale))
    batch = _batch(inert)
    args = (batch["z"], batch["q_in"], batch["z_in"], batch["tss_ras"])

    contribution = _loss_for(scaled).plant_rhs(*args) - _loss_for(inert).plant_rhs(*args)
    expected = inert.kinetics.conversion(batch["z"]) * scale
    np.testing.assert_allclose(
        contribution.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12
    )


def test_zero_reaction_plant_has_no_biology_in_the_physics_term():
    """With reactions off the torch RHS must be pure transport plus aeration."""
    plant = Bsm1Plant(Bsm1Config(reaction_scale=0.0))
    batch = _batch(plant)
    rhs = _loss_for(plant).plant_rhs(
        batch["z"], batch["q_in"], batch["z_in"], batch["tss_ras"]
    )
    reactive = _loss_for(Bsm1Plant()).plant_rhs(
        batch["z"], batch["q_in"], batch["z_in"], batch["tss_ras"]
    )
    assert not torch.allclose(rhs, reactive), "reaction_scale=0 changed nothing"


def test_positivity_term_is_structurally_zero_for_valid_states(rig):
    plant, _, loss = rig
    batch = _batch(plant)
    assert float(loss.positivity_loss(batch["z"])) == 0.0


def test_positivity_term_fires_on_negative_states(rig):
    plant, _, loss = rig
    batch = _batch(plant)
    assert float(loss.positivity_loss(-batch["z"])) > 0.0


def test_balance_term_vanishes_on_a_consistent_trajectory(rig):
    """A trajectory that is flat and at steady inflow must close its own balance."""
    plant, _, loss = rig
    batch = _batch(plant, n=8)
    flat = batch["z"][:1].expand_as(batch["z"]).contiguous()
    value = float(loss.balance_loss(batch["t"], flat, batch["q_in"], batch["z_in"], batch["tss_ras"]))
    assert np.isfinite(value)


def test_masked_ic_loss_ignores_unmeasured_entries(rig):
    """With an IC mask, ground truth outside the sensed entries must be inert.

    NaN-poisoning the masked-out entries of ``z0_true`` proves they never enter
    the forward value or the backward gradient - the same discipline
    ``tests/test_leakage.py`` applies to the post-t0 trajectory.
    """
    plant, operator, _ = rig
    mask = np.zeros((plant.cfg.n_tanks, plant.n_components))
    mask[4, plant.vault.index("S_O")] = 1.0
    mask[4, plant.vault.index("S_NH")] = 1.0
    mask[1, plant.vault.index("S_NO")] = 1.0
    masked = Asm1Loss(
        plant=plant,
        operator=operator,
        state_scale=np.ones(plant.n_components),
        target_scale=np.ones(len(TARGETS)),
        device=torch.device("cpu"),
        dtype=torch.float64,
        ic_mask=mask,
    )

    z0_pred = torch.rand(1, plant.cfg.n_tanks, plant.n_components, dtype=torch.float64)
    z0_pred.requires_grad_(True)
    z0_true = torch.rand(1, plant.cfg.n_tanks, plant.n_components, dtype=torch.float64)
    z0_true[:, mask == 0.0] = float("nan")

    value = masked.ic_loss(z0_pred, z0_true)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(z0_pred.grad).all()

    # The masked loss must equal the plain mean over just the sensed entries.
    sensed = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    expected = torch.mean((z0_pred.detach()[sensed] - z0_true[sensed]) ** 2)
    assert float(value) == pytest.approx(float(expected), rel=1e-12)


def test_total_reports_every_term(rig):
    plant, _, loss = rig
    batch = _batch(plant)
    targets = torch.zeros(batch["z"].shape[0], len(TARGETS), dtype=torch.float64)
    parts = loss.total(
        weights=LossWeights(),
        t=batch["t"], z=batch["z"], dz_dt=None,
        q_in=batch["q_in"], z_in=batch["z_in"], tss_ras=batch["tss_ras"],
        targets=targets, z0_pred=batch["z"][:1], z0_true=batch["z"][:1],
    )
    values = parts.detached()
    assert set(values) == {"total", "data", "physics", "ic", "positivity", "balance"}
    assert all(np.isfinite(v) for v in values.values())
    assert values["ic"] == 0.0

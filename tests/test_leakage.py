"""Ground truth must not reach the optimiser beyond the supplied initial state.

Two layers, deliberately not redundant.

**Static** - always runs, needs no data. Tokenises the source (so comments and
docstrings cannot hide or fake a match) and requires that every reference to
``truth_reactor`` / ``truth_y`` in the training path is exactly ``[0]``: the
initial condition, which is a boundary condition supplied to all four models
alike. It also classifies *every* module under ``src/``, so a new file that
touches ground truth fails until someone deliberately files it under the
training rules or the evaluation allowlist. Fail-closed by construction.

**Runtime** - needs the generated datasets. Replaces every ground-truth sample
after ``t = 0`` with NaN and requires that the loss and its gradients stay
finite, across every curriculum stage, on both datasets, for both architectures,
including the physics/collocation path. A NaN reaching the optimiser through any
indirect route the tokeniser cannot see shows up here.

Neither layer subsumes the other. The static layer catches a bad line the
runtime layer never executes; the runtime layer catches leakage through a
renamed field, an aliased array or a helper the static scan does not know to
read.

Set ``ASM1_STRICT_TESTS=1`` to turn the runtime layer's "datasets missing" skip
into a failure - use it when you want the green run to mean full coverage.
"""

from __future__ import annotations

import io
import os
import tokenize
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

TRAINING_ROOTS = ("src/train", "src/models", "src/observers")
OBSERVER_ROOT = "src/observers"

EVALUATION_ALLOWLIST = frozenset({"src/data/sensors.py", "src/eval/report.py"})

TRUTH_NAMES = frozenset({"truth_reactor", "truth_y"})
CLEAN_OBSERVATION_NAMES = frozenset({"obs_clean"})

STRICT = os.environ.get("ASM1_STRICT_TESTS", "").strip().lower() not in {"", "0", "false", "no"}


def _code_tokens(path: Path) -> list[tokenize.TokenInfo]:
    """Real code tokens: comments, docstrings and string literals removed.

    A line-based parser has to guess where docstrings start and end, and a single
    mis-guess makes it skip real code - which fails open on a test whose whole
    job is to be fail-closed. The tokeniser does not guess.
    """
    source = path.read_text(encoding="utf-8")
    keep = {tokenize.NAME, tokenize.OP, tokenize.NUMBER}
    return [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type in keep
    ]


def _name_hits(path: Path, names: frozenset[str]) -> list[tuple[int, str, bool]]:
    """Every occurrence of ``names``, with whether it is an ``[0]`` access.

    Returns ``(line number, name, is_initial_condition_access)``. The pattern
    checked is the exact four-token sequence ``NAME [ 0 ]``, so neither a slice
    (``[0:5]``) nor a second statement on the same line can pass by accident.
    """
    tokens = _code_tokens(path)
    hits: list[tuple[int, str, bool]] = []
    for i, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string not in names:
            continue
        tail = tokens[i + 1 : i + 4]
        initial = (
            len(tail) == 3
            and tail[0].type == tokenize.OP and tail[0].string == "["
            and tail[1].type == tokenize.NUMBER and tail[1].string == "0"
            and tail[2].type == tokenize.OP and tail[2].string == "]"
        )
        hits.append((token.start[0], token.string, initial))
    return hits


def _modules(root: str) -> list[Path]:
    return sorted(p for p in (REPO / root).rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def test_tokeniser_sees_through_comments_and_docstrings():
    """Guard the guard: the static layer is only as good as its tokeniser."""
    probe = REPO / "src" / "train" / "run.py"
    tokens = _code_tokens(probe)
    strings = {t.string for t in tokens}
    assert "truth_reactor" in strings, "tokeniser lost a real code reference"
    docstring_mentions = probe.read_text(encoding="utf-8").count("truth_reactor")
    code_mentions = sum(1 for t in tokens if t.string == "truth_reactor")
    assert code_mentions < docstring_mentions, (
        "tokeniser is counting prose as code - the static layer would be scanning "
        "documentation instead of the training path"
    )


def test_training_path_reads_truth_only_at_time_zero():
    offenders: list[str] = []
    scanned = 0
    for root in TRAINING_ROOTS:
        for path in _modules(root):
            scanned += 1
            for line, name, initial in _name_hits(path, TRUTH_NAMES):
                if not initial:
                    offenders.append("%s:%d  %s (not an [0] access)" % (_rel(path), line, name))
    assert scanned, "TRAINING_ROOTS matched no modules - the scan is vacuous"
    assert not offenders, (
        "Ground truth is read outside the supplied initial condition in the "
        "training path:\n  " + "\n  ".join(offenders)
    )


def test_training_path_never_reads_the_noise_free_observations():
    """Models train on ``obs``. Touching ``obs_clean`` would fake the noise sweep."""
    offenders: list[str] = []
    for root in TRAINING_ROOTS:
        for path in _modules(root):
            for line, name, _ in _name_hits(path, CLEAN_OBSERVATION_NAMES):
                offenders.append("%s:%d  %s" % (_rel(path), line, name))
    assert not offenders, (
        "Training path reads the noise-free observations:\n  " + "\n  ".join(offenders)
    )


def test_every_module_touching_truth_is_classified():
    """Fail-closed: a new file anywhere under src/ cannot quietly read truth."""
    training = {_rel(p) for root in TRAINING_ROOTS for p in _modules(root)}
    unclassified: list[str] = []
    for path in _modules("src"):
        rel = _rel(path)
        if not _name_hits(path, TRUTH_NAMES):
            continue
        if rel in training or rel in EVALUATION_ALLOWLIST:
            continue
        unclassified.append(rel)
    assert not unclassified, (
        "These modules read ground truth but belong to neither the training path "
        "(where only [0] is allowed) nor EVALUATION_ALLOWLIST. Classify them "
        "deliberately:\n  " + "\n  ".join(unclassified)
    )


def test_observers_never_name_ground_truth_at_all():
    """Anchors carry the initial state; the observer package reads no truth, not even [0]."""
    modules = _modules(OBSERVER_ROOT)
    assert any(p.name == "anchors.py" for p in modules), "src/observers/anchors.py is missing"
    offenders = [
        "%s:%d  %s" % (_rel(path), line, name)
        for path in modules
        for line, name, _ in _name_hits(path, TRUTH_NAMES | CLEAN_OBSERVATION_NAMES)
    ]
    assert not offenders, "Observer code names ground truth:\n  " + "\n  ".join(offenders)


def test_evaluation_allowlist_has_no_dead_entries():
    """A stale allowlist entry would silently widen what is permitted."""
    stale = [
        rel for rel in sorted(EVALUATION_ALLOWLIST)
        if not (REPO / rel).exists() or not _name_hits(REPO / rel, TRUTH_NAMES)
    ]
    assert not stale, "EVALUATION_ALLOWLIST entries no longer read truth: %s" % (stale,)


def test_evaluation_helpers_are_kept_out_of_the_graph():
    """``Trainer.predict`` is the only place predictions are made for scoring."""
    source = (REPO / "src/train/run.py").read_text(encoding="utf-8")
    index = source.index("def predict")
    preceding = source[:index].rstrip().splitlines()[-1].strip()
    assert preceding == "@torch.no_grad()", (
        "Trainer.predict must be decorated with @torch.no_grad(); found %r" % preceding
    )


SIGMA_TAG = "0p05"


def _require_datasets() -> Path:
    data_dir = REPO / "results" / "raw"
    needed = [
        data_dir / ("obs_%s_sigma%s.npz" % (scenario, SIGMA_TAG))
        for scenario in ("dry", "constant")
    ]
    missing = [p.name for p in needed if not p.exists()]
    if not missing:
        return data_dir
    message = (
        "runtime leakage check needs %s - run 'python -m scripts.generate_data' "
        "first (RUNBOOK step 4, then re-run the tests at step 5)" % (missing,)
    )
    if STRICT:
        pytest.fail("ASM1_STRICT_TESTS is set: " + message)
    pytest.skip(message)
    raise AssertionError("unreachable")


def _poison(dataset) -> None:
    """Replace every ground-truth sample after t = 0 with NaN, in place."""
    dataset.truth_reactor = dataset.truth_reactor.copy()
    dataset.truth_reactor[1:] = np.nan
    dataset.truth_y = dataset.truth_y.copy()
    dataset.truth_y[1:] = np.nan


@pytest.mark.parametrize("model", ["cl_pinn", "cl_lstm"])
def test_loss_and_gradients_stay_finite_when_hidden_truth_is_poisoned(model):
    """Every stage, both datasets, both architectures, physics path included.

    The curriculum's first stage trains on the ``constant`` dataset, so poisoning
    only ``dry`` would leave that path unchecked. Backward is run as well: a NaN
    can appear in the gradient even when the forward loss is finite.
    """
    torch = pytest.importorskip("torch")
    data_dir = _require_datasets()

    from src.train.run import RunConfig, Trainer

    trainer = Trainer(
        RunConfig(
            run_id="_leakage_probe_%s" % model, model=model, noise=0.05,
            profile="quick", steps_quick=1, device="cpu", dtype="float64",
            data_dir=str(data_dir),
        )
    )
    for key in ("dry", "constant"):
        _poison(trainer.data[key])
    trainer.train_set = trainer.data["dry"].window(0.0, trainer.cfg.train_end_day)
    trainer._tensor_cache.clear()

    assert trainer.schedule.stages, "schedule is empty - the check would be vacuous"

    for stage in trainer.schedule.stages:
        batch = trainer._stage_tensors(stage)
        assert np.isfinite(batch["targets"].detach().cpu().numpy()).all(), (
            "stage %s: the measured targets themselves became NaN, so this stage "
            "proves nothing" % stage.name
        )

        weights = trainer._weights_for(stage.weights_end)
        z = trainer.model(batch["t"], batch["q_in"], batch["z_in"])
        parts = trainer.loss.total(
            weights=weights,
            t=batch["t"], z=z, dz_dt=None,
            q_in=batch["q_in"], z_in=batch["z_in"], tss_ras=batch["tss_ras"],
            targets=batch["targets"], z0_pred=z[:1], z0_true=batch["z0_true"],
        )
        total = parts.total
        for name, value in parts.detached().items():
            assert np.isfinite(value), (
                "stage %s: loss term %s became %s with poisoned ground truth"
                % (stage.name, name, value)
            )

        if trainer.cfg.arch == "pinn":
            colloc = trainer._collocation(stage, batch)
            z_c, dz_c = trainer.model.state_and_derivative(
                colloc["t"], colloc["q_in"], colloc["z_in"]
            )
            residual = trainer.loss.physics_residual(
                z_c, dz_c, colloc["q_in"], colloc["z_in"], colloc["tss_ras"]
            )
            physics = torch.mean(residual ** 2)
            assert torch.isfinite(physics), (
                "stage %s: the physics residual became NaN with poisoned ground "
                "truth - the collocation path is reading a hidden state" % stage.name
            )
            total = total + weights.physics * physics

        trainer.model.zero_grad(set_to_none=True)
        total.backward()
        bad = [
            name for name, p in trainer.model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert not bad, (
            "stage %s: gradients became non-finite with poisoned ground truth: %s"
            % (stage.name, bad[:5])
        )


@pytest.mark.parametrize("poison_t0", [False, True], ids=["after_t0", "including_t0"])
def test_observers_stay_finite_when_hidden_truth_is_poisoned(poison_t0):
    """EKF and EKS on poisoned datasets: outputs must stay finite.

    ``after_t0`` keeps truth[0] (the A0 anchor is copied from it before the
    poisoning); ``including_t0`` poisons every sample, as when the anchor comes
    from a file. A short window keeps the test fast.
    """
    pytest.importorskip("torch")
    data_dir = _require_datasets()
    from src.data.sensors import ObservationDataset
    from src.observers.pipeline import ObserverSpec, run_estimators

    dry = ObservationDataset.load(data_dir / ("obs_dry_sigma%s.npz" % SIGMA_TAG)).window(0.0, 1.0)
    anchor = {"z0_mean": dry.truth_reactor[0].copy(), "z0_rel_std": np.full((5, 14), 0.05)}
    _poison(dry)
    if poison_t0:
        dry.truth_reactor[0] = np.nan
        dry.truth_y[0] = np.nan
    spec = ObserverSpec(
        estimators=("ekf", "eks"), q_mode="fixed", q_fixed=(0.03, 0.01),
        train_end_day=0.5, holdout_days=(0.5, 1.0), rain=False,
    )
    out = run_estimators(dry, None, anchor, spec)
    for name in ("ekf", "eks"):
        assert not out[name]["info"]["diverged"]
        for key, arr in out[name]["predictions"].items():
            assert np.isfinite(arr).all(), "%s/%s became non-finite" % (name, key)


THETA_PROBE = ["muA", "bA", "muH", "bH"]


def _poison_everything(dataset) -> None:
    """Replace every ground-truth sample, including t = 0, with NaN, in place."""
    dataset.truth_reactor = np.full_like(dataset.truth_reactor, np.nan)
    dataset.truth_y = np.full_like(dataset.truth_y, np.nan)


@pytest.mark.parametrize("influent_mode", ["exact", "composite"])
@pytest.mark.parametrize("model", ["cl_pinn", "cl_pinn_theta", "lstm"])
def test_anchor_runs_never_read_truth_even_at_time_zero(model, influent_mode, tmp_path, monkeypatch):
    """With anchor_file set the Trainer must not need any truth sample at all.

    The datasets are poisoned as they are loaded, before the Trainer exists, so
    ``__init__`` (Z(0), output scale, IC weights), every curriculum stage,
    the collocation path, the kinetic prior, one real optimisation step and
    ``finalise`` (including the rain prediction) all run on poisoned data.
    """
    torch = pytest.importorskip("torch")
    data_dir = _require_datasets()

    from anchor_utils import write_anchor
    from src.data.sensors import ObservationDataset
    from src.train.run import RunConfig, Trainer

    anchor = write_anchor(tmp_path / "anchor.npz", data_dir, sigma_tag=SIGMA_TAG, rel_std="graded")
    original = ObservationDataset.load.__func__
    loaded: list[str] = []

    def poisoned_load(cls, path):
        dataset = original(cls, path)
        _poison_everything(dataset)
        loaded.append(Path(path).name)
        return dataset

    monkeypatch.setattr(ObservationDataset, "load", classmethod(poisoned_load))

    trainer = Trainer(
        RunConfig(
            run_id="_leakage_anchor_%s_%s" % (model, influent_mode), model=model, noise=0.05,
            profile="quick", steps_quick=1, log_every=1, device="cpu", dtype="float64",
            data_dir=str(data_dir), out_dir=str(tmp_path / "runs"),
            anchor_file=str(anchor), influent_mode=influent_mode,
            total_derivative=True, ras_filter_window=4,
            trainable_kinetics=THETA_PROBE if model == "cl_pinn_theta" else [],
        )
    )
    assert {"obs_dry_sigma%s.npz" % SIGMA_TAG, "obs_constant_sigma%s.npz" % SIGMA_TAG} <= set(loaded)
    assert np.isnan(trainer.data["dry"].truth_reactor).all(), "poisoning did not take"
    assert np.isfinite(trainer.z0).all() and np.isfinite(trainer.state_scale).all()
    assert trainer.schedule.stages, "schedule is empty - the check would be vacuous"

    params = list(trainer.model.parameters())
    if trainer.adapter is not None:
        params += list(trainer.adapter.parameters())
    for stage in trainer.schedule.stages:
        batch = trainer._stage_tensors(stage)
        assert torch.isfinite(batch["z0_true"]).all(), stage.name
        assert torch.isfinite(batch["targets"]).all(), stage.name
        parts = trainer.step_loss(stage, trainer._weights_for(stage.weights_end))
        for name, value in parts.detached().items():
            assert np.isfinite(value), "stage %s: %s = %s" % (stage.name, name, value)
        for p in params:
            p.grad = None
        parts.total.backward()
        bad = [i for i, p in enumerate(params) if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, "stage %s: non-finite gradients in parameters %s" % (stage.name, bad[:5])

    summary = trainer.train()
    assert all(np.isfinite(v) for k, v in trainer.history[-1].items()
               if isinstance(v, float)), trainer.history[-1]
    with np.load(tmp_path / "runs" / trainer.cfg.run_id / "predictions.npz") as preds:
        for key in preds.files:
            assert np.isfinite(preds[key]).all(), key
    assert summary["anchor_file"] == str(anchor)

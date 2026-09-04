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

#: Directories whose modules run inside the training loop. Globbed, not listed,
#: so a newly added module is covered automatically instead of silently skipped.
TRAINING_ROOTS = ("src/train", "src/models")

#: Modules that may read ground truth freely: they build the datasets or score
#: predictions after training. Kept deliberately minimal - a file that needs to
#: join this list should be a conscious decision, not a default.
EVALUATION_ALLOWLIST = frozenset({"src/data/sensors.py", "src/eval/report.py"})

TRUTH_NAMES = frozenset({"truth_reactor", "truth_y"})
#: The noise-free observations. Training must use the noisy ``obs`` instead, or
#: the reported noise robustness would be a fiction.
CLEAN_OBSERVATION_NAMES = frozenset({"obs_clean"})

STRICT = os.environ.get("ASM1_STRICT_TESTS", "").strip().lower() not in {"", "0", "false", "no"}


# --------------------------------------------------------------------------
# static layer
# --------------------------------------------------------------------------
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
    # The module docstring names truth_reactor too; it must not become a token.
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


# --------------------------------------------------------------------------
# runtime layer
# --------------------------------------------------------------------------
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
    # z0 and the output scale were captured in __init__ from truth[0], which is
    # the supplied boundary condition. Everything after t = 0 is poisoned now.
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

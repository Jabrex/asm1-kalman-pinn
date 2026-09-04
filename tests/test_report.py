"""Report collection: verification probes must not leak into the benchmark."""

from __future__ import annotations

from src.eval.report import collect_runs


def test_underscore_run_dirs_are_ignored(tmp_path):
    """verify_model writes _-prefixed probe runs; the report must skip them.

    The probe dir gets deliberately invalid JSON: if collect_runs ever opens
    it, the test fails with a decode error instead of silently passing.
    """
    runs = tmp_path / "runs"
    probe = runs / "_verify_probe"
    probe.mkdir(parents=True)
    (probe / "summary.json").write_text("{not valid json", encoding="utf-8")
    (probe / "predictions.npz").write_bytes(b"")
    assert collect_runs(runs, tmp_path / "raw") == []

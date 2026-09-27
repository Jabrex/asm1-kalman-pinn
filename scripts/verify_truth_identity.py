"""Identity checks for the v1.1 truth-plant generator.

Fast check (seconds)
    For every mismatch directory given (default results/raw_k025, raw_k050,
    raw_k075, raw_k100 when present), the eight standard columns of
    ``obs_constant_sigma*.npz`` must be bit-identical to ``results/raw``,
    because with ``--constant-from nominal`` the constant scenario comes from
    the nominal plant with the v1.0 noise seeds.

Slow check (about 1 min, skipped with --fast-only)
    ``generate_data --truth-preset vault20 --constant-from truth`` into a
    temporary directory must reproduce ``results/raw`` sim_constant and
    sim_dry arrays, and every obs_dry / obs_constant ``obs`` and ``obs_clean``
    array, bit for bit. This proves the override path is a no-op for the vault
    set and that the v1.0 seeds are untouched. Run it with the same
    ``OPENBLAS_NUM_THREADS`` as the generation runs: on the development machine
    an unset value, 4 and 24 threads reproduce v1.0 bit for bit, while 1 thread
    takes a different LAPACK path and differs by about 5e-12 relative.

Checksums (--checksums PATH)
    Writes the SHA-256 of every ``*.npz`` under the given data directories, so a
    regenerated dataset can be compared file by file.

Exit code 0 = all checks pass.

Usage: python -m scripts.verify_truth_identity [--fast-only] [--dirs DIR ...] [--checksums PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import generate_data
from src.data.sensors import NOISE_LEVELS, ObservationDataset
from src.data.simulate import SimulationResult

RAW = REPO_ROOT / "results" / "raw"
MISMATCH_DIRS = ("raw_k025", "raw_k050", "raw_k075", "raw_k100")
CHECKSUM_DIRS = ("raw", "raw_k025", "raw_k050", "raw_k075", "raw_k100", "raw_k000_off", "raw_rand")
N_STANDARD = 8


def check_constant_columns(directory: Path) -> list[str]:
    failures = []
    for sigma in NOISE_LEVELS:
        name = "obs_constant_sigma%s.npz" % generate_data.sigma_tag(sigma)
        ref = ObservationDataset.load(RAW / name)
        new = ObservationDataset.load(directory / name)
        if new.channels[:N_STANDARD] != ref.channels:
            failures.append("%s/%s: channel names differ" % (directory.name, name))
        for field in ("obs", "obs_clean"):
            if not np.array_equal(getattr(new, field)[:, :N_STANDARD], getattr(ref, field)):
                failures.append("%s/%s: %s standard columns differ" % (directory.name, name, field))
        if new.meta.get("constant_from") != "nominal":
            failures.append("%s/%s: meta constant_from is %r"
                            % (directory.name, name, new.meta.get("constant_from")))
    return failures


def check_vault20_regeneration() -> list[str]:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        code = generate_data.main([
            "--out", tmp, "--truth-preset", "vault20", "--constant-from", "truth",
            "--scenarios", "constant", "dry",
        ])
        if code != 0:
            return ["generate_data returned %d" % code]
        for scenario in ("constant", "dry"):
            ref = SimulationResult.load(RAW / ("sim_%s.npz" % scenario))
            new = SimulationResult.load(Path(tmp) / ("sim_%s.npz" % scenario))
            for field in ("t", "y", "reactor", "influent", "q_in"):
                if not np.array_equal(getattr(new, field), getattr(ref, field)):
                    failures.append("sim_%s.%s differs" % (scenario, field))
            for sigma in NOISE_LEVELS:
                name = "obs_%s_sigma%s.npz" % (scenario, generate_data.sigma_tag(sigma))
                a = ObservationDataset.load(RAW / name)
                b = ObservationDataset.load(Path(tmp) / name)
                for field in ("obs", "obs_clean"):
                    if not np.array_equal(getattr(a, field), getattr(b, field)):
                        failures.append("%s.%s differs" % (name, field))
                if a.seed != b.seed:
                    failures.append("%s seed %d != %d" % (name, b.seed, a.seed))
    return failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(out: Path, names: tuple[str, ...] = CHECKSUM_DIRS) -> int:
    base = REPO_ROOT / "results"
    files = {
        path.relative_to(base).as_posix(): _sha256(path)
        for name in names if (base / name).exists()
        for path in sorted((base / name).rglob("*.npz"))
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                               "files": files}, indent=2), encoding="utf-8")
    print("Wrote %s (%d files)" % (out, len(files)))
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--dirs", nargs="*", default=None)
    parser.add_argument("--checksums", default=None, help="write SHA-256 of every dataset file here")
    args = parser.parse_args(argv)

    dirs = [Path(d) for d in args.dirs] if args.dirs is not None else [
        REPO_ROOT / "results" / d for d in MISMATCH_DIRS if (REPO_ROOT / "results" / d).exists()
    ]
    failures: list[str] = []
    for d in dirs:
        found = check_constant_columns(d)
        print("constant-scenario standard columns  %-10s %s" % (d.name, "PASS" if not found else "FAIL"))
        failures += found
    if not args.fast_only:
        found = check_vault20_regeneration()
        print("vault20 regeneration reproduces results/raw  %s" % ("PASS" if not found else "FAIL"))
        failures += found
    for line in failures:
        print("  " + line)
    if args.checksums and not failures:
        write_checksums(Path(args.checksums))
    print("PASS" if not failures else "FAIL (%d)" % len(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

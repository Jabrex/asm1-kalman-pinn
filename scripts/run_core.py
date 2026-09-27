"""Run the v1.1 smoke and core GPU queues under a budget guard (group G6).

    python -m scripts.run_core --phase smoke-pair --workers 2
    python -m scripts.run_core --phase determinism
    python -m scripts.run_core --phase smoke
    python -m scripts.run_core --phase E4 E5 E6 E7 E3 --list
    python -m scripts.run_core --phase E4 E5 E6 E7 E3 --workers 1

Each job is one scripts.run_all subprocess (one config, one seed, a model and
noise subset) with --resume, so finished runs are never retrained. A job starts
only if the eq already charged in results/v11/gpu_ledger.csv, plus the jobs in
flight, plus this job stays within --cap (default 45.05; never above 50). Every
attempted run is written to the ledger when its job ends; runs of a job that a
killed process left behind are charged on the next start (status
'interrupted' if unfinished).

Full-profile phases refuse to start unless the v1.1.0-prereg tag exists and the
frozen paths match it. --allow-drift REASON proceeds past a mismatch and appends
the reason to results/v11/deviations.log; a missing tag is never bypassed.
Failed runs are not rerun, except infrastructure crashes with --retry-infra (the
crashed directory is kept as <run_id>__crash<N>). Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import gpu_ledger
from scripts.v11_plan import (
    CONCURRENCY, CORE_CAP, CORE_PHASES, HARD_CAP, INFLIGHT, LEDGER, REPO, Job, core_queue,
    determinism_job, log_deviation, nominal_eq, planned_runs, realistic_k, registration_state,
    sigma_of, sigma_tag, smoke_queue,
)

POLL_SECONDS = 10.0
PHASES = ("smoke-pair", "determinism", "smoke", *CORE_PHASES)


@dataclass
class Pending:
    job: Job
    todo: list[tuple[str, str]]
    eq: float


def budget_allows(charged: float, inflight: float, next_eq: float, cap: float) -> bool:
    return charged + inflight + next_eq <= cap + 1e-9


def select(phases: list[str], realistic: str) -> list[Job]:
    jobs: list[Job] = []
    for phase in phases:
        if phase == "smoke":
            jobs.extend(smoke_queue(realistic))
        elif phase == "smoke-pair":
            jobs.extend(smoke_queue(realistic)[:2])
        elif phase == "determinism":
            jobs.append(determinism_job(realistic))
        elif phase in CORE_PHASES:
            jobs.extend(j for j in core_queue(realistic) if j.phase == phase)
        else:
            raise ValueError("unknown phase %r; expected one of %s" % (phase, PHASES))
    unique, seen = [], set()
    for job in jobs:
        key = (job.config, job.seed, job.out_dir, job.models, job.noise)
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def _next_crash_index(run_dir: Path) -> int:
    return 1 + sum(1 for _ in run_dir.parent.glob("%s__crash*" % run_dir.name))


def plan(jobs: list[Job], retry_infra: bool) -> tuple[list[Pending], list[str]]:
    pending: list[Pending] = []
    blocked_all: list[str] = []
    for job in jobs:
        listed = planned_runs(job)
        expected = job.expected_run_ids()
        if sorted(listed) != sorted(expected):
            raise RuntimeError("run_all lists %s for %s seed %d; the plan expects %s"
                               % (listed, job.config, job.seed, expected))
        todo: list[tuple[str, str]] = []
        blocked: list[str] = []
        for run_id, model in listed:
            run_dir = REPO / job.out_dir / run_id
            status = gpu_ledger.run_status(run_dir)
            if status in ("ok", "failed_numeric"):
                continue
            if status == "error_infra" and retry_infra:
                run_dir.rename(run_dir.with_name("%s__crash%d" % (run_id, _next_crash_index(run_dir))))
                todo.append((run_id, model))
            elif status in ("error_infra", "error_other"):
                blocked.append(Path(job.out_dir, run_id).as_posix())
            else:
                todo.append((run_id, model))
        if todo and blocked:
            raise RuntimeError("%s seed %d mixes failed runs %s with pending runs %s; run_all would retrain the "
                               "failed ones. Run the pending model by hand with --models." % (job.config, job.seed, blocked, todo))
        blocked_all.extend(blocked)
        if todo:
            pending.append(Pending(job, todo, round(sum(nominal_eq(m, job.profile) for _, m in todo), 6)))
    return pending, blocked_all


def _load(inflight: Path) -> list[dict]:
    path = REPO / inflight
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _save(inflight: Path, entries: list[dict]) -> None:
    path = REPO / inflight
    if not entries:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _train_minutes(run_dir: Path, fallback: float) -> float:
    summary = REPO / run_dir / "summary.json"
    if summary.exists():
        return float(json.loads(summary.read_text(encoding="utf-8"))["train_seconds"]) / 60.0
    return fallback


def recover_interrupted(ledger: Path = LEDGER, inflight: Path = INFLIGHT) -> list[dict]:
    rows = []
    for entry in _load(inflight):
        for run_id, model in entry["todo"]:
            run_dir = Path(entry["out_dir"]) / run_id
            status = gpu_ledger.run_status(REPO / run_dir)
            rows.append(gpu_ledger.record(run_dir, entry["phase"], nominal_eq(model, entry["profile"]),
                                          _train_minutes(run_dir, 0.0), entry["workers"], model, entry["profile"],
                                          sigma_of(run_id), entry["seed"], ledger,
                                          status="interrupted" if status == "missing" else None))
    _save(inflight, [])
    return rows


def _log_name(job: Job) -> str:
    return "run_core_%s_seed%d_%s.log" % ("-".join(job.models), job.seed, "-".join(sigma_tag(s) for s in job.noise))


def execute(pending: list[Pending], workers: int, cap: float, ledger: Path = LEDGER, inflight: Path = INFLIGHT) -> int:
    queue = list(pending)
    running: list[tuple[subprocess.Popen, Pending, float, object]] = []
    failures: list[str] = []
    while queue or running:
        while queue and len(running) < workers:
            item = queue[0]
            charged = gpu_ledger.total_eq(gpu_ledger.read(ledger))
            in_flight = sum(entry[1].eq for entry in running)
            if not budget_allows(charged, in_flight, item.eq, cap):
                if running:
                    break
                print("BUDGET STOP: charged %.2f + next job %.2f > cap %.2f; %d jobs not started"
                      % (charged, item.eq, cap, len(queue)), flush=True)
                return 2
            queue.pop(0)
            out_dir = REPO / item.job.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            log = open(out_dir / _log_name(item.job), "a", encoding="utf-8")
            _save(inflight, _load(inflight) + [{
                "phase": item.job.phase, "out_dir": item.job.out_dir, "profile": item.job.profile,
                "seed": item.job.seed, "workers": workers, "todo": [list(t) for t in item.todo],
                "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}])
            print("START %-20s seed %d  %s  (%.2f eq)" % (item.job.stem, item.job.seed,
                                                          [r for r, _ in item.todo], item.eq), flush=True)
            proc = subprocess.Popen([sys.executable, "-m", "scripts.run_all", *item.job.run_all_argv()],
                                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, item, time.perf_counter(), log))
        time.sleep(POLL_SECONDS)
        for entry in list(running):
            proc, item, started, log = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            log.close()
            share = (time.perf_counter() - started) / 60.0 / len(item.todo)
            for run_id, model in item.todo:
                run_dir = Path(item.job.out_dir) / run_id
                row = gpu_ledger.record(run_dir, item.job.phase, nominal_eq(model, item.job.profile),
                                        _train_minutes(run_dir, share), workers, model, item.job.profile,
                                        sigma_of(run_id), item.job.seed, ledger)
                print("END   %s  status=%s  %s min" % (row["run_dir"], row["status"], row["minutes"]), flush=True)
                if row["status"] != "ok":
                    failures.append(row["run_dir"])
            _save(inflight, [e for e in _load(inflight)
                             if not (e["out_dir"] == item.job.out_dir and e["todo"] == [list(t) for t in item.todo])])
    rows = gpu_ledger.read(ledger)
    print("Queue finished: %d failed; ledger total %.2f eq (cap %.2f)" % (len(failures), gpu_ledger.total_eq(rows), cap))
    for path in failures:
        print("  FAILED %s" % path)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", nargs="+", required=True, choices=PHASES)
    parser.add_argument("--workers", type=int, default=1, choices=[1, 2])
    parser.add_argument("--cap", type=float, default=CORE_CAP)
    parser.add_argument("--list", action="store_true", help="print the pending jobs and exit")
    parser.add_argument("--retry-infra", action="store_true")
    parser.add_argument("--allow-drift", default=None, metavar="REASON")
    args = parser.parse_args(argv)
    if args.cap > HARD_CAP:
        parser.error("--cap %.2f is above the hard cap %.2f" % (args.cap, HARD_CAP))

    realistic = realistic_k()
    jobs = select(args.phase, realistic)
    if any(j.profile == "full" for j in jobs):
        state, detail = registration_state()
        if state == "no_tag":
            print("REFUSED: %s. Full-profile runs start only after the pre-registration." % detail)
            return 3
        if state == "drift":
            if not args.allow_drift:
                print("REFUSED: %s. Pass --allow-drift REASON to proceed and log it." % detail)
                return 3
            log_deviation(args.allow_drift, detail, "run_core")
    if args.workers == 2 and set(args.phase) != {"smoke-pair"}:
        decision = json.loads((REPO / CONCURRENCY).read_text(encoding="utf-8")) if (REPO / CONCURRENCY).exists() else {}
        if decision.get("workers") != 2:
            print("REFUSED: --workers 2 needs %s with workers=2 (throughput and determinism check)" % CONCURRENCY)
            return 3
    if not args.list:
        for row in recover_interrupted():
            print("RECOVERED %s  status=%s  %.2f eq" % (row["run_dir"], row["status"], float(row["eq"])))

    pending, blocked = plan(jobs, args.retry_infra)
    for p in pending:
        print("  %-11s %-20s seed %d  %-44s %.2f eq" % (p.job.phase, p.job.stem, p.job.seed,
                                                       ",".join(r for r, _ in p.todo), p.eq))
    for path in blocked:
        print("  BLOCKED (failed, not rerun) %s" % path)
    charged = gpu_ledger.total_eq(gpu_ledger.read())
    print("%d jobs pending, %.2f eq to charge; ledger holds %.2f eq; cap %.2f eq"
          % (len(pending), sum(p.eq for p in pending), charged, args.cap))
    if args.list:
        return 0
    return execute(pending, args.workers, args.cap)


if __name__ == "__main__":
    raise SystemExit(main())

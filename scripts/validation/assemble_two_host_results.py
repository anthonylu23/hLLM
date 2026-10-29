"""Machine-readable summary of a two-host qualification run directory.

Every value is read from preserved files; missing evidence is reported as absent,
never filled in. Run it after the launcher has stopped both guards, so guard records
end with their exit events. The layout is the one `scripts/validation/two_host/`
writes (and the October 2026 runs used):

- profiles: a directory holding `cuda-memory.*` or `mlx-memory.*` and the matching
  `cuda-profile.guard.jsonl` / `mlx-profile.guard.jsonl`. The run directory itself is
  `cuda` and `mac/` is `mlx`; `mac/rejected-N/` directories are listed under
  `mlx_rejected`; any other directory `<backend>-profile-<label>` (also directly in
  the run directory) is `<backend>_<label>`.
- baselines: the run directory (`final`) and every `baseline-<label>/` holding
  `baseline-reload-soak.json`.
- P5 attempts: every `concurrent-attempt-*/` and then the run directory; the last is
  `p5`. Besides the guard summaries sealed in the evidence, each attempt lists the
  final guard records the launcher copied after stopping the guards (`final_guards`),
  and `guard_stops` names every guard that stopped its worker itself rather than
  being stopped by the launcher, even when the sealed record missed it.

The output names the run directory by its base name only, so no host path is
recorded. `--link KEY=VALUE` adds top-level cross-references (for example the
summary of an earlier night in the same directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

ATTEMPT_FILES = re.compile(r"^concurrent.*\.jsonl$")
# The exit reason a guard records when the launcher's cleanup stops it.
LAUNCHER_STOP = "guard received signal"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def text(path: Path) -> str | None:
    return path.read_text().strip() if path.is_file() else None


class Summary:
    def __init__(self, root: Path) -> None:
        self.root = root

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)) if path != self.root else "."

    def guard(self, path: Path) -> dict | None:
        if not path.is_file():
            return None
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        observations = [r for r in rows if not r.get("event") and "available_bytes" in r]
        exit_ = next((r for r in rows if r.get("event") == "exit"), None)
        times = [r["unix_time"] for r in rows if "unix_time" in r]
        gaps = [b - a for a, b in pairwise(times)]
        return dict(
            file=self.relative(path),
            record_sha256=sha(path),
            observations=len(observations),
            max_gap_seconds=round(max(gaps), 1) if gaps else None,
            stop_reason=exit_ and exit_.get("reason"),
            returncode=exit_ and exit_.get("returncode"),
            min_available_gib=round(min(r["available_bytes"] for r in observations) / 2**30, 3)
            if observations
            else None,
            max_available_gib=round(max(r["available_bytes"] for r in observations) / 2**30, 3)
            if observations
            else None,
            swapout_mib=round(
                (observations[-1]["swapout_bytes"] - observations[0]["swapout_bytes"]) / 2**20
            )
            if observations
            else None,
        )

    def profile(self, d: Path, backend: str) -> dict:
        stem = f"{backend}-memory"
        artifact = d / f"{stem}.json"
        preflight = d / f"{stem}.preflight.json"
        record: dict = dict(directory=self.relative(d), note=text(d / "README.txt"))
        if preflight.is_file():
            p = load(preflight)
            record["preflight"] = dict(
                status=p["status"],
                reasons=p["reasons"],
                schema=p["schema_version"],
                host_available_bytes=p["host_budget"]["available_bytes"],
                host_available_after_checkpoint_read=p.get(
                    "host_available_bytes_after_checkpoint_read"
                ),
                sampled_before_checkpoint_read=p.get(
                    "availability_sampled_before_checkpoint_read", False
                ),
                source_revision=p["key"]["environment"]["source_revision"],
                binary_digest=p["key"]["environment"]["binary_digest"],
                admission_capacity=p["admission_capacity"],
            )
        if artifact.is_file():
            a = load(artifact)
            fit = load(d / f"{stem}.fit.json")["assessment"]
            record["artifact"] = dict(
                artifact_digest=a["artifact_digest"],
                measured_at=a["conditions"]["measured_at"],
                completed=a["measurement"]["completed"],
                source_revision=a["key"]["environment"]["source_revision"],
                binary_digest=a["key"]["environment"]["binary_digest"],
                fit=dict(
                    status=fit["status"],
                    policy=fit.get("policy"),
                    requested_mlx_policy=fit.get("requested_mlx_policy"),
                    host_envelope_bytes=fit.get("host_envelope_bytes"),
                    device_envelope_bytes=fit.get("device_envelope_bytes"),
                    reasons=fit["reasons"],
                    notes=fit.get("policy_notes", []),
                ),
            )
        record["guard"] = self.guard(d / f"{backend}-profile.guard.jsonl")
        return record

    def profiles(self) -> dict:
        result: dict = {}
        mac = self.root / "mac"
        subdirectories = sorted(
            p
            for base in (self.root, mac)
            if base.is_dir()
            for p in base.iterdir()
            if p.is_dir() and p != mac
        )
        for d in [self.root, mac, *subdirectories]:
            for backend in ("cuda", "mlx"):
                stem = f"{backend}-memory"
                if not ((d / f"{stem}.json").is_file() or (d / f"{stem}.preflight.json").is_file()):
                    continue
                if d in (self.root, mac):
                    if (d == mac) != (backend == "mlx"):
                        continue
                    result[backend] = self.profile(d, backend)
                elif d.name.startswith("rejected-"):
                    result.setdefault(f"{backend}_rejected", []).append(self.profile(d, backend))
                else:
                    label = d.name.removeprefix(f"{backend}-profile-").replace("-", "_")
                    result[f"{backend}_{label}"] = self.profile(d, backend)
        return result

    def baseline(self, d: Path) -> dict:
        p = d / "baseline-reload-soak.json"
        if not p.is_file():
            return dict(directory=self.relative(d), present=False, note=text(d / "README.txt"))
        b = load(p)
        cycles = b.get("cycles", [])
        results = [r for c in cycles for r in c.get("requests", [])]
        return dict(
            directory=self.relative(d),
            present=True,
            report_sha256=sha(p),
            completed=b.get("completed"),
            scope=b.get("scope"),
            plan_digest=b["plan"]["plan_digest"],
            cycles=len(cycles),
            requests=len(results),
            # No requests is no exactness evidence, not vacuously exact.
            all_exact=bool(results) and all(r.get("exact_reference_match") for r in results),
            worker_binaries={k: v["binary_digest"] for k, v in b["workers"].items()},
            guards=dict(
                cuda=self.guard(d / "baseline-cuda.guard.jsonl"),
                mlx=self.guard(d / "mac/final/baseline-mlx.guard.jsonl")
                or self.guard(d / "mac/baseline-mlx.guard.jsonl")
                or self.guard(d / "baseline-mlx.guard.jsonl"),
            ),
            note=text(d / "README.txt"),
        )

    def baselines(self) -> dict:
        result = {
            d.name.removeprefix("baseline-").replace("-", "_"): self.baseline(d)
            for d in sorted(self.root.glob("baseline-*"))
            if d.is_dir()
        }
        result["final"] = self.baseline(self.root)
        return result

    def attempt(self, d: Path) -> dict:
        e = d / "concurrent.json"
        record: dict = dict(
            directory=self.relative(d), note=text(d / "README.txt"), evidence_present=e.is_file()
        )
        if not e.is_file():
            return record
        ev = load(e)
        res = ev["results"]
        baseline = ev["preconditions"]["baseline"]
        record.update(
            evidence_sha256=sha(e),
            evidence_digest=ev["evidence_digest"],
            schema=ev["schema_version"],
            acceptance=res["acceptance"],
            reasons=res["acceptance_reasons"],
            failure_reason=res["failure_reason"],
            finished_at=res.get("finished_at"),
            source_digest=ev["identity"]["source_digest"],
            workers=[
                dict(
                    worker=w["worker_id"],
                    binary_digest=w["worker_binary_digest"],
                    process_id=w["process_id"],
                )
                for w in ev["identity"]["workers"]
            ],
            baseline={k: baseline[k] for k in ("scope", "cycles", "requests", "all_exact")},
            profiles=[
                dict(
                    worker=p["worker_id"],
                    artifact_digest=p["artifact_digest"],
                    measured_at=p["measured_at"],
                )
                for p in ev["preconditions"]["profiles"]
            ],
            gates=[
                dict(
                    worker=g["worker_id"],
                    cycle=g["cycle"],
                    sampled_at=g["sampled_at"],
                    status=g["status"],
                    reasons=g["reasons"],
                    host_available_bytes=g["host_budget"]["available_bytes"],
                    device_available_bytes=(g["device_budget"] or {}).get("available_bytes"),
                )
                for g in ev["preconditions"]["gates"]
            ],
            rounds=len(res["rounds"]),
            comparisons=len(res["comparisons"]),
            exact=sum(c["exact"] for c in res["comparisons"]),
            overlap_rows=len(res["overlap"]),
            physical=[
                dict(
                    worker=p["worker_id"],
                    cycle=p["cycle"],
                    policy=p["policy"],
                    status=p["status"],
                    reasons=p["reasons"],
                    host_peak_bytes=p["host_peak_bytes"],
                    host_envelope_bytes=p["host_envelope_bytes"],
                    device_peak_bytes=p["device_peak_bytes"],
                    device_envelope_bytes=p["device_envelope_bytes"],
                    host_available_bytes=p["host_budget"]["available_bytes"],
                    device_available_bytes=(p["device_budget"] or {}).get("available_bytes"),
                )
                for p in res["physical"]
            ],
            logical_peaks=[
                dict(worker=p["worker_id"], within_admission=p["within_admission"])
                for p in res["logical_peaks"]
            ],
            guards=[
                dict(
                    worker=g["worker_id"],
                    outcome=g["outcome"],
                    reasons=g["reasons"],
                    observations=g["observations"],
                    process_id=g["process_id"],
                    record_sha256=g["record_sha256"],
                )
                for g in res["guards"]
            ],
            cleanup=[dict(worker=c["worker_id"], verified=c["verified"]) for c in res["cleanup"]],
            snapshots={
                w["worker_id"]: dict(
                    count=len(w["snapshots"]),
                    allocator_missing=sum(1 for s in w["snapshots"] if s["allocator"] is None),
                    device_samples=len(w["device_samples"]),
                    peak_concurrent_requests=max(
                        (s["peak_concurrent_requests"] for s in w["snapshots"]), default=0
                    ),
                )
                for w in ev["observations"]["workers"]
            },
            configuration={k: v for k, v in ev["configuration"].items() if k != "workers"}
            | dict(workers=ev["configuration"]["workers"]),
        )
        # Guard snapshots (`concurrent.guard-<worker>.jsonl`), mirrors, final copies,
        # device samples and the observation stream, each with its hash.
        files = {}
        for base in (d, d / "mac", d / "mac" / "final"):
            if base.is_dir():
                for p in sorted(base.iterdir()):
                    if p.is_file() and ATTEMPT_FILES.match(p.name):
                        files[str(p.relative_to(d))] = dict(
                            sha256=sha(p), rows=len(p.read_text().splitlines())
                        )
        if files:
            record["files"] = files
        final_guards = dict(
            cuda=self.guard(d / "concurrent-cuda.guard.jsonl"),
            mlx=self.guard(d / "mac/final/concurrent-mlx.guard.jsonl")
            or self.guard(d / "mac/concurrent-mlx.guard.jsonl"),
        )
        record["final_guards"] = final_guards
        record["guard_stops"] = [
            dict(worker=k, reason=g["stop_reason"])
            for k, g in final_guards.items()
            if g and g["stop_reason"] and not g["stop_reason"].startswith(LAUNCHER_STOP)
        ]
        return record

    def identities(self, directory: Path, pattern: str, exclude: set[str]) -> dict:
        return {
            p.stem: dict(
                base_revision=(i := load(p))["base_revision"],
                source_digest=i["source_digest"],
                binaries=i["binaries"],
            )
            for p in sorted(directory.glob(pattern))
            if p.stem not in exclude
        }

    def summary(self, links: dict[str, str]) -> dict:
        r = self.root
        result: dict = dict(
            run_directory=r.name,
            **links,
            source_identities=dict(
                linux=self.identities(
                    r / "inputs", "source-identity-*.json", {"source-identity-pre-build"}
                ),
                mac=self.identities(r / "mac", "source-identity*.json", set()),
            ),
            inputs={p.name: sha(p) for p in sorted((r / "inputs").glob("*.json"))},
            p1=self.profiles(),
            baselines=self.baselines(),
            p5_attempts=[self.attempt(d) for d in sorted(r.glob("concurrent-attempt-*"))]
            + [self.attempt(r)],
        )
        result["p5"] = result["p5_attempts"][-1]
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output", type=Path, help="summary JSON to write (not overwritten)")
    parser.add_argument("--link", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    links = dict(item.split("=", 1) for item in args.link)
    result = Summary(args.run_directory.resolve()).summary(links)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempts = [(a["directory"], a.get("acceptance", "no evidence")) for a in result["p5_attempts"]]
    print("attempts:", attempts)
    final = result["p5"]
    if final.get("evidence_present"):
        print(
            "final:",
            final["acceptance"],
            final["evidence_digest"][:16],
            "exact",
            final["exact"],
            "/",
            final["comparisons"],
        )


if __name__ == "__main__":
    main()

"""Measure repeated native assignments without treating reservations as physical usage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from checkpoint_run import make_plan, snapshots
from hllm_control.controller import DeploymentSession
from hllm_control.models import DType, ModelManifest


def require_clean(samples: list[dict], *, unloaded: bool) -> None:
    fields = ["activeRequests", "reservedCacheBytes", "reservedWorkspaceBytes"]
    if unloaded:
        fields.append("loadedWeightBytes")
    if any(int(sample["memory"].get(field, 0)) for sample in samples for field in fields):
        raise RuntimeError("native reservations did not retire")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", nargs="+", choices=["mlx", "cuda"], default=["cuda"])
    parser.add_argument("--mlx-endpoint", default="127.0.0.1:50161")
    parser.add_argument("--cuda-endpoint", default="127.0.0.1:50163")
    parser.add_argument("--split", type=int, default=14)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    manifest = ModelManifest.model_validate_json(args.manifest.read_text())
    reference = json.loads(args.reference.read_text())["long_generation"]
    if args.cycles < 1 or not 1 <= args.new_tokens <= len(reference["generated_ids"]):
        parser.error("cycles must be positive and new-tokens must fit the reference")
    plan = make_plan(manifest, args.workers, DType.F16, args.split)
    endpoints = {"mlx": args.mlx_endpoint, "cuda": args.cuda_endpoint}
    result = {
        "label": args.label,
        "manifest_digest": manifest.manifest_digest,
        "plan": plan.model_dump(mode="json"),
        "maximum_total_tokens": len(reference["token_ids"]) + args.new_tokens,
        "started_unix_time": time.time(),
        "before": snapshots(args.workers, endpoints),
        "cycles": [],
        "completed": False,
    }
    require_clean(result["before"], unloaded=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cycle in range(args.cycles):
        record = {"cycle": cycle, "started_unix_time": time.time()}
        result["cycles"].append(record)
        try:
            start = time.perf_counter()
            with DeploymentSession(manifest, plan, endpoints) as session:
                record["load_seconds"] = time.perf_counter() - start
                record["after_load"] = snapshots(args.workers, endpoints)
                generated = []
                start = time.perf_counter()
                for event in session.generate(
                    reference["token_ids"],
                    maximum_new_tokens=args.new_tokens,
                    stop_token_ids=[],
                    timeout=args.timeout,
                ):
                    if event.HasField("token"):
                        if not generated:
                            record["time_to_first_token_seconds"] = time.perf_counter() - start
                            record["at_first_observed_token"] = snapshots(args.workers, endpoints)
                        generated.append(event.token.token_id)
                record["instrumented_generation_seconds"] = time.perf_counter() - start
                record["generated_ids"] = generated
                record["exact_reference_match"] = (
                    generated == reference["generated_ids"][: args.new_tokens]
                )
                record["after_request"] = snapshots(args.workers, endpoints)
                require_clean(record["after_request"], unloaded=False)
            record["after_unload"] = snapshots(args.workers, endpoints)
            require_clean(record["after_unload"], unloaded=True)
            if not record["exact_reference_match"]:
                raise RuntimeError("generation differs from the reference; see recorded cycle")
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["finished_unix_time"] = time.time()
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"cycle": cycle, "after_unload": record["after_unload"]}), flush=True)
    result["completed"] = True
    result["finished_unix_time"] = time.time()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

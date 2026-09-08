"""Run a prepared checkpoint on one or two explicitly selected native workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict
from hllm_control.controller import DeploymentSession
from hllm_control.models import DeploymentPlan, DType, ModelManifest, PlanningMode, StageAssignment
from hllm_control.proto import common_pb2, control_pb2, control_pb2_grpc
from hllm_control.serialization import canonical_json_bytes
from hllm_control.wire import deployment_plan_to_proto, model_manifest_to_proto


def make_plan(
    manifest: ModelManifest, names: list[str], dtype: DType, split: int = 14
) -> DeploymentPlan:
    if (
        len(names) not in (1, 2)
        or len(set(names)) != len(names)
        or (len(names) == 2 and not 0 < split < manifest.config.num_layers)
    ):
        raise ValueError("expected one worker or two workers with an interior split")
    ranges = (
        [(0, manifest.config.num_layers)]
        if len(names) == 1
        else [(0, split), (split, manifest.config.num_layers)]
    )
    plan = DeploymentPlan(
        plan_id="qualification-" + "-".join(names),
        plan_digest="qualification-plan",
        manifest_digest=manifest.manifest_digest,
        workload_id="full-checkpoint",
        planning_mode=PlanningMode.FEASIBILITY,
        execution_dtype=dtype,
        activation_dtype=DType.F16,
        split_layer=split if len(names) == 2 else 0,
        selected_candidate_id="qualification",
        duplicated_tensor_groups=("token_embeddings",)
        if len(names) == 2 and manifest.config.tied_embeddings
        else (),
        stages=tuple(
            StageAssignment(
                stage_index=i,
                worker_id=name,
                layer_start=start,
                layer_end=end,
                owns_token_embedding=i == 0,
                owns_final_norm=i == len(names) - 1,
                owns_lm_head=i == len(names) - 1,
                owns_sampling=i == len(names) - 1,
            )
            for i, (name, (start, end)) in enumerate(zip(names, ranges, strict=True))
        ),
    )
    digest = hashlib.sha256(
        canonical_json_bytes(plan.model_dump(mode="json", exclude={"plan_id", "plan_digest"}))
    ).hexdigest()
    return plan.model_copy(
        update={"plan_id": f"qualification-{digest[:16]}", "plan_digest": digest}
    )


def snapshots(names: list[str], endpoints: dict[str, str]) -> list[dict]:
    result = []
    for name in names:
        with grpc.insecure_channel(endpoints[name]) as channel:
            stub = control_pb2_grpc.WorkerControlStub(channel)
            memory = stub.GetMemoryReport(common_pb2.Empty(), timeout=5)
            metrics = stub.GetMetrics(common_pb2.Empty(), timeout=5)
            result.append(
                {"worker": name, "memory": MessageToDict(memory), "metrics": MessageToDict(metrics)}
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", nargs="+", choices=("mlx", "cuda"), required=True)
    parser.add_argument("--mlx-endpoint", default="127.0.0.1:50161")
    parser.add_argument("--cuda-endpoint", default="127.0.0.1:50163")
    parser.add_argument("--split", type=int, default=14)
    parser.add_argument("--dtype", choices=("F32", "F16"), default="F16")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--probe-spec-only", action="store_true")
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Require exact agreement for the 256-token reference continuation",
    )
    parser.add_argument("--budget-unified", type=int, default=4 * 1024**3)
    parser.add_argument("--budget-host", type=int, default=2 * 1024**3)
    parser.add_argument("--budget-device", type=int, default=6 * 1024**3)
    args = parser.parse_args()
    manifest = ModelManifest.model_validate_json(args.manifest.read_text())
    plan = make_plan(manifest, args.workers, DType(args.dtype), args.split)
    endpoints = {"mlx": args.mlx_endpoint, "cuda": args.cuda_endpoint}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.probe_spec_only:
        if min(args.budget_unified, args.budget_host, args.budget_device) <= 0:
            parser.error("probe budgets must be positive byte counts")
        if len(args.workers) != 1:
            parser.error("a full-model numerical probe requires exactly one worker")
        load = control_pb2.LoadStageRequest(
            plan=deployment_plan_to_proto(plan),
            manifest=model_manifest_to_proto(manifest),
            stage_index=0,
        )
        args.output.write_text(
            json.dumps(
                {
                    "load_request": MessageToDict(load),
                    "budget": (
                        {"unified": args.budget_unified}
                        if args.workers == ["mlx"]
                        else {"host": args.budget_host, "device": args.budget_device}
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        return
    reference = json.loads(args.reference.read_text())
    long = reference["long_generation"]
    if len(long["generated_ids"]) != 256:
        raise ValueError("reference long_generation.generated_ids must contain exactly 256 tokens")
    result = {
        "workers": args.workers,
        "plan": plan.model_dump(mode="json"),
        "runs": [],
        "started_unix_time": time.time(),
    }
    with DeploymentSession(manifest, plan, endpoints) as session:
        result["loaded_memory"] = [MessageToDict(r) for r in session.memory_reports()]
        prompts = [(case["prompt"], case["token_ids"], 16, None) for case in reference["cases"]]
        prompts.append((long["prompt"], long["token_ids"], 256, long["generated_ids"]))
        for prompt, ids, count, expected in prompts:
            start = time.perf_counter()
            first = None
            generated = []
            active = None
            for event in session.generate(
                ids, maximum_new_tokens=count, stop_token_ids=[], timeout=args.timeout
            ):
                if event.HasField("token"):
                    if first is None:
                        first = time.perf_counter() - start
                        active = [MessageToDict(r) for r in session.memory_reports()]
                    generated.append(event.token.token_id)
            elapsed = time.perf_counter() - start
            if len(generated) != count:
                raise RuntimeError(f"expected {count} tokens, received {len(generated)}")
            reports = session.memory_reports()
            if any(
                r.active_requests or r.reserved_cache_bytes or r.reserved_workspace_bytes
                for r in reports
            ):
                raise RuntimeError("completed request retained sequence reservations")
            prefix = 0
            if expected:
                for actual, target in zip(generated, expected, strict=True):
                    if actual != target:
                        break
                    prefix += 1
            record = {
                "prompt": prompt,
                "prompt_tokens": len(ids),
                "generated_ids": generated,
                "seconds": elapsed,
                "time_to_first_token_seconds": first,
                "active_memory": active,
                "completed_memory": [MessageToDict(r) for r in reports],
                "reference_common_prefix": prefix if expected else None,
            }
            result["runs"].append(record)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "workers": args.workers,
                        "count": count,
                        "seconds": elapsed,
                        "ttft": first,
                        "reference_common_prefix": record["reference_common_prefix"],
                    }
                ),
                flush=True,
            )
    result["after_unload"] = snapshots(args.workers, endpoints)
    for entry in result["after_unload"]:
        memory = entry["memory"]
        if any(
            int(memory.get(key, 0))
            for key in (
                "loadedWeightBytes",
                "activeRequests",
                "reservedCacheBytes",
                "reservedWorkspaceBytes",
            )
        ):
            raise RuntimeError("unload retained model or sequence residency")
    result["completed"] = True
    result["exact_reference_match"] = all(
        item["reference_common_prefix"] == len(item["generated_ids"])
        for item in result["runs"]
        if item["reference_common_prefix"] is not None
    )
    result["finished_unix_time"] = time.time()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.require_exact and not result["exact_reference_match"]:
        raise RuntimeError("generation differs from the reference; see recorded results")


if __name__ == "__main__":
    main()

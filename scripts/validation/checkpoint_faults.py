"""Exercise cancellation, deadlines and admission on already running split workers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import grpc
from checkpoint_run import make_plan, snapshots
from hllm_control.controller import DeploymentSession
from hllm_control.models import DType, ModelManifest


def clean(session: DeploymentSession) -> None:
    until = time.monotonic() + 10
    while True:
        if all(
            not (r.active_requests or r.reserved_cache_bytes or r.reserved_workspace_bytes)
            for r in session.memory_reports()
        ):
            return
        if time.monotonic() >= until:
            raise RuntimeError("sequence reservations did not retire")
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mlx-endpoint", required=True)
    parser.add_argument("--cuda-endpoint", required=True)
    parser.add_argument("--split", type=int, default=14)
    parser.add_argument("--decode-deadline", type=float, default=1.5)
    parser.add_argument(
        "--admission-tokens",
        type=int,
        help="Total context to reserve; default is the model context limit",
    )
    args = parser.parse_args()
    manifest = ModelManifest.model_validate_json(args.manifest.read_text())
    reference = json.loads(args.reference.read_text())["long_generation"]
    ids = reference["token_ids"]
    admission_tokens = (
        args.admission_tokens
        if args.admission_tokens is not None
        else manifest.config.maximum_sequence_length
    )
    if not len(ids) < admission_tokens <= manifest.config.maximum_sequence_length:
        parser.error("admission-tokens must exceed the prompt and fit the model context")
    if args.decode_deadline <= 0:
        parser.error("decode-deadline must be positive")
    endpoints = {"mlx": args.mlx_endpoint, "cuda": args.cuda_endpoint}
    results = []
    for names in (["mlx", "cuda"], ["cuda", "mlx"]):
        plan = make_plan(manifest, names, DType.F16, args.split)
        with DeploymentSession(manifest, plan, endpoints) as session:
            for fault in ("client_cancel", "deadline", "admission"):
                started = time.monotonic()
                received = 0
                rejected = False
                stream = session.generate(
                    ids,
                    maximum_new_tokens=(
                        admission_tokens - len(ids) if fault == "admission" else 256
                    ),
                    stop_token_ids=[],
                    timeout=args.decode_deadline if fault == "deadline" else 30,
                )
                try:
                    for event in stream:
                        if event.HasField("token"):
                            received += 1
                            if fault == "client_cancel":
                                break
                except grpc.RpcError as error:
                    expected = (
                        grpc.StatusCode.RESOURCE_EXHAUSTED
                        if fault == "admission"
                        else grpc.StatusCode.DEADLINE_EXCEEDED
                    )
                    if error.code() != expected:
                        raise
                    rejected = True
                except RuntimeError as error:
                    # Generation surfaces a native deadline as a non-completed terminal.
                    if fault != "deadline" or "DEADLINE_EXCEEDED" not in str(error):
                        raise
                    rejected = True
                finally:
                    stream.close()
                if fault == "client_cancel" and received != 1:
                    raise RuntimeError("cancellation did not happen after the first token")
                if fault != "client_cancel" and not rejected:
                    raise RuntimeError("expected request rejection")
                if fault == "deadline" and received == 0:
                    raise RuntimeError("deadline expired before decode was exercised")
                clean(session)
                recovered = [
                    e.token.token_id
                    for e in session.generate(
                        ids, maximum_new_tokens=4, stop_token_ids=[], timeout=30
                    )
                    if e.HasField("token")
                ]
                if recovered != reference["generated_ids"][:4]:
                    raise RuntimeError("fresh generation did not recover")
                clean(session)
                results.append(
                    {
                        "workers": names,
                        "fault": fault,
                        "tokens_before_fault": received,
                        "seconds_including_recovery": time.monotonic() - started,
                        "recovered_ids": recovered,
                    }
                )
                print(json.dumps(results[-1]), flush=True)
        after = snapshots(names, endpoints)
        if any(int(e["memory"].get("loadedWeightBytes", 0)) for e in after):
            raise RuntimeError("unload retained weights")
    args.output.write_text(json.dumps({"passed": True, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()

# hLLM Runtime

hLLM Runtime is a heterogeneous inference system designed to split one decoder-only
language model across different accelerator platforms. The first target is an Apple MLX
worker and an NVIDIA CUDA worker connected through Tailscale.

The repository currently implements Milestone 0: model inspection, versioned manifests,
hardware and workload profiles, memory estimation, and exhaustive two-worker placement
planning. It does not run distributed inference yet.

## Development setup

Install `uv`, then let it provision the pinned Python version and dependencies:

```bash
uv sync --python 3.12
uv run hllm --help
```

Run the quality suite with:

```bash
uv run pytest
uv run ruff check .
uv run pyright
mkdir -p build/proto
uv run python -m grpc_tools.protoc -Iproto --python_out=build/proto \
  proto/common.proto proto/model.proto proto/profile.proto proto/placement.proto \
  proto/control.proto proto/execution.proto proto/telemetry.proto
```

## Prepare and plan

Prepare an indexed or single-file Llama-compatible Safetensors model without loading its
tensor payloads:

```bash
uv run hllm prepare /models/Falcon3-3B-Base \
  --model-id tiiuae/Falcon3-3B-Base \
  --revision <pinned-hugging-face-revision> \
  --output build/falcon3.manifest.json
```

Enumerate both worker orders and every contiguous split point:

```bash
uv run hllm plan \
  --manifest build/falcon3.manifest.json \
  --workers examples/profiles/workers-m3pro-3060ti.yaml \
  --links examples/profiles/links-m3pro-3060ti.yaml \
  --workload examples/workloads/interactive.yaml \
  --settings examples/profiles/planner-feasibility.yaml \
  --output build/deployment-plan.json \
  --report build/planning-report.json
```

The checked-in memory budgets remain conservative configured estimates. Link RTT and
effective directional throughput are point-in-time observations from the test pair;
Milestone 5 replaces them with dry-load, compute, conversion, and payload-specific link
measurements.

See [the Milestone 0 implementation notes](docs/milestone-0.md) and [the full project
specification](SPEC.md). The [validation report](docs/validation/milestone-0.md) records the
pinned real-model and cross-platform checks.

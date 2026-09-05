# hLLM Runtime

hLLM Runtime is a heterogeneous inference system designed to split one decoder-only
language model across different accelerator platforms. The first target is an Apple MLX
worker and an NVIDIA CUDA worker connected through Tailscale.

The repository currently implements Milestone 0: model inspection, versioned manifests,
hardware and workload profiles, memory estimation, and exhaustive two-worker placement
planning. Native CPU reference kernels, tensor buffers, Safetensors loading, and a
gRPC control service are also present. The service validates stage metadata and tracks
reservations; it does not yet load executable stages or run distributed inference.

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
uv run python scripts/generate_proto.py --check
```

For native development, install the C++ Protobuf and gRPC development packages
(including their CMake configs), then run:

```bash
uv run cmake --preset dev
uv run cmake --build --preset dev
uv run ctest --preset dev
```

Use the `asan` preset in all three commands to enable AddressSanitizer and
UndefinedBehaviorSanitizer. CMake uses installed GoogleTest and nlohmann_json packages
when available and fetches pinned versions otherwise. After changing schemas, run
`uv run python scripts/generate_proto.py` to refresh the checked-in Python bindings.

## Prepare and plan

Prepare an indexed or single-file Llama-compatible or dense Qwen3 Safetensors model
without loading its tensor payloads. Qwen3-4B-Base is the new benchmark target:

```bash
uv run hllm prepare /models/Qwen3-4B-Base \
  --model-id Qwen/Qwen3-4B-Base \
  --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output build/qwen3.manifest.json
```

Falcon remains supported, with its existing results retained:

```bash
uv run hllm prepare /models/Falcon3-3B-Base \
  --model-id tiiuae/Falcon3-3B-Base \
  --revision <pinned-hugging-face-revision> \
  --output build/falcon3.manifest.json
```

Enumerate both worker orders and every contiguous split point:

```bash
uv run hllm plan \
  --manifest build/qwen3.manifest.json \
  --workers examples/profiles/workers-m3pro-3060ti.yaml \
  --links examples/profiles/links-m3pro-3060ti.yaml \
  --workload examples/workloads/interactive.yaml \
  --settings examples/profiles/planner-feasibility.yaml \
  --output build/qwen3.deployment-plan.json \
  --report build/qwen3.planning-report.json
```

The checked-in memory budgets remain conservative configured estimates. Link RTT and
effective directional throughput are point-in-time observations from the test pair;
Milestone 5 replaces them with dry-load, compute, conversion, and payload-specific link
measurements.

See [the Milestone 0 implementation notes](docs/milestone-0.md) and [the full project
specification](SPEC.md). The [validation report](docs/validation/milestone-0.md) records the
pinned real-model and cross-platform checks.

See [the code quality review](docs/code-quality-review.md) for fixes, validation, and
the remaining native integration work.

[Qwen3 support and validation](docs/qwen3.md) describes the pinned model, supported
semantics, independent CPU reference tests, and estimated placement. It is not yet a
full-checkpoint inference benchmark; distributed stage execution remains unfinished.

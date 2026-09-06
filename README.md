# hLLM Runtime

hLLM Runtime is a heterogeneous inference system designed to split one decoder-only
language model across different accelerator platforms. The first target is an Apple MLX
worker and an NVIDIA CUDA worker connected through Tailscale.

The repository implements model preparation and placement planning (Milestone 0) and a
native CPU pipeline (Milestone 1). Dense Llama and Qwen3 stages load assigned Safetensors
weights, run prefill and greedy decode across two native processes, and stream token IDs
through a Python controller. Request admission, cancellation, deadlines, and cleanup are
covered by numerical and process integration tests. Milestone 2 adds an optional Linux
CUDA worker that executes Llama/Qwen3 with F32/F16 weights and caches on the GPU. Tiny-model
numerical and mixed CPU/CUDA process tests cover both stage orders, pageable/pinned
boundaries, lifecycle faults and memory cleanup. See the
[mixed qualification report](docs/validation/mixed-cpu-cuda.md). Full-checkpoint
fit assessment and MLX remain next.

See [the CPU pipeline implementation and runnable demo](docs/milestone-1.md) and
[the model/backend extension boundaries](docs/model-extensibility.md).
See [the CUDA integration status and build instructions](docs/milestone-2.md) for
Milestone 2 implementation details and remaining work.

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

See [the code quality review](docs/code-quality-review.md) for historical fixes and
[the CUDA PR review](docs/code-quality-review-cuda.md) for the audit of PRs 5–9.

[Qwen3 support and validation](docs/qwen3.md) describes the pinned model, supported
semantics, independent CPU reference tests, and estimated placement. It is not yet a
full-checkpoint inference benchmark. Tiny-model CPU pipeline validation is documented
in the Milestone 1 notes.

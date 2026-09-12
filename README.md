# hLLM Runtime

hLLM Runtime is a heterogeneous inference system designed to split one decoder-only
language model across different accelerator platforms. The first target is an Apple MLX
worker and an NVIDIA CUDA worker connected through Tailscale.

Stages load assigned Safetensors weights, run prefill and greedy decode, and stream
token IDs through a Python controller. Request admission, cancellation, deadlines, and
cleanup are covered by numerical and process integration tests.

The approved M5 qualification target uses F16 resident weights, F32 execution/KV,
and F16 wire transfers. The production sweep is blocked on deadline-status validation
and timing drift; see the [failure investigation](docs/validation/milestone-5-planner.md#mixed-sweep-failure-investigation--2026-09-12).
Production acceptance and the final audit remain pending;
see the [precision target](docs/milestone-5-precision-proposal.md).

## Milestone status

| Milestone | Scope | Status | Notes |
| --- | --- | --- | --- |
| 0 | Model preparation and placement planning | Complete | [docs/milestone-0.md](docs/milestone-0.md) |
| 1 | Native CPU pipeline | Complete | [docs/milestone-1.md](docs/milestone-1.md) |
| 2 | Linux CUDA worker (F32/F16, pinned transfers) | Complete | [docs/milestone-2.md](docs/milestone-2.md), [mixed qualification](docs/validation/mixed-cpu-cuda.md) |
| 3 | Apple Silicon MLX worker (unified memory) | Complete | [docs/milestone-3.md](docs/milestone-3.md), [MLX qualification](docs/validation/mlx.md) |
| 4 | Full Qwen3-0.6B checkpoint across MLX/CUDA over Tailscale | Complete | [docs/milestone-4.md](docs/milestone-4.md), [cross-machine report](docs/validation/full-checkpoint-cross-machine.md) |
| 5 | Measured automatic placement | In progress | [docs/milestone-5.md](docs/milestone-5.md) |

Milestone 5.1–5.4 add versioned measured profiles, isolated native memory and
compute/conversion probes, directional native gRPC profiling, and admission/fit
checks. See the [profiling workflow](docs/milestone-5-profiling.md),
[memory qualification](docs/validation/milestone-5-memory.md), and
[timing qualification](docs/validation/milestone-5-timing.md). Measured planner
integration (5.5) and the independent sweep runner (5.6) are implemented; the full
54-candidate acceptance run remains pending. See the [placement workflow](docs/milestone-5-placement.md)
and [current validation/gates](docs/validation/milestone-5-planner.md).
Qwen3-4B-Base is blocked by the current Mac load preflight and remains unqualified.
The [Milestone 5 implementation plan](docs/milestone-5-implementation-plan.md) defines
the remaining slices and the initial 512-prompt/256-output, concurrency-1 workload.
Later milestones cover continuous batching (6) and ROCm with additional stages (7);
see [the full project specification](SPEC.md).

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
[Milestone 5](docs/milestone-5.md) replaces them with measured profiles.

See [the model/backend extension boundaries](docs/model-extensibility.md),
[Qwen3 support and validation](docs/qwen3.md), and
[the Milestone 0 implementation notes](docs/milestone-0.md). The
[validation report](docs/validation/milestone-0.md) records the pinned real-model and
cross-platform checks.

Historical code quality reviews: [general fixes](docs/code-quality-review.md),
[the CUDA PR audit](docs/code-quality-review-cuda.md), and
[the runtime and controller review](docs/code-quality-review-runtime.md).
Tiny-model CPU pipeline validation is documented in the Milestone 1 notes.

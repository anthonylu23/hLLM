# hLLM Runtime

hLLM Runtime is a heterogeneous inference system designed to split one decoder-only
language model across different accelerator platforms. The first target is an Apple MLX
worker and an NVIDIA CUDA worker connected through Tailscale.

Stages load assigned Safetensors weights, run prefill and greedy decode, and stream
token IDs through a Python controller. Request admission, cancellation, deadlines, and
cleanup are covered by numerical and process integration tests.

**Current status, open gates and next steps are in [docs/STATUS.md](docs/STATUS.md).**

Qualified precision targets, workloads and open decisions are on the status page. M5 WAN
performance acceptance is deferred; see the
[engineering closeout](docs/validation/milestone-5-wan-closeout.md).

## Milestone status

| Milestone | Scope | Status | Notes |
| --- | --- | --- | --- |
| 0 | Model preparation and placement planning | Complete | [docs/milestone-0.md](docs/milestone-0.md) |
| 1 | Native CPU pipeline | Complete | [docs/milestone-1.md](docs/milestone-1.md) |
| 2 | Linux CUDA worker (F32/F16, pinned transfers) | Complete | [docs/milestone-2.md](docs/milestone-2.md), [mixed qualification](docs/validation/mixed-cpu-cuda.md) |
| 3 | Apple Silicon MLX worker (unified memory) | Complete | [docs/milestone-3.md](docs/milestone-3.md), [MLX qualification](docs/validation/mlx.md) |
| 4 | Full Qwen3-0.6B checkpoint across MLX/CUDA over Tailscale | Complete | [docs/milestone-4.md](docs/milestone-4.md), [cross-machine report](docs/validation/full-checkpoint-cross-machine.md) |
| 5 | Measured automatic placement | Implemented; WAN acceptance deferred | [docs/milestone-5.md](docs/milestone-5.md) |
| 6 | Concurrent HTTP serving, sampling and native scheduling | Implemented; see workload-specific qualification | [docs/milestone-6.md](docs/milestone-6.md) |

Milestone 5.1–5.4 add versioned measured profiles, isolated native memory and
compute/conversion probes, directional native gRPC profiling, and admission/fit
checks. See the [profiling workflow](docs/milestone-5-profiling.md),
[memory qualification](docs/validation/milestone-5-memory.md), and
[timing qualification](docs/validation/milestone-5-timing.md). Measured planner
integration (5.5) and the independent sweep runner (5.6) are implemented; the
54-candidate WAN acceptance run is deferred. See the
[placement workflow](docs/milestone-5-placement.md) and
[current validation/gates](docs/validation/milestone-5-planner.md).

Qwen3-4B-Base runs across the MLX and CUDA hosts with uniform F16. Native profiles and
serving metrics expose macOS process physical footprint alongside RSS and allocator
usage; see the [measurement semantics](docs/milestone-5-profiling.md#process-physical-footprint).
The [footprint fit policy](docs/milestone-5-profiling.md#mlx-footprint-policy) is the
MLX evaluation default. Qualified 4B workloads, their reports and caveats are listed on
[the status page](docs/STATUS.md).

Milestone 6 adds persistent text/chat serving with bounded concurrent requests and
native prefill/decode scheduling; see [usage and limits](docs/milestone-6.md).
Configurable sampling, bounded log probabilities, opt-in chunked prefill and decode
batching are implemented. Bounded weight conversion reduces MLX load admission.
See the [M6 qualification report](docs/validation/milestone-6.md)
and [focused validation sweep](docs/validation/milestone-6-sweep.md), including the observed
batch-dependent seeded-sampling limit. ROCm and additional stages are Milestone 7 in
[the full project specification](SPEC.md).

## Development setup

Install `uv`, then let it provision Python 3.12 and the locked dependencies:

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

[GitHub Actions](.github/workflows/ci.yml) defines these Python checks plus Linux CPU
and AddressSanitizer/UBSan build-and-CTest jobs using matching, pinned native
gRPC/Protobuf dependencies; pushes to `codex/**` branches only warm that dependency
cache for stacked pull requests.
Accelerator and full-checkpoint runs remain explicit hardware qualification tasks.
See [validation tools and reproduction](scripts/validation/README.md), including the
[two-host launchers](scripts/validation/README.md#two-host-launchers).

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
without loading its tensor payloads. Qwen3-4B-Base is the benchmark target:

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
effective directional throughput are point-in-time observations from the test pair.
For measured selection, use the [Milestone 5 placement workflow](docs/milestone-5-placement.md)
with a compatible profile bundle and the measured objective settings. Two-stage prompts
longer than one boundary message (1638 tokens for Qwen3-4B at F16) need a chunked
workload; see [long prompts and chunked prefill](docs/milestone-5-placement.md#long-prompts-and-chunked-prefill).

See [the model/backend extension boundaries](docs/model-extensibility.md),
[Qwen3 support and validation](docs/qwen3.md), and
[the Milestone 0 implementation notes](docs/milestone-0.md). The
[validation report](docs/validation/milestone-0.md) records the pinned real-model and
cross-platform checks.

Historical code quality reviews and merged plans are in [docs/archive](docs/archive/README.md).
Tiny-model CPU pipeline validation is documented in the Milestone 1 notes.

For GPU-independent validation, run the guarded [CPU reload rehearsal](scripts/validation/README.md#cpu-rehearsal-while-the-gpu-is-busy).
The [concurrency-two qualification design](docs/concurrency-qualification.md) defines
the overlap and combined-memory evidence record; a record qualifies only the workload,
caps and binaries it was measured with.

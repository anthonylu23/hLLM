# Milestone 1 — native CPU pipeline

Milestone 1 implements executable dense Llama and Qwen3 stages, native two-worker
prefill and greedy decode, a Python deployment controller, streamed token events,
and request cleanup. The local numerical and process integration suites establish the
exit criterion for tiny deterministic models. Full-checkpoint performance is not claimed.

## Architecture and ownership

The [model extensibility boundaries](model-extensibility.md) guide future model support.
`runtime::StageBackend` owns opaque sequence state and accepts owned host activation
staging data. This does not expose device tensors or cache layouts to the common runtime.
The CPU implementation selects explicit `llama.v1` or `qwen3.v1` semantics; the controller
and transport contain no model-family branches. `forward` is shared by prefill and decode;
the execution service validates their distinct phase, shape and position requirements.

The CPU loader validates architecture revision 1, required features, dense semantics,
partition ownership, required tensor shapes, and source metadata before reading payloads.
Unknown revisions/features fail closed. Only assigned tensors are loaded. F16, BF16, and
F32 checkpoint payloads convert to resident F32; tied embeddings share storage in an
unsplit stage and are explicitly duplicated in a two-stage plan. Files must resolve inside
the configured model root. Model files are trusted local deployment inputs and must remain
immutable while loading. Manifest digests identify artifacts; they are not signatures.

Execution capabilities now advertise F32 computation only. FP16 is the activation wire
format; caches remain F32. Existing MLX/CUDA profiles are configured planning targets,
not executable backends. Python accepts one-stage plans for the unsplit reference; the
placement planner continues to enumerate two-stage plans.

## Execution and lifecycle

1. `DeploymentSession` loads the downstream worker, then stage zero; failed setup rolls
   back stages already loaded by the session.
2. Python sends prompt token IDs once through `Generation.Generate` and consumes events.
3. Stage zero reserves local state and opens one native `StageExecution.Execute` stream
   to the final worker for the request. That stream remains open across all decode steps.
4. Stage zero embeds and executes its layers, sends dense little-endian FP16 activations,
   receives the final worker's greedy token, and emits it to Python. Native workers drive
   the loop without Python scheduling or activation relay.
5. Both stages release sequence state on completion. Stage zero waits for downstream
   cleanup acknowledgment before emitting usage and the completed terminal event.

One request and one sequence are active per worker. The final stage samples the first
maximum on ties. Stop IDs and maximum output length terminate generation natively. The
controller defaults stop IDs to the manifest EOS IDs; callers can override them.
The CLI streams integer token IDs. Text tokenization/detokenization and the public
OpenAI-compatible HTTP API remain serving-layer work; this milestone's interface accepts
already-tokenized prompts.

RPC identity, deployment version, sequence number, phase, token position, tensor shape,
cache slot, payload size and finite values are checked. Empty prompts, zero output budgets,
invalid token IDs, conflicting reservations, and unsupported encodings fail explicitly.
Checksums are not negotiated in this version; a nonempty checksum field is rejected.
Messages are capped at 16 MiB and activation payloads at 8 MiB. Large prefills are rejected;
chunked prefill and continuous batching are later work.

A request defaults to a 60-second deadline; explicit deadlines must be within one hour
and are bounded by the RPC deadline. A watchdog interrupts blocking stream operations
when the client disconnects, control cancellation arrives, or the deadline expires.
CPU compute checks cancellation between layers. A reservation reaper releases expired
unclaimed requests. Cancellation and cleanup are idempotent. An executing request keeps
its stage and cache alive until execution exits, so cancellation cannot race unloading.
Exact load retries preserve state; changed loads require an explicit unload first.

Failures surface as non-OK gRPC status. Completion produces a terminal event. In
particular, cancellation may interrupt the RPC before a terminal event can be delivered;
the controller treats that as failure/cancellation and issues best-effort cleanup to all
stages. A surviving worker releases request state when the peer stream fails. No emitted
output is silently replayed. `DeploymentSession` is intended for exclusive ownership of
its loaded deployment and unloads it when the session closes.

## Memory admission and reporting

Before loading, validate float32 resident weight bytes plus the largest source conversion
buffer and a fixed allowance against the configured host budget. Before reserving a
request, validate resident weights plus exact float32 KV storage and a conservative dense
workspace/transport allowance. Cache allocation succeeds before reservation is accepted.
Allocation errors reject the operation without publishing partially loaded state.

`MemoryReport` adds loaded weight bytes, reserved cache bytes, reserved workspace bytes,
and active request count. These are runtime-owned payload accounting and conservative
reservations, not measured RSS or a hard process memory limit. Protobuf metadata, gRPC
internals and allocator overhead require operating-system headroom. Use conservative
worker budgets; measured backend workspace and peak-memory profiling remain Milestone 5.

## Local demo

Build and write the synthetic checkpoint:

```bash
uv run cmake --preset dev
uv run cmake --build --preset dev
uv run python scripts/create_cpu_demo.py build/cpu-demo
uv run hllm prepare build/cpu-demo --output build/cpu-demo.manifest.json
uv run hllm plan \
  --manifest build/cpu-demo.manifest.json \
  --workers examples/profiles/workers-cpu-local.yaml \
  --links examples/profiles/links-cpu-local.yaml \
  --workload examples/workloads/cpu-demo.yaml \
  --settings examples/profiles/planner-cpu.yaml \
  --output build/cpu-demo.plan.json --report build/cpu-demo.report.json
```

Start each worker in its own terminal:

```bash
build/native/dev/cpp/hllm-worker-cpu --listen 127.0.0.1:50051 \
  --worker-id cpu-a --model-root build/cpu-demo --memory-limit-bytes 536870912
```

```bash
build/native/dev/cpp/hllm-worker-cpu --listen 127.0.0.1:50052 \
  --worker-id cpu-b --model-root build/cpu-demo --memory-limit-bytes 536870912
```

Run the controller:

```bash
uv run hllm generate --manifest build/cpu-demo.manifest.json \
  --plan build/cpu-demo.plan.json --workers examples/profiles/workers-cpu-local.yaml \
  --token-ids 1,4,2 --max-new-tokens 32
```

These tiny synthetic weights exercise the pipeline and do not produce meaningful prose.
The demo binds loopback and uses insecure gRPC locally. Private cross-machine deployment,
Tailscale qualification and deployment credentials remain integration work for Milestone 4.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run python scripts/generate_proto.py --check
uv run ctest --preset dev
```

CTest includes the real-process integration suite. It can also be run directly with
`uv run pytest tests/integration -q`; set `HLLM_CPU_WORKER` to test another native build.
The integration suite fails with a build instruction if the worker binary is missing.

- Loaded F32 stages match the independent Transformers Qwen3 layer oracle at `2e-5`
  absolute tolerance for prefill and incremental decode.
- FP16 boundary activations and resulting Qwen3 hidden states match the unsplit oracle
  at `1e-3` absolute tolerance. This tolerance applies to the tiny fixture, not arbitrary
  full-size checkpoints or later GPU backends.
- Two real worker processes match unsplit greedy tokens at every valid split and both
  worker orders for tiny four-layer Llama and two-layer Qwen3 models.
- F16/BF16 storage conversion, 256 generated tokens, stop IDs, repeated requests,
  memory rejection, partial-load rollback, client cancellation, worker loss, malformed
  messages, idle-stream deadlines and control cancellation are covered.
- The actual prepare/plan/generate CLI flow is tested against running native workers.

The historical Apple ASan startup issue remains a separate toolchain limitation; do not
infer ASan coverage from normal tests. See the validation record below for checks run.

## Validation record — 2026-09-05

Passed 35 Python unit tests, 37 native tests, and 13 real-process integration tests.
Ruff, Pyright (including the integration suite), generated-protobuf reproducibility,
and diff whitespace checks passed. The demo checkpoint preparation and placement flow
also ran successfully. Downstream resource-exhaustion status reaches the client unchanged,
and its regression test verifies that the driver releases its reservation.

A separate UndefinedBehaviorSanitizer build passed the native and process integration
suites with `halt_on_error=1`. Reproduce it with:

```bash
uv run cmake --preset dev -B build/native/ubsan \
  -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined"
uv run cmake --build build/native/ubsan
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  uv run ctest --test-dir build/native/ubsan --output-on-failure
```

This establishes UBSan coverage for this local suite. ASan and full-checkpoint inference
remain unverified.

## Next steps

Milestone 2 is a CUDA backend implementing the same stage and sequence contract. Start
with tiny-model parity against the CPU golden suite, including FP16 boundaries and
lifecycle tests; add CUDA buffers, streams, memory reporting and sampling before running
a full checkpoint. Milestone 3 adds MLX, Milestone 4 qualifies cross-machine execution,
and Milestone 5 measures memory and placement performance. Choose the next model family
with the user before expanding state, partition or modality contracts.

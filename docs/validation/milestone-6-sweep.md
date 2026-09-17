# Milestone 6 focused validation sweep — 2026-09-16

Status: hardware matrix and sustained resource checks completed, with the seeded
sampling limitation below. This is a pre-PR correctness and resource-lifecycle sweep
of Qwen3-0.6B. Real 4B execution, M5 WAN acceptance, and controlled performance claims
remain separate follow-ups.

[Machine-readable results](milestone-6-sweep-results.json) preserve workload settings,
binary identities, artifact hashes, passing runs, failed sampled diagnostics, and memory
counters. The ordinary worker binaries match the final binaries from the earlier M6
sampled smoke; the new build change affects sanitizer configurations only.

## Scope

- Both MLX→CUDA and CUDA→MLX, split at layer 14, F16 execution and boundaries.
- Independently enable prefill chunks of 128 and ready-work decode batches up to 4;
  retain whole-prompt / serial execution as the reference configuration.
- Concurrency 1 and 4, mixed prompt lengths, 64 output tokens, a queued extra arrival,
  cancellation, recovery, and unload checks.
- At least 15 minutes of greedy traffic with three alternative log probabilities
  at concurrency 4 per stage order. The original positive-temperature sampled soak
  failed its strict reproduction check; the adjustment and diagnostics are recorded below.
- Process RSS, CUDA process memory, native allocator/reservation counters, and Mac
  memory pressure sampled throughout. Hardware workloads run sequentially.

Raw evidence lives under ignored `build/m6-sweep-20260916/`; the explicit workload
is recorded in `plan.json`. Native transport uses the existing SSH forwards over
Tailscale, with no firewall changes.

## Hardware results

All eight combinations of stage order, chunk size (0/128), and decode batch limit
(1/4) passed at concurrency 1 and 4: **160 requests**, including cancelled prefixes,
matched the corresponding whole-prompt, serial greedy reference. Every batching-enabled
concurrency-4 run dispatched actual multi-request batches on both workers (maximum
observed batch size 3).

The two adjusted long runs used chunk size 128, batch limit 4, greedy decoding, and
three alternative log probabilities:

| Stage order | Time at concurrency 4 | Concurrent rounds | Total requests | Intentional cancellations |
| --- | ---: | ---: | ---: | ---: |
| MLX → CUDA | 905.95 s | 116 | 590 | 118 |
| CUDA → MLX | 905.73 s | 106 | 540 | 108 |

Totals include each run's concurrency-1 reference/recovery work. Together with the
matrix, these are **1,290 requests and 258 intentional cancellations**. All token,
finite/bounded probability-metadata, retirement, recovery, and unload checks passed.
The two unbatched sampled diagnostic runs passed a separate 40 requests. Failed batched
sampled runs are excluded from those passing totals.

Across the matrix and long runs, 20 deployment load/unload cycles retired all logical
model and request reservations. Every concurrency-4 round returned cache/workspace
reservations to zero. These are repeated mixed-length cycles with drain checks between
rounds, not a claim of uninterrupted saturation or full-context qualification.

## Memory observations

MLX's post-round allocator-active value stayed constant through all 116 rounds of the
first long run and all 106 rounds of the second. CUDA increased by 8,519,680 bytes
(8.125 MiB) early in the first run, then stayed at the same value for the final **104
rounds**. It was constant throughout the second run. The bounded observation windows
show a plateau, not an indefinite leak-free guarantee.

After both long runs unloaded, MLX allocator-active bytes were zero. CUDA still reported
42,598,400 allocator-active bytes plus cached memory, while logical model/cache/workspace
reservations were zero. These retained allocator counters are reported separately from
request ownership.

| Long run | Maximum sampled MLX RSS | Maximum sampled CUDA RSS | Maximum sampled CUDA process GPU memory |
| --- | ---: | ---: | ---: |
| MLX → CUDA | 1.042 GiB | 1.866 GiB | 1.016 GiB |
| CUDA → MLX | 1.047 GiB | 2.003 GiB | 1.016 GiB |

Half-second sampling can miss load transients. Linux's process-lifetime RSS high-water
counter reached **2.241 GiB** during the shared long-run worker lifetime. RSS, GPU usage,
and allocator counters measure different things and must not be added together. Raw
samples and later-window RSS statistics are retained in the result artifact.

Mac pressure stayed normal in every recorded sample, with **zero new swap-out pages**
through each monitored hardware phase. The stop guard never fired. Existing swap was
not cleared, and unrelated applications/workloads were left running.

## Added regression coverage

- A cancelled batch member remains owned until the dispatch finishes; peers finish
  normally, and an injected batch failure releases all waiters and permits later work.
- A queued native deadline expires without entering the backend.
- A nonzero-position prefill chunk receives its turn after the bounded decode streak.
- Real HTTP shutdown with active and queued requests, worker loss during generation,
  queued client disconnect, fail-closed admission, and surviving-stage cleanup.
- A tensor truncated after metadata inspection fails on the later conversion chunk.

The serving soak harness now accepts `--minimum-duration-seconds`, applied at the
highest requested concurrency, while preserving the minimum number of rounds.

## Findings and fixes

### Seeded sampling depends on batch composition

The original sampled soak stopped on a token mismatch against its sequential reference;
it is preserved as a **failed run**, not counted as a successful long soak. Strict
repeats used temperature 0.8, top-p 0.95, top-k 40, seed 42, and 64 output tokens.
Both stage orders passed at concurrency 1 and 4 with serial dispatch. Both failed the
exact-token check when dynamic batching was enabled.

At the first MLX→CUDA mismatch (output index 12, zero-based), the token prefix was
identical, while top-token log probabilities shifted by about 0.00271. CUDA→MLX first
diverged at index 13: tokens 25 and 911 had equal reference log probability, but batching
made token 911 higher by 0.015625. These observations are consistent with numerical
changes affecting the seeded categorical draw. They do not establish bitwise sampling
reproducibility under changing batch shapes.

No production sampling policy was changed. Batch execution remains opt-in; documentation
now states the observed limit and directs seeded comparisons to serial dispatch. The
long combined-mode memory/cleanup runs use greedy decoding with probability metadata
and retain strict token comparisons. They do **not** qualify 15 minutes of non-greedy
batched execution. The sampling-drift result remains a PR-visible limitation.

The harness now retains failing token traces and bounded probability metadata to make
these failures diagnosable.

### Sanitizer build and dependency coverage

The Linux sanitizer attempt identified missing instrumentation on the generated
Protobuf target. The runtime and its generated message code now receive consistent
ASan/UBSan flags when `HLLM_ENABLE_SANITIZERS=ON`; ordinary builds are unchanged.
The initial debug configuration also encountered an Abseil dependency link mismatch;
the validation build uses `-O1 -g -DNDEBUG` with sanitizers enabled.

After the generated-code fix, a separate container-annotation failure remained in
prebuilt Protobuf parsing. A standalone 400-element repeated-field serialize/parse
program reproduces it without hLLM runtime code. This is a dependency/instrumentation
limitation; the failed logs and reproducer are preserved with the raw evidence.

Sanitizer coverage is deliberately reported in two groups:

- **32 targeted native tests passed** with ASan, UBSan, leak detection, and container
  annotations enabled, covering scheduler lifetime/failure, sampling, dense loading,
  CPU stage behavior, and admission accounting.
- **72 native tests and 15 HTTP/chunking/harness cases passed** with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:detect_container_overflow=0` and
  `UBSAN_OPTIONS=halt_on_error=1`. Container-annotation coverage is excluded for this
  group; ordinary address, lifetime, leak, and undefined-behavior checks remain enabled.

The non-sanitized focused Python suite passed **135 tests**; both serving-soak harness
cases passed again after failed-trace diagnostics were added. Both accelerator numerical
CTest entries passed. Another **8 MLX and 8 pageable CUDA boundary-fault cases** passed,
covering control cancellation and deadlines in prefill/decode in both mixed stage orders.
Ruff, Pyright, generated protobuf checks, and diff checks passed.

## Reproduction

Use the worker setup and model assets from [M6 serving instructions](../milestone-6.md).
For the matrix, vary native `--max-decode-batch 1/4`, controller
`--prefill-chunk-tokens 0/128`, and both stage orders with `--rounds 2 --levels 1 4`.

For each adjusted long run, start both workers with `--max-active-requests 4`,
`--max-cached-tokens 4096`, and `--max-decode-batch 4`. Write
`{"temperature": 0, "logprobs": 3}` to a sampling JSON file, then run:

```bash
uv run python -m scripts.validation.serving_soak \
  build/model.manifest.json /models/Qwen3-0.6B build/sweep-mlx-cuda.json \
  --workers mlx cuda --mlx-endpoint HOST:PORT --cuda-endpoint HOST:PORT \
  --rounds 2 --output-tokens 64 --levels 1 4 --prefill-chunk-tokens 128 \
  --minimum-duration-seconds 900 --sampling-json build/greedy-probabilities.json
```

Reverse `--workers` and use a new output path for the other order. Physical samplers
must cover each worker independently. The raw plan preserves the original sampled
workload and the explicit reason for adjusting the sustained phase.

## Remaining limits before wider qualification

- Dynamic batch composition changes seeded sampled outputs. Exact reproduction across
  batch shapes is not established; keep this visible in the PR and serving documentation.
- Full container-annotation coverage needs a consistently instrumented Protobuf dependency
  stack. The broader sanitizer results above exclude that annotation check.
- These short, bounded-context workloads do not establish full-context memory fit,
  concurrency 8, pinned CUDA batching, sustained non-greedy batching, indefinite memory
  behavior, WAN placement acceptance, or a performance speedup.

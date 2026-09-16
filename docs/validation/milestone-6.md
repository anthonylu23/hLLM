# Milestone 6 follow-up qualification — 2026-09-15

The [2026-09-16 focused sweep](milestone-6-sweep.md) adds sustained memory/cleanup
coverage and records seeded-output drift with dynamic batching on longer sampled requests.

The full Qwen3-0.6B checkpoint completed the mixed-length HTTP soak in both MLX/CUDA
stage orders at concurrency 1, 2 and 4. Chunked prefill plus ready-work decode batching
also completed both orders at concurrency 1 and 4. All recorded tokens, including
cancelled prefixes, matched the corresponding original single-request greedy baseline.

[Machine-readable results](milestone-6-results.json) preserve raw-artifact hashes,
worker binary identities, timings, native batch observations and memory results.
Raw reports remain under ignored `build/m6-baseline/`. These are explicit feasible
placements, not measured-planner acceptance or a controlled speedup benchmark.

## Workload and transport

- Pinned `Qwen/Qwen3-0.6B`, split at layer 14 of 28, F16 execution and boundaries.
- Two prompts: a short completion and a repeated explanatory prompt. Exact text is
  retained in the result artifact. Every completed request generates 64 tokens.
- Five rounds per concurrency level, each with `concurrency + 1` arrivals, staggered
  by 10 ms. The first client disconnects after two nonempty fragments; each recorded
  cancellation consumed exactly two native tokens.
- Single-request baselines precede each level; cleanup is checked after every round,
  followed by recovery and final unload. The baseline includes 118 HTTP requests;
  the chunking/batching runs include another 82.
- Native gRPC uses bidirectional SSH forwards over Tailscale. This is a distinct
  transport observation from direct worker ports or forced-DERP testing. No firewall
  rules or existing workloads were changed.
- Initial workers used a 2.734375 GiB MLX admission cap and 2 GiB CUDA host/device
  caps. Four-request admission counts combined reservations. Physical availability
  and competing desktop/Java workloads were checked before running.

The first concurrency-4 attempt exposed repeated ASGI cancellation interrupting native
retirement. The fix shields retirement inside the native generator itself, and checks
failed cleanup again after queued admission. A focused regression reproduces level
cancellation; both stage orders then passed the full soak. The failed artifact is
preserved as `build/m6-baseline/mlx-cuda-before-cancel-fix.json`.

## Observed latency

Medians below cover completed requests, including mixed lengths and queueing. The request
mix differs across concurrency levels. Other applications remained active, some validation
work overlapped, and network conditions were uncontrolled. These numbers do not isolate
an optimization's causal effect or establish a sustained-throughput improvement.

| Stage order | Concurrency | Baseline TTFT / ITL | Chunk 128 + batch limit 4 TTFT / ITL |
| --- | ---: | ---: | ---: |
| MLX → CUDA | 1 | 0.873 s / 0.097 s | 0.601 s / 0.093 s |
| MLX → CUDA | 2 | 0.465 s / 0.095 s | — |
| MLX → CUDA | 4 | 0.230 s / 0.090 s | 0.554 s / 0.086 s |
| CUDA → MLX | 1 | 0.681 s / 0.088 s | 0.730 s / 0.095 s |
| CUDA → MLX | 2 | 0.446 s / 0.086 s | — |
| CUDA → MLX | 4 | 0.494 s / 0.103 s | 0.588 s / 0.094 s |

TTFT is measured from HTTP submission to the first consumed native token; first-text
latency is recorded separately. `before_native_seconds` includes HTTP, tokenization and
queue time; it is not a pure queue-time measurement. Native scheduler wait is recorded
separately. Raw reports retain per-request intervals and total durations.

At concurrency 4, MLX→CUDA dispatched 49 MLX and 17 CUDA decode batches;
CUDA→MLX dispatched 41 CUDA and 152 MLX batches. Actual batch sizes, rather than only
configured limits, are retained in the artifact. No collection delay was introduced.
The latency observations justify keeping both optimizations opt-in.

## Memory and loading

Every baseline round returned cache/workspace reservations to zero and allocator active
bytes to the same per-level value. Model unload retired logical model state; MLX allocator
active bytes returned to zero. CUDA retained framework/cache allocations, recorded
separately from model/request ownership.

Physical sampling covered both complete baseline run windows. Maximum sampled RSS was
1.501 GiB for MLX and 2.227 GiB for CUDA across those windows; CUDA process-GPU usage
peaked at 0.988 GiB. These are separate counters and must not be added together. Half-second
sampling can miss transients. Chunking/batching runs include allocator and reservation
observations; their physical RSS was not independently sampled.

The bounded loader separately completed two full-checkpoint **MLX first-stage** cycles
with 128 prompt / 64 output tokens and a 1.5 GiB cap. Lifetime RSS peak was
1,105,936,384 bytes (about 1.030 GiB), and allocator active bytes returned to zero after
both unloads. The old loader's conservative requirement for that same assignment was
about 2.73 GiB. This comparison establishes reduced load-admission requirements; the
workload differs from historical M5 memory probes, so it is not a controlled RSS ratio.

A separate full-model MLX F16 probe under a 2 GiB cap passed all nine independent F32
reference decisions and the declared layer/logit error bounds after conversion changed.
The new F16 4B endpoint-only estimate is about 1.453 GiB, excluding transformer weights,
KV and inference workspace. No 4B payload was loaded; actual 4B fit remains unqualified.

## Regression and limits

CPU/MLX/CUDA tests cover seeded sampling, bounded probability metadata, independent
sequence positions in batches, chunk sizes 1/3/8, queue fairness, cancellation, HTTP
streaming, overload, deadlines, cleanup and loader corruption. The tiny release-build
harness test uses simultaneous arrivals and 2 ms observation intervals so short requests
actually overlap; full-model runs retain the documented 10 ms / 100 ms settings.

Final regression results: 120 Python tests plus both serving-soak harness cases passed;
79 CPU/MLX and 82 CPU/CUDA CTest entries passed. The local CPU integration entry was
rerun after correcting the tiny workload's observation timing. Ruff, Pyright, generated
protobuf checks and diff whitespace checks passed. Local Apple ASan initialization still
hangs before `main`; no local sanitizer coverage is claimed.

The long greedy soaks above were captured before the probability-metadata extension;
their exact worker binary hashes are retained. A final full-checkpoint sampled smoke
then passed on the final native binaries in **both stage orders**, with concurrency 1
and 4, chunk size 128 and decode batch limit 4. It used temperature 0.8, top-p 0.95,
top-k 40, seed 42 and three alternative log probabilities. Each completed request
generated 16 tokens; one round per level produced 26 requests total, including four
cancelled prefixes. Completed requests received finite, bounded probability metadata
for every token. Tokens matched the corresponding same-configuration sequential
baseline, and cleanup/unload checks passed. This short run does not replace the longer
greedy soak or qualify new binary-bound placement profiles.

Still unqualified: indefinite memory plateaus, controlled WAN throughput, concurrency 8,
pinned CUDA batching, full model context, real 4B execution and M5's exhaustive within-15%
placement comparison. See [usage and next steps](../milestone-6.md).

## Reproduction

Build current workers and start them with the documented model roots, feasible memory
budgets and `--max-active-requests 4 --max-cached-tokens 4096`. Supply endpoints reachable
from the controller and both native stages. Sample each worker's process separately with
`scripts/validation/memory_watch.py` if physical coverage is required.

```bash
uv run python -m scripts.validation.serving_soak \
  build/model.manifest.json /models/Qwen3-0.6B build/soak-mlx-cuda.json \
  --workers mlx cuda --mlx-endpoint HOST:PORT --cuda-endpoint HOST:PORT \
  --rounds 5 --output-tokens 64 --levels 1 2 4
```

Reverse `--workers` for the other order. For the optimization run, start native workers
with `--max-decode-batch 4` and add `--prefill-chunk-tokens 128 --levels 1 4`. Use new
output paths to preserve earlier evidence. Keep greedy settings fixed when comparing
execution configurations.

For the sampled smoke, write the sampling fields above as a JSON object and pass
`--sampling-json path/to/sampling.json --rounds 1 --output-tokens 16`, alongside the
chunking/batching settings. Use a separate output path for each stage order.

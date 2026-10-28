# Qwen3-4B concurrency two: October 20, 2026 re-run and HTTP serving soak

Re-run of P5 on the corrected MLX worker, the open item from the
[October 13 record](qwen3-4b-concurrency-20261013.md), followed by the bounded HTTP
serving soak at levels 1 and 2. It is also the first run driven end to end by the
tracked [two-host launchers](../../scripts/validation/README.md#two-host-launchers).
Run directory `hllm-4b-fit-20261020-042423` on both hosts (ignored, not committed).
The user re-approved the 5.125 GiB MLX cap for this run on October 19 and freed Mac
memory before the MLX profile. Times are UTC.

## Result

| Item | Value |
| --- | --- |
| Record | `concurrent.json`, schema 1.1, evidence digest `76ebd2240984ac3a4e08484c265392490ebf92977cab73abee5eb89a0ec02043` |
| Acceptance | `accepted`, no reasons, first attempt |
| Window | 04:59:46Z–05:11:01Z |
| Workload | 128 prompt / 256 output tokens, 384-token reservation per request, uniform F16, forward split (MLX layers 0–14, CUDA layers 15–35), plan `1ed77abe…`, reference `329ea04b…` (the October 10 oracle, inputs byte-identical to the October 11 run) |
| Procedure | 3 cycles × (fresh fit gate on both workers, load, 1 serial request, 3 paired rounds, 1 cancellation round, unload); 15 rounds, 27 requests |
| Exactness | 27/27 exact against the reference |
| Overlap | 24 rows, proven on both workers from each worker's own retirement events; peak concurrent requests 2 on both |
| Retirement reasons | each cycle's victim retired `cancelled` on both workers while the survivor held its reservation; the other 24 requests retired `completed`. This is the first P5 record with the corrected reasons (PR #30) |
| Caps | MLX unified 5.125 GiB (5,502,926,848 B), CUDA host 3.75 GiB / device 6.125 GiB; `--max-active-requests 2 --max-cached-tokens 768 --request-observations on` on both, decode batch 1 |
| Logical peaks | within admission on both. MLX: weights 3,805,836,800 B + cache 47,185,920 B + workspace 1,390,473,216 B = 5,243,495,936 B |
| Guards | both `normal`, bound to the worker PIDs, 331 (MLX) and 341 (CUDA) observations, no stop |
| Telemetry | MLX allocator sample in all 3,550 snapshots (none missing); CUDA 21,377 snapshots and 1,285 device samples bound to the worker's reported PID |
| Cleanup | verified on both workers after every cycle; no GPU compute process or hLLM worker left on either host |
| Source | `main` at `6f70bd4`, source digest `276254ea…`; MLX worker `57fd5bd9…`, CUDA worker `aae504e5…`, rebuilt for this run |

Per-cycle gates and physical envelopes (all `safe`):

| Cycle | Gate time | MLX available | MLX envelope (footprint-max) | CUDA host avail / envelope | CUDA device avail / envelope |
| --- | --- | --- | --- | --- | --- |
| 0 | 04:59:46Z | 8.90 GiB | 5435 MiB (peak 4708) | 8.40 GiB / 3311 MiB | 7.32 GiB / 5932 MiB (peak 5160) |
| 1 | 05:03:27Z | 9.49 GiB | 5436 MiB (peak 4709) | 7.33 GiB / 4210 MiB | 7.29 GiB / 5932 MiB |
| 2 | 05:07:09Z | 9.06 GiB | 5436 MiB (peak 4709) | 7.40 GiB / 4232 MiB | 7.31 GiB / 5932 MiB |

The physical peaks match October 13: the MLX footprint within 7 MiB (4708–4709 MiB
against 4708–4715 MiB) and the CUDA device peak exactly, so the allocator-read correction in PR #27 did not change the memory
behavior the October 13 record measured. The October 13 caveat is resolved.

Preconditions bound into the record:

- **Baseline:** `baseline-reload-soak.json`, 3 cycles, 6/6 exact, 04:53Z–04:59Z, same
  plan, reference and worker binaries.
- **P1 profiles (fresh, same binaries):** CUDA `11ddfebc…` measured 04:34:28Z (safe,
  `cuda-rss-device-v1`, device envelope 5.72 GiB); MLX `737875f9…` measured 04:51:38Z at
  the 5.125 GiB cap (safe, `mlx-footprint-max-v1`, envelope 5.28 GiB, preflight sampled
  7.08 GiB before the checkpoint read).

Results file: [`qwen3-4b-concurrency-20261020-results.json`](qwen3-4b-concurrency-20261020-results.json),
written by `assemble_two_host_results.py` at the end of `run.sh concurrent`.

## HTTP serving soak (P5 follow-up)

`serving_soak.py` against freshly started guarded workers on the same binaries and
profiles, bound to the same plan, with the fit gate checked before each level:
05:13:46Z–05:24:59Z, levels 1 and 2, 5 rounds each, greedy decoding, 254 output tokens.
Each level first serves one request per prompt as its token reference, then runs 5
rounds of level + 1 staggered requests (one waits in the server's queue, one is cancelled
after its first tokens), then one recovery request; the harness fails a level on any
token difference from its reference.

| Level | Requests | Completed | Cancelled | Peak active (MLX, CUDA) | First token, median / max |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 13 | 8 | 5 | 1, 1 | 1.00 s / 1.29 s |
| 2 | 18 | 13 | 5 | 2, 2 | 1.54 s / 2.11 s |

Every completed request for the same prompt produced the identical 254 tokens at both
levels, and every cancelled request's tokens are a prefix of them. Guards stayed normal:
minimum available 5.88 GiB on Linux and 4.90 GiB on the Mac.

The first attempt was refused by the soak's gate before any model load. The longer
built-in prompt is 130 tokens, so `--output-tokens 256` reserves 386 tokens per
request, above the profiled 384. It is preserved in the run directory, and the
accepted attempt used 254 output tokens to fill the profiled reservation exactly.

Summary: [`qwen3-4b-serving-20261020-summary.json`](qwen3-4b-serving-20261020-summary.json)
(levels, counts and the SHA-256 of the full report, which stays in the run directory).

## What this does not claim

No throughput or latency claim (the first-token times above are observations of this
run, not a benchmark), nothing above concurrency two, no workload other than 128/256
(the serving soak's built-in prompts are 5 and 130 tokens), no cap other than the ones
above, and nothing about the reverse order. The workers do not include the shared
stage workspace from [PR #35](https://github.com/anthonylu23/hLLM/pull/35); a run at
the original 4.375 GiB MLX cap needs that change and fresh profiles.

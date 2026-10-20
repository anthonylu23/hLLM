# Qwen3-4B concurrency two at the 4.375 GiB MLX cap: shared stage workspace, October 20, 2026

Qualification of the shared stage workspace (this PR, head `c27bd8a`) on both hosts,
run straight after the [P5 re-run on `main`](qwen3-4b-concurrency-20261020.md) with
the same plan, reference and workload. The only intended difference is the admission
rule, so the MLX cap goes back to the original P1 value of 4.375 GiB, at which the
October 13 attempt 1 was refused for the second request. Run directory
`hllm-4b-fit-20261020-053205` on both hosts (ignored, not committed). Times are UTC.

## Result

| Item | Value |
| --- | --- |
| Record | `concurrent.json`, schema 1.1, evidence digest `9c19cf2122a5b79673f99aac05f7388b82eceeee2dd14d983a346ce81b06990f` |
| Acceptance | `accepted`, no reasons, first attempt |
| Window | 05:50:04Z–06:01:43Z |
| Workload | 128 prompt / 256 output tokens, 384-token reservation per request, uniform F16, MLX layers 0–14 and CUDA layers 15–35, plan `1ed77abe…`, reference `329ea04b…` |
| Procedure | 3 cycles × (fresh fit gate, load, 1 serial, 3 paired rounds, 1 cancellation round, unload); 27 requests |
| Exactness | 27/27 exact |
| Overlap | 24 rows proven on both workers; peak concurrent requests 2 on both |
| Retirement | 3 victims `cancelled` and 24 `completed` on each worker |
| Caps | **MLX unified 4.375 GiB (4,697,620,480 B)**; CUDA host 3.75 GiB / device 6.125 GiB; 2 active requests, 768 cached tokens, decode batch 1 |
| Guards | both `normal`, 334 (MLX) and 342 (CUDA) observations |
| Telemetry | MLX allocator sample in all 3,617 snapshots; CUDA 21,521 snapshots and 1,290 device samples bound to the worker's PID |
| Source | digest `e67d8a13…` (PR head `c27bd8a`); MLX worker `e2f12db5…`, CUDA worker `e0d409bb…` |

Logical peaks, from the workers' own memory reports:

| Worker | Weights | Cache (2 requests) | Workspace (shared) | Combined | Cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX (unified) | 3,805,836,800 | 47,185,920 | 695,236,608 | **4,548,259,328** | 4,697,620,480 |
| CUDA (device) | 5,017,011,712 | 66,060,288 | 654,979,072 | 5,738,051,072 | 6,576,668,672 |

The MLX stage holds two requests with 142 MiB of its cap to spare. On the `main`
workers earlier the same morning its combined peak was 5,243,495,936 B (workspace
1,390,473,216 B, one per request), which needs the 5.125 GiB cap. The CUDA device peak
falls by one workspace too (6,393,030,144 B on `main`).

Physical envelopes per cycle (all `safe`; MLX policy `mlx-footprint-max-v1`, CUDA
`cuda-rss-device-v1`):

| Cycle | MLX available | MLX envelope (peak) | CUDA host avail / envelope | CUDA device avail / envelope (peak) |
| --- | --- | --- | --- | --- |
| 0 | 8.86 GiB | 5435 MiB (4708) | 8.38 GiB / 3311 MiB | 7.32 GiB / 5932 MiB (5160) |
| 1 | 8.80 GiB | 5436 MiB (4709) | 7.38 GiB / 4174 MiB | 7.31 GiB / 5932 MiB (5160) |
| 2 | 8.67 GiB | 5436 MiB (4709) | 7.33 GiB / 4198 MiB | 7.29 GiB / 5932 MiB (5160) |

The physical peaks equal the `main` run's to within 1 MiB: the change alters what
admission reserves, not what the workers allocate. The physical gate still decides
fit independently of the cap.

Preconditions bound into the record: a 3-cycle baseline, 6/6 exact (05:44Z–05:50Z),
and fresh profiles on the same binaries: MLX `7fd90cc9…` at 4.375 GiB, measured
05:32:32Z (safe, envelope 5.28 GiB, preflight sampled 8.68 GiB before the checkpoint
read), and CUDA `5a98fb0b…`, measured 05:42:41Z (safe, device envelope 5.72 GiB).

Results file: [`qwen3-4b-shared-workspace-20261020-results.json`](qwen3-4b-shared-workspace-20261020-results.json).

This record makes the same non-claims as the P5 records: no throughput figure, no
concurrency above two, no other workload or split. The MLX profiler binary is
unchanged by this PR (`5a4a51ae…`); the CUDA profiler changed only in how it labels
pinned staging, so both profiles were taken fresh on the binaries served.

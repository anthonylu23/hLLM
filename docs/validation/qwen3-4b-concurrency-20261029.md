# Qwen3-4B concurrency two at the 4.375 GiB MLX cap on `main`, October 29, 2026

P5 re-run on `main` at `b1ba4f6`, after PRs #34–#38 merged. These are the first
qualification binaries with the shared stage workspace (#35) and with transport buffers
reserved per request, the review fix that followed the
[October 20 shared-workspace run](qwen3-4b-shared-workspace-20261020.md). The plan,
reference, workload and caps are the October 20 ones, including the original P1 MLX
cap of 4.375 GiB. Run directory `hllm-4b-fit-20261028-030120` on both hosts (ignored,
not committed). Times are UTC.

## Result

| Item | Value |
| --- | --- |
| Record | `concurrent.json`, schema 1.1, evidence digest `12165dcec3a9296d76f8ba945a990aa5088a072ed09eaa9f9d9bbfff332b44b9` |
| Acceptance | `accepted`, no reasons, second attempt (the first was stopped by host memory pressure from another workload; [below](#attempt-1)) |
| Window | 02:53:19Z–03:04:57Z |
| Workload | 128 prompt / 256 output tokens, 384-token reservation per request, uniform F16, MLX layers 0–14 and CUDA layers 15–35, plan `1ed77abe…`, reference `329ea04b…` |
| Procedure | 3 cycles × (fresh fit gate, load, 1 serial, 3 paired rounds, 1 cancellation round, unload); 27 requests |
| Exactness | 27/27 exact |
| Overlap | 24 rows proven by retirement events, 12 on each worker; peak concurrent requests 2 on both |
| Retirement | 3 victims `cancelled` and 24 `completed` on each worker |
| Caps | **MLX unified 4.375 GiB (4,697,620,480 B)**; CUDA host 3.75 GiB / device 6.125 GiB; 2 active requests, 768 cached tokens, decode batch 1 |
| Guards | both `normal`, 341 (MLX) and 353 (CUDA) observations |
| Telemetry | MLX allocator sample in all 3,589 snapshots; CUDA 22,303 snapshots and 1,328 device samples bound to the worker's PID |
| Source | digest `6081e665…` (`main` at `b1ba4f6`); MLX worker `a00aacb5…`, CUDA worker `7597e412…` |

Logical peaks, from the workers' own memory reports:

| Worker | Weights | Cache (2 requests) | Workspace (shared) | Combined | Cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX (unified) | 3,805,836,800 | 59,000,832 | 695,236,608 | **4,560,074,240** | 4,697,620,480 |
| CUDA (device) | 5,017,011,712 | 66,060,288 | 654,979,072 | 5,738,051,072 | 6,576,668,672 |
| CUDA (host) | 0 | 11,814,912 | 10,831,872 | 22,646,784 | 4,026,531,840 |

The MLX stage holds two requests with 131 MiB of its cap to spare. Its combined
reservation is exactly the 4,560,074,240 B the October 20 report predicted for
per-request transport buffers. Each request's cache is its 23,592,960 B of KV plus
5,907,456 B of transport (`24 × 384 + 6 × 384 × 2560`). The CUDA stage reserves the
same transport in its host domain (11,814,912 B = 2 × 5,907,456), and its device
figures equal October 20's.

Physical envelopes per cycle (all `safe`; MLX policy `mlx-footprint-max-v1`, CUDA
`cuda-rss-device-v1`):

| Cycle | MLX available | MLX envelope (peak) | CUDA host avail / envelope | CUDA device avail / envelope (peak) |
| --- | --- | --- | --- | --- |
| 0 | 6.37 GiB | 5435 MiB (4709) | 8.08 GiB / 3311 MiB | 7.32 GiB / 5932 MiB (5160) |
| 1 | 8.01 GiB | 5436 MiB (4709) | 7.14 GiB / 4210 MiB | 7.29 GiB / 5932 MiB (5160) |
| 2 | 8.74 GiB | 5436 MiB (4709) | 7.16 GiB / 4233 MiB | 7.29 GiB / 5941 MiB (5168) |

The MLX physical peaks equal October 20's to within 1 MiB, and the CUDA device peak to
within 8 MiB. Reserving transport per request changes what admission reserves, not
what the workers allocate. The physical gate still decides fit independently of the
cap.

Preconditions bound into the record: a 3-cycle baseline, 6/6 exact
(02:35:48Z–02:41:54Z), and fresh profiles on the same binaries. MLX `bcdfbc05…` was
measured at 4.375 GiB at 02:33:10Z: safe, envelope 5.28 GiB, preflight sampled
7.10 GiB before the checkpoint read. CUDA `d80731c7…` was measured at 02:31:47Z: safe,
device envelope 5.72 GiB, host envelope 4.09 GiB. Profiler binaries: MLX `98888c8a…`,
CUDA `6e90da1d…`.

Results file: [`qwen3-4b-concurrency-20261029-results.json`](qwen3-4b-concurrency-20261029-results.json)
(redacted). It includes both attempts and the refused baselines.

## Attempt 1

02:42:24Z–02:51:08Z, same binaries, profiles and baseline, evidence digest
`717a026e…`, preserved as `concurrent-attempt-1/`. Cycles 0 and 1 completed with 18/18
exact and the same MLX reservation, 4,560,074,240 B. The cycle 2 serial request then
failed with `UNAVAILABLE` "Stream removed (Socket closed)".

The Linux resource guard had stopped the CUDA worker at 02:51:08.97Z, reporting "new
swap-out exceeds configured 60-second limit" (256 MiB). Another project on the Linux
host had just started a Rust `cargo test --workspace` with swap already 92% full.
Available memory fell from about 6.9 GiB to 4.3 GiB, and swap-out rose by 0.93 GiB in
14 s, 0.87 GiB of it in the last 2 s. That was not a worker defect: the guard acted as designed. After that
workload paused, attempt 2 ran on the same profiles and baseline.

The soak wrote `concurrent.json` at 02:51:08.40Z, before the guard's exit record. So
the record, and the assembled results file, list the CUDA guard as `normal`. The exit
event is in `concurrent-cuda.guard.jsonl`. See the follow-ups below.

## Refusals before the run

None of these loaded the model; they are preserved in the run directory with READMEs.

- **CUDA profile, twice at preflight.** The device preflight needs
  `cap × 1.1 + 512 MiB + 256 MiB` = 8,039,641,907 B available for the 6.125 GiB cap.
  The first attempt sampled 7,753,170,944 B (and then less, after an embedding model
  loaded on the GPU). The second sampled 7,788,822,528 B with a remote-desktop daemon
  (134 MiB) and a browser (42 MiB) on the GPU. With both closed, the accepted profile
  sampled 8,055,160,832 B. The accepted October 20 profiles had 8.06–8.08 GB, so the
  desktop session's GPU use is the binding constraint on this card.
- **Baseline, twice before load.** First, `run.sh` was given a relative settings path,
  so the copy to the Mac failed. Second, the soak's fresh-fit check refused both
  workers because `inputs/*-workers.json` still carried the October 20 binary digests.
  The assembled results file lists that second baseline with `all_exact: true` over
  zero requests.

## Follow-ups

- The concurrency soak should finish reading the guard records before it writes
  `concurrent.json`, and the results assembler should take guard exits from the guard
  files, as it already does for baselines. A guard-caused failure should not be
  recorded as `normal`.
- The assembler should not report `all_exact` for a baseline with no requests.
- Run setup should take the worker binary digests from the build instead of
  hand-edited `inputs/*-workers.json`, and `run.sh` should resolve its settings path.
- Host memory, not only the GPU, has to be quiet for the length of a run: other
  workloads' builds can push the Linux host into swap.

This record makes the same non-claims as the P5 records: no throughput figure, no
concurrency above two, no other workload or split.

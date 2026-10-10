# Qwen3-4B two-host serving qualification — October 10, 2026

Status: **the baseline serving-fit gate passed.** Fresh same-day isolated profiles on
both hosts were `safe`, and the six-cycle same-process reload soak completed 24 exact
256-token continuations and six exact cancellation prefixes in the same two worker
processes, with serving peaks inside the prelaunch budgets on both hosts. This
qualifies the 128-prompt/256-output, concurrency-one, uniform-F16 deployment of
Qwen3-4B-Base on this host pair under `footprint-v1` in **both** stage orders: MLX→CUDA
at split 15 and, in the P4 run below, CUDA→MLX at split 21. The P3 run below extends
the forward order to a 512-prompt/256-output workload with a 5.125 GiB MLX cap. It
does not qualify larger contexts than 768 cached tokens, concurrency two, an
indefinite reload plateau, or M5 WAN performance.

This run executes P0, P1 and P2 of the [overnight backlog](../overnight-backlog.md)
with the CUDA host free for the first time since September 28. The
[machine-readable result](qwen3-4b-serving-20261010-results.json) records hashes,
guard summaries, per-cycle outcomes and serving peaks. Raw evidence stays under the
ignored run directories named below. P4 and P3 followed the same night and are
reported below; P5 was not attempted.

## Scope and identity

All times below are UTC taken from the raw guard and soak records; the hosts run in
Central and Eastern time, and the first revision of this report mis-converted several
of them.

Both hosts ran `main` at commit `a2b62a9` (merge of PR #21) with a clean tree. The
Linux snapshot is `~/Projects/experiments/hllm-4b-fit-20261008-232736/source` on the
CUDA host; the Mac used its working tree. The shared source digest over tracked
native, protocol, Python, script and test files is
`abadd3fd2f634c217787942b77a295289c74e585ba977d2c39b556332b70b1bd`. Both hosts hashed
the pinned `Qwen/Qwen3-4B-Base` revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`
and all three shards matched the September 18 record; the saved independent
128-prompt/256-output F16 reference matched that checkpoint digest.

The CUDA worker and profilers were rebuilt from this commit on October 8 because the
previous Linux binaries predated the RSS lifetime-peak clamp in `77dabf9`. The Mac
MLX worker and profiler were rebuilt the same evening. New binary digests:

| Executable | SHA-256 prefix |
| --- | --- |
| `hllm-worker-cuda` | `4b177c767c99` |
| `hllm-profile-memory-cuda` | `f17a52275659` |
| `hllm-worker-mlx` | `d49f4ee12588` |
| `hllm-profile-memory-mlx` | `8ae5d0cf92b4` |

CUDA smoke checks passed on the new binaries: 7 of 7 CTest entries including
`CudaNumericalParity`, `MemoryProfile-cuda` and `ConcurrentPipeline-cuda`. Three focused
MLX CTest entries passed on the Mac.

The workload is unchanged from September 28: uniform F16 weights, execution, KV and
wire; MLX layers 0–14 and CUDA layers 15–35; 128 prompt tokens, 256 outputs,
concurrency one, 384-token reservation. Admission caps remained MLX 4.375 GiB,
CUDA host 3.75 GiB and CUDA device 6.125 GiB, with 10% safety, 256 MiB extra overhead,
1 GiB host headroom and 512 MiB device headroom.

## Resource conditions

The user stopped a Minecraft server on the CUDA host before the run; it had held about
4 GiB resident plus 4.8 GiB of swap. The host then showed about 9 GiB available.
On October 10 the user freed Mac memory; the Mac showed 7.7–8.1 GiB free-plus-inactive
at launch, above the 6.06 GiB the MLX preflight requires. `caffeinate -i -s` ran on the
Mac for the whole window. No other application was stopped by the agent.

Two earlier gate refusals are preserved and were not retried under unchanged
conditions:

- October 9, 04:34 UTC: the Mac preflight refused the MLX profile with 5.21 GiB
  available (`build/4b-fit-20261008-232736/rejected-1/` on the Mac).
- October 10, 05:50 UTC: the Linux guard stopped the first CUDA profile after
  334 MiB of new swap-out in under 60 seconds, with 7.3 GiB still available. The
  kernel paged out idle anonymous memory instead of dropping file cache. The page
  cache was dropped once (`echo 3 > drop_caches`) and the profile was rerun; the
  rejected evidence is in `cuda-profile-rejected-1/`.

Profiles from October 8–9 passed but expired before the soak could run, because the
Mac repeatedly dropped off Tailscale on October 9 (see attempts below). They are kept
under `p1-20261008/` and `p1-20261009/` and are not part of this acceptance.

## P1 — fresh isolated profiles (October 10, 05:52–05:53 UTC)

Both profiles completed three load/execute/cleanup/unload cycles with schema-1.3
telemetry, ordered phases, clean retirement and a `safe` fit on the same binaries.

| Measurement | MLX stage 0 | CUDA stage 1 |
| --- | ---: | ---: |
| Policy applied | `mlx-footprint-max-v1` (no fallback) | `cuda-rss-device-v1` |
| Availability at preflight | 7.18 GiB host | 8.64 GiB host, 7.50 GiB device |
| Lifetime RSS peak | 4.37 GiB | 3.49 GiB |
| Peak OS physical footprint | 4.57 GiB | n/a |
| Envelope incl. allowances and headroom | 5.28 GiB host | 4.09 GiB host, 5.72 GiB device |
| Footprint after each unload | 3.79 GiB | n/a |
| Guard minimum availability | 4.17 GiB | 6.62 GiB |
| New swap-out during profile | 0 | 137 MiB |

The same MLX evidence replayed under `conservative-v1` yields `unsafe` with a
9.06 GiB envelope (`mlx-memory.conservative-fit.json`), so the footprint policy, not the
conservative sum, is what admits this stage on an 18 GiB Mac. The CUDA device
availability at preflight was 7.50 GiB against a 7.49 GiB requirement; the desktop's
154 MiB of VRAM leaves almost no margin for this cap on an 8 GiB card.

## P2 — same-process reload soak

Attempt 5 ran `reload_soak.py --cycles 6 --requests 4 --idle-seconds 10 --timeout 180`
from the CUDA host at 06:02 UTC with the workers under independent 30-minute guards,
the Mac footprint helper on the direct worker child, and bidirectional SSH forwards
over a direct Tailscale path. It completed in 1,174 seconds.

| Check | Result |
| --- | --- |
| Cycles completed | 6 of 6 |
| Exact 256-token continuations against the independent oracle | 24 of 24 |
| Cancellation after three tokens with exact prefix | 6 of 6 |
| Fresh fit before every deployment | 12 of 12 `safe` (MLX `mlx-footprint-max-v1`, CUDA `cuda-rss-device-v1`) |
| Reservation and weight retirement after every cycle | clean on both workers |
| Per-request wall time | 37.1–44.5 s (polling harness, not a benchmark) |

Reported CUDA device availability before cycles 1–5 was 2.52–2.55 GiB with 4.78 GiB
held inactive by the worker's own allocator; the effective availability of 7.30–7.35 GiB
admitted each reload. The MLX host availability stayed between 8.27 and 8.88 GiB.

Residual memory after unload and ten idle seconds:

| Cycle | MLX footprint | MLX RSS | CUDA RSS | CUDA allocator active / cached |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.88 GiB | 0.84 GiB | 1.30 GiB | 0.01 / 4.78 GiB |
| 1 | 0.88 GiB | 0.83 GiB | 1.32 GiB | 0.01 / 4.78 GiB |
| 2 | 0.09 GiB | 0.04 GiB | 1.32 GiB | 0.01 / 4.78 GiB |
| 3 | 0.15 GiB | 0.11 GiB | 1.32 GiB | 0.01 / 4.78 GiB |
| 4 | 0.15 GiB | 0.11 GiB | 1.32 GiB | 0.01 / 4.78 GiB |
| 5 | 0.08 GiB | 0.04 GiB | 1.13 GiB | 0.01 / 4.78 GiB |

The MLX residual fell from about 0.9 GiB to under 0.2 GiB from the third cycle on;
this is the observed trend over six cycles, not evidence of an indefinite plateau. The
CUDA process keeps its cached device blocks until exit, as the M6 report also noted.
Logical weights, cache and workspace reservations returned to zero on both workers after
every unload.

## Serving peaks versus prelaunch budgets

Serving peaks are compared against the availability recorded at the first fresh
preflight, with the unchanged 10% safety, 256 MiB extra overhead and headroom (1 GiB
host, 512 MiB device). An isolated profile is not serving evidence; this is the first
same-process serving comparison on both hosts.

| Domain | Serving peak | Envelope with allowances | Prelaunch availability | Isolated-profile envelope |
| --- | ---: | ---: | ---: | ---: |
| MLX host (OS footprint lifetime peak) | 4.58 GiB | 6.28 GiB | 8.27 GiB | 5.28 GiB |
| CUDA host (RSS lifetime peak) | 3.50 GiB | 5.10 GiB | 8.92 GiB | 4.09 GiB |
| CUDA device (process VRAM, `nvidia-smi` every 2 s, 545 samples) | 4.97 GiB | 6.22 GiB | 7.35 GiB | 5.72 GiB |

All three serving envelopes fit their prelaunch availability. The MLX serving footprint
peak equals the isolated profile's 4.57 GiB within 10 MiB; the CUDA process VRAM peak
of 4.97 GiB sits under the profile's 5.72 GiB device envelope. The CUDA allocator peak
was 4.76 GiB with a 4.71 GiB active maximum. Guard observations: the Mac guard saw a
4.82 GiB minimum availability, normal pressure throughout and zero new swap-outs over
607 samples; the Linux guard saw a 7.33 GiB minimum and 179 MiB of new swap-out spread
over 609 samples, never exceeding the 60-second limit.

## P4 — reverse-order live serving (CUDA → MLX, split 21)

The same inputs were rerun with CUDA owning layers 0–20 and MLX owning layers 21–35
plus the final norm, head and sampling (plan digest `a653ceac6134`, identical to the
September 18 reverse plan). Caps, allowances and the independent oracle were unchanged.
Both fresh profiles were `safe` (06:26–06:28 UTC), and the six-cycle soak that
followed (06:30 UTC, 1,168 s) matched the forward order exactly.

| Measurement | CUDA stage 0 (0–20) | MLX stage 1 (21–35 + head) |
| --- | ---: | ---: |
| Isolated profile policy | `cuda-rss-device-v1` | `mlx-footprint-max-v1` (no fallback) |
| Isolated envelope | 4.08 GiB host, 5.72 GiB device | 5.28 GiB host |
| Isolated peak | 3.48 GiB RSS | 4.57 GiB footprint, 4.37 GiB RSS |
| Conservative replay of the same evidence | n/a | `unsafe`, 9.06 GiB |
| Soak: exact continuations / cancel prefixes | 24 of 24 / 6 of 6 | |
| Soak: fresh preflights | 12 of 12 `safe` | |
| Serving peak | 3.53 GiB RSS, 5.00 GiB process VRAM | 4.58 GiB footprint |
| Serving envelope vs prelaunch availability | 5.13 ≤ 9.21 GiB host; 6.25 ≤ 7.32 GiB device | 6.29 ≤ 8.73 GiB |
| Guard minimum availability / new swap-out | 7.02 GiB / 0 | 4.81 GiB / 0 |

Per-request wall time was 38.3–41.5 s. The MLX final stage, which the September 18
conservative assessment rejected at 6.5–6.7 GiB, fits under the footprint policy with
the same 4.375 GiB cap. Its idle residual after unload was 0.08–0.15 GiB from the first
cycle on; the CUDA worker again retained 4.78 GiB of cached device blocks until exit,
which the credited gate admitted on cycles 1–5 (2.50–2.52 GiB reported free). This is
live reverse-order serving through RPC, not the September 18 offline stage replay.

## P3 — 512-prompt/256-output context (768 cached tokens)

A fresh independent oracle was generated on the CUDA host at 06:50 UTC with hLLM
workers stopped, using the pinned reference environment (`torch 2.13.0+cu130`, `transformers 4.57.6`,
Accelerate 1.15.0 added that night because the CPU-offload device map requires it) and
`checkpoint_reference.py --plain --dtype f16 --gpu-layers 26 --prompt-tokens 512
--output-tokens 256`. It produced 512 prompt and 256 generated tokens in 68.9 s with a
5.79 GiB peak CUDA allocation; the guard saw a 6.56 GiB minimum host availability and
98 MiB of new swap-out. The prompt is the tool's deterministic token repetition, so the
Base model's continuation largely repeats it; exact token equality is still the check.

The MLX cap was raised to 5.125 GiB (5,502,926,848 bytes) for this workload, as the
backlog proposed; CUDA caps were unchanged. Both isolated profiles passed on the same
binaries (06:52–06:54 UTC), and the forward-order MLX→CUDA plan was reused.

| Measurement | MLX stage 0 (cap 5.125 GiB) | CUDA stage 1 (caps unchanged) |
| --- | ---: | ---: |
| Native peak reservation (weights + cache + workspace) | 5.01 GiB of 5.125 | 6.10 GiB of 6.125 device |
| Policy applied | `mlx-footprint-max-v1` (no fallback) | `cuda-rss-device-v1` |
| Availability at preflight | 10.62 GiB | 9.13 GiB host, 7.50 GiB device |
| Isolated envelope | 5.53 GiB host | 4.09 GiB host, 5.87 GiB device |
| Peak | 4.80 GiB footprint, 4.38 GiB RSS | 3.49 GiB RSS |
| Footprint after each unload | 4.53 GiB | n/a |
| Guard minimum availability / new swap-out | 4.80 GiB / 0 | 7.78 GiB / 45 MiB |

The conservative replay of the 768-token MLX evidence is also `safe` here, at
9.32 GiB, only because 10.62 GiB happened to be available at that preflight; it is
not a general conservative-policy pass. The CUDA stage sits 27 MiB under its native
device cap and the MLX stage 118 MiB under its new cap, so this workload is at the
admission limit of the current caps rather than comfortably inside it.

The bounded soak (`--cycles 3 --requests 2`, 06:56 UTC, 383 s) ran against the new
oracle with the workers started at `--max-cached-tokens 768` and the MLX worker at the
5.125 GiB cap.

| Check | Result |
| --- | --- |
| Exact 256-token continuations after a 512-token prompt | 6 of 6 |
| Cancellation after three tokens with exact prefix | 3 of 3 |
| Fresh fit before every deployment | 6 of 6 `safe` |
| Per-request wall time | 41.7–49.4 s |
| Serving peak, MLX | 4.80 GiB footprint; envelope 6.53 ≤ 9.27 GiB available |
| Serving peak, CUDA | 3.52 GiB RSS, 5.11 GiB process VRAM; device envelope 6.37 ≤ 7.32 GiB |
| Guard minimum availability / new swap-out | Mac 4.64 GiB / 0; Linux 6.75 GiB / 0.1 MiB |

The MLX residual after unload was 0.88 GiB in all three cycles, and the CUDA worker
retained 4.92 GiB of cached device blocks (credited on cycles 1–2 with 2.38 GiB
reported free). The 768-token workload therefore fits and serves exactly on this pair,
but with the admission margins noted above; any larger context needs new caps and a
fresh independent oracle, not an extrapolation from this run.

## Soak attempts and the CUDA reload gate

Five launches were needed. The first three failed for the same external reason and
produced no cycle:

| Attempt | Outcome |
| --- | --- |
| 1 (Oct 9, 13:43 UTC) | Mac left Tailscale 98 s in; stream closed after 254 tokens. |
| 2–3 (Oct 9) | Mac reconnected for under three minutes each time; the tunnel never completed. |
| 4 (Oct 10, 05:55 UTC) | Cycle 0 passed: four exact continuations, exact cancel prefix, clean unload. Cycle 1 refused by the fresh CUDA fit gate. |
| 5 (Oct 10, 06:02 UTC) | Reported below. |

Attempt 4 exposed a gate defect rather than a memory failure. After unload the CUDA
worker's caching allocator kept 4.78 GiB reserved with 0.01 GiB active, so
`cudaMemGetInfo` reported only 2.52 GiB free and the 6.14 GiB device envelope could not
fit. The same process reuses those blocks on the next load, so the gate was rejecting
a reload the worker could perform. `reload_soak.py` now credits only the worker's own
inactive cached device-allocator bytes, never active bytes, and records the reported,
credited and effective availability in every cycle's preflight. Host budgets, MLX
budgets, allowances and the production fit policies are unchanged; the tool change is
covered by a unit test and documented in the
[tools README](../../scripts/validation/README.md#reload-soak). Attempt 4's full
report is preserved as `soak-attempt-4-cuda-gate/`.

## Cleanup

Only the recorded guard, worker and tunnel processes were stopped, by signalling the
guards. Both guard logs end with a signal-15 exit record. Afterwards neither host had an
hLLM worker, profiler, guard or soak process; no CUDA compute process remained and VRAM
returned to 154 MiB; ports 50291 and 50293 were free on both hosts; the Mac showed
8.6 GiB free-plus-inactive with normal pressure and the Linux host 9.4 GiB available.
`caffeinate` was left to expire on its own. No background job or automation was
scheduled. See `cleanup.json` in the Linux run directory.

## What this does and does not qualify

Qualified by this run, for these binaries and this host pair:

- Physical fit of the 4B MLX stage under the opt-in `footprint-v1` policy with
  actual same-process serving peaks, and of the CUDA stage under its existing policy.
- Six same-process reload cycles with exact tokens, exact cancellation prefixes,
  reservation cleanup and fresh fit before every deployment, in both stage orders.
- The 512-prompt/256-output workload in forward order with a 5.125 GiB MLX cap, over
  three reload cycles against a fresh independent oracle.

Not qualified: contexts above 768 cached tokens, concurrency two (P5), any indefinite reload plateau, throughput or latency, the M5
WAN acceptance sweep, and `footprint-v1` as a default. The CUDA device cap leaves
under 20 MiB of preflight margin on the 8 GiB card with the desktop running, and the
Linux host swapped idle memory under load even with 7 GiB available; both are
operating constraints to record before expanding the workload. The fresh-fit credit
for same-process cached allocator bytes is a validation-tool change awaiting review
in the accompanying PR; the production activation path is unchanged.

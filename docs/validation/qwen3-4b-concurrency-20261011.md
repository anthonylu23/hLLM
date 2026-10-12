# Qwen3-4B concurrency-two night — October 11, 2026

Status: **the concurrency-two experiment (P5) did not run.** The CUDA side was fully
prepared (fresh profile, worker, sampler, tunnel and launch sequence) and a new
1024-prompt oracle was produced, but the Mac never met the MLX profile preflight
while the user's desktop applications were running (two refusals, 4.66 and 5.25 GiB observed against 6.06 GiB; watched for nearly three hours). No concurrency-two capacity
is qualified. All times are UTC and taken from the raw guard and profile records.

## Scope and identity

| Item | Value |
| --- | --- |
| Branch | `codex/concurrency-evidence` ([PR #25](https://github.com/anthonylu23/hLLM/pull/25)), stacked on `codex/footprint-default` ([PR #24](https://github.com/anthonylu23/hLLM/pull/24)) |
| Snapshot both hosts built from | `97aa6b0`, source digest `440da50c…` (recorded as `--source-revision` in every profile) |
| Harness snapshot | `0fd5b71`, source digest `f3bcece1…`: a Python-only guard-parsing fix in `concurrent_soak.py`; native binaries byte-identical |
| CUDA worker / memory profiler | `577b9ca0…` / `0320d981…` (RelWithDebInfo, arch 8.6, conda gRPC prefix) |
| MLX worker / memory profiler | `09a79a4c…` / `e7b126f8…` (`build/native/m6-mlx`, RelWithDebInfo) |
| Checkpoint | `Qwen/Qwen3-4B-Base` revision `906bfd4b…`, checkpoint digest `85b0e87d…`, manifest `6a0c45bf…` |
| Plan | forward `['mlx', 'cuda']`, split 15, uniform F16, 128/256 workload, 384 cached tokens (same inputs as October 10, checksums in `inputs/sha256.txt`) |
| Run directories | Linux `~/Projects/experiments/hllm-4b-fit-20261011-044533/`, Mac `build/4b-fit-20261011-044533/` |
| CUDA smoke | 7 of 7 CTest entries on the rebuilt binaries (`logs/cuda-smoke-ctest.log`) |

The Minecraft server stayed stopped. No other GPU process ran at any point
(`nvidia-smi` showed zero compute applications before every launch).

## P1 — fresh profiles

### CUDA (stage 1, layers 15–35): passed on the second attempt

- **Attempt 1, 04:48:43Z–04:49:13Z, stopped by the guard** ("new swap-out exceeds
  configured 60-second limit"). The page cache had been dropped at 04:48:34Z. The
  guard saw availability fall from 9.27 to 7.05 GiB and 265 MiB of swap-out in the
  window: 254 MiB during the cycle-0 weight load (profiler RSS peak 2.89 GiB) and
  11 MiB at 30 s, against the 256 MiB limit. Cycle 0 had completed every phase. Swap is
  an 8 GiB zram device that already held 3.6 GiB of cold desktop pages; the burst was
  the kernel moving more of them to zram during the load, not the profiler's own
  memory. Preserved in `cuda-profile-rejected-1/` with its reasoning. No allowance was
  changed.
- **Attempt 2, 04:50:41Z–04:51:51Z, passed.** The page cache was dropped again at
  04:50:39Z. The only changed condition was that attempt 1 had already moved the cold
  pages into zram; the October 10 retry had passed for the same reason. Guard: 35
  observations, 40 MiB total swap-out, minimum availability 6.90 GiB, exit normal.
  Profile `cuda-memory.json`: schema 1.3, complete, measured at 04:50:49Z, fit **safe**
  under `cuda-rss-device-v1`, host envelope 4.09 GiB, device envelope 5.72 GiB, caps
  3.75 GiB host and 6.125 GiB device. These are the October 10 envelopes to the
  hundredth, so the rebuilt binaries behave as before.

### MLX (stage 0, layers 0–14): refused at preflight

One attempt at 04:52:07Z–04:52:13Z, refused before any model load: "host/unified:
configured probe cap plus physical allowances exceeds availability". The preflight
needs cap × 1.1 + 256 MiB + 1 GiB headroom = 6.062 GiB of free-plus-inactive memory
and observed 4.66 GiB (5,004,361,728 bytes). Preserved in the Mac's `rejected-1/` and
mirrored to `mac/rejected-1/`. The repository's own `host_available_bytes()` read 5.22,
5.61, 5.56, 5.17 and 4.98 GiB between 04:43Z and 04:52Z; the largest resident
processes were the Helium browser (about 2.4 GiB across its processes) and Spotify.
The agent does not quit applications, so this stayed the user's call. A background
watch re-measured availability every five minutes afterwards; its readings are in the
section below.

## Independent oracle: 1024-prompt / 256-output F16

`checkpoint_reference.py --plain --dtype f16 --gpu-layers 26 --prompt-tokens 1024
--output-tokens 256` ran in the pinned reference environment (Torch 2.13.0+cu130,
Transformers 4.57.6, Accelerate 1.15.0) under the Linux guard with no hLLM worker
running, 04:55:24Z–04:57:00Z. Generation took 69.9 s with a peak CUDA allocation of
6.07 GiB; the guard recorded 179 MiB of swap-out and a 6.45 GiB availability minimum,
exit normal. Output: `inputs/reference-1024-256-f16.json` (sha256 `e7844d1b…`, 1024
prompt and 256 generated token IDs on checkpoint digest `85b0e87d…`), also copied to
the Mac with its checksum. It is a pinned reference for a future larger-context
experiment; nothing was served against it tonight.

### What the larger-context experiment would need

From the October 10 768-token profiles, the CUDA stage at 21 layers reserved 6.10 of
its 6.125 GiB device cap. At 1280 cached tokens its KV grows by roughly 45 MiB, so that
split no longer fits the cap and the stage must shrink by one layer (split 16). The MLX
stage then carries 16 layers and 1280 tokens: about 5.25 GiB of native reservation, a
cap near 5.5 GiB and a preflight of about 7.3 GiB free-plus-inactive, further from
tonight's Mac than the baseline. The feasibility planner with the checked-in nominal
budgets (`examples/profiles/workers-m3pro-3060ti.yaml`) marks every forward split from
11 up as feasible, which only shows that the nominal budgets are not the binding ones.

## P5 — concurrency two

**Did not run.** The experiment requires both P1 profiles fresh on the same binaries, and
the MLX profile never existed. The background watch (`mac/availability-watch-1.log`,
the repository's `host_available_bytes()` every five minutes, 04:52:43Z–06:43:07Z,
23 readings) never met the preflight:

| Reading | Value |
| --- | --- |
| Minimum / maximum | 4.76 GiB (04:52Z) / 5.71 GiB (06:43Z) |
| Typical | 5.3–5.5 GiB |
| Preflight requirement | 6.06 GiB free-plus-inactive |
| Watch trigger | requirement plus 200 MiB (6.26 GiB) |

The MLX profile was not retried: the refused condition (the user's desktop applications
resident) never changed, and retrying unchanged resource conditions is excluded. The
GPU side remained ready the whole time and nothing of it ran: `concurrent-cuda-worker.sh`
(two active requests, 768 cached tokens, `--request-observations on`), `tunnel.sh`,
`watch-cuda.sh` and `run-concurrent.sh` (the full sequence: guards on both hosts, both
workers, the soak driver with `--cycles 2 --pairs 3`, evidence assembly and cleanup) are
staged in the run directory, as are the Mac's `mlx-profile.sh` and `mlx-worker.sh`.

### Second MLX attempt, 07:29:50Z: refused again

A second watch (`mac/availability-watch-2.log`, 06:45Z–07:29Z) fired at 07:29:31Z on
6.44 GiB, the first reading above the requirement plus margin; the Mac had been idle
for over three hours, so the applications had released memory on their own. The MLX
profile was launched at 07:29:50Z and refused at 07:29:54Z before any model load:

| Sample | Free-plus-inactive |
| --- | --- |
| Watch, 07:29:31Z | 6.44 GiB |
| Guard, 07:29:50Z (launch) | 6.48 GiB |
| Guard, +2 s | 5.43 GiB |
| Guard, +4 s / preflight, 07:29:54Z | 5.25 GiB |
| Requirement | 6.06 GiB |

About 1.2 GiB of inactive pages were activated or reclaimed by the launch itself (uv,
the guard and the CLI starting) in the four seconds between the watch reading and the
preflight sample. Preserved as `rejected-2/` on the Mac and `mac/rejected-2/` here. Not
retried: availability was 5.48 GiB a minute later. No allowance was changed. A third
watch (`mac/availability-watch-3.log`, 07:31Z–09:31Z, 24 readings) used the requirement
plus a 1.3 GiB launch allowance (7.36 GiB) as its trigger and never fired: the Mac read
5.16–6.03 GiB. The night was closed at 09:31Z without a third launch.

Lesson for the tooling: on macOS the free-plus-inactive reading is not stable at the
second scale, and the preflight samples it after the launch has already touched the
inactive file cache. A pre-launch watch needs a launch allowance of at least 1.2 GiB
on top of the preflight requirement, or the preflight should sample before the CLI's
own imports. This is a follow-up, not a change to tonight's gate.

*October 12 correction.* A third attempt (`mac/rejected-3/`, launched 18:31:06Z from
main `894897a`, binaries unchanged) was refused the same way: the guard read 8.01 GiB
at launch, 7.96 GiB two seconds later and 6.34 GiB at four seconds, and the preflight
sampled 5.91 GiB against 6.06 GiB. Importing the CLI on its own was then measured to
cost nothing (6.00 GiB before, 6.06 GiB after). The cause is the runner's own
checkpoint digest: it reads the whole checkpoint before sampling availability, and
on macOS the pages it has just read sit on the active list, outside the
free-plus-inactive proxy. The runner now samples availability before that read and
records the post-read value alongside it (preflight schema 1.1); the requirement
and allowances are unchanged.

To run P5 by hand once the Mac has about 0.9 GiB more free-plus-inactive memory, and
before the CUDA profile expires at 04:50:49Z on October 12:

```sh
# Mac, serial, about six minutes: MLX P1 profile
nohup ~/Documents/Projects/hLLM/build/4b-fit-20261011-044533/mlx-profile.sh &
# Linux, after the Mac profile is safe: both workers, soak driver, assembly, cleanup
~/Projects/experiments/hllm-4b-fit-20261011-044533/run-concurrent.sh
python ~/Projects/experiments/hllm-4b-fit-20261011-044533/assemble_results.py \
  ~/Projects/experiments/hllm-4b-fit-20261011-044533 \
  docs/validation/qwen3-4b-concurrency-20261011-results.json
```

After the expiry the CUDA profile must be re-measured first (`cuda-profile.sh`, with the
page cache dropped and the guard's 256 MiB swap-out limit unchanged).

**October 12 addendum.** The review follow-up on the harness (schema 1.1) binds the
`--baseline-report` to the worker binaries under test. The staged
`inputs/baseline-reload-soak.json` is the October 10 reload soak on the October 10
binaries (`d49f4ee1…` / `4b177c76…`), not the rebuilt ones (`09a79a4c…` /
`577b9ca0…`), so the harness will now refuse it before loading. P5 therefore needs a
fresh reload soak on the rebuilt binaries first (both workers under their guards,
`reload_soak.py` with the same plan and reference, about 20 minutes of GPU time), and
`run-concurrent.sh` must point `--baseline-report` at that report. The run
directory's `source/` clone also needs the updated harness.

## Cleanup

Verified at 06:43Z–06:45Z after the first watch ended, and again at 07:31Z after the second MLX refusal (guard and profiler had exited on their own; only `caffeinate` remained).

- **Linux:** no hLLM worker, memory profiler, soak driver, resource guard, memory
  sampler or SSH tunnel process; `nvidia-smi` lists zero compute applications and
  129 MiB of device memory in use (the desktop). The only processes the agent started
  tonight, the CUDA profiler (attempts 1 and 2) and the oracle, ran under their guards
  and exited; their PIDs are in `cuda-profile.pid`, `cuda-profile-rejected-1/` and
  `oracle-1024.pid`. The 8 GiB zram swap still holds 3.75 GiB of cold desktop pages, so
  the next weight load should drop the page cache first, as tonight's attempts did.
- **Mac:** only the user's `caffeinate -i` (PID 72845) is running; no MLX worker,
  profiler, guard or mirror loop. The run directory holds `inputs/`, `logs/`,
  `rejected-1/` and the two staged scripts, unchanged.
- The Minecraft service stays stopped; the user decides when it returns. No user
  application was stopped on either host.
- Results file: [`qwen3-4b-concurrency-20261011-results.json`](qwen3-4b-concurrency-20261011-results.json)
  (`assemble_results.py` over the run directory; `p5` is null, both MLX refusals are
  under `p1.mlx_rejected`, the watch readings under `mac_availability_watch`).

## What this night qualifies

- Nothing new on capacity. The CUDA profile is fresh, safe evidence for the forward
  split on the rebuilt binaries and expires 24 hours after 04:50:49Z.
- The concurrency-two tooling is implemented and its CPU rehearsal produces an
  accepted record; the two-host measurement remains the open P5 item and needs the Mac
  at or above 6.06 GiB free-plus-inactive at the preflight sample, which tonight meant
  about 7.3 GiB at the watch reading (about 0.9 GiB more than its best).
- A pinned 1024/256 oracle now exists for the larger-context follow-up.

# Overnight qualification backlog

Updated September 30, 2026. GPU allocation may remain occupied for several days.
The CPU-only work below can proceed independently.

September 28 status: The user resumed this backlog overnight. The fresh Mac
baseline profile passed; CUDA qualification is blocked by an unrelated active GPU
workload. No automation is scheduled. See the
[September 28 run report](validation/qwen3-4b-fit-20260928.md) for scope and results.
That session is complete and no owned model process remains. The Linux CI recipe
passed all 80 CTest entries, and the CUDA memory profiler is rebuilt. Resume with
fresh resource checks; GPU runtime checks and P1's CUDA profile are next.

## Handoff for the next Codex

Continue on `codex/4b-qualification` in `/Users/anthony/Documents/Projects/hLLM`.
Read the applicable `AGENTS.md`, this backlog and `scripts/validation/README.md`.
Inspect all existing changes before editing; telemetry, prior 4B reports and the
new policy/tooling are now grouped into review commits. Do not reset or overwrite that work.
Use the computer-fleet skill for `anthonylu@anthonypc`; inspect the current remote
checkout before syncing anything. Keep previous evidence immutable. The user authorized implementation, scoped commits and draft PR preparation on
September 30. Do not merge or alter unrelated workloads without authorization.

The new policy is **opt-in**. A fresh 4B Mac isolated profile passed; fresh two-host
serving qualification remains pending.
Keep `conservative-v1` as the default. Missing/incoherent telemetry falls back to
the existing RSS-plus-allocator rule. Always report the actual policy used.

## Completed during the low-memory session

- [x] Implement `footprint-v1` with complete schema-1.3 telemetry, RSS/allocator
  cross-checks, unchanged allowances and recorded conservative fallback.
- [x] Propagate explicit policy through profiling CLI, measured bundles, fresh
  activation, native sweep execution and independent exclusion validation.
- [x] Add regression tests for physical gates, old hashes, fallback and policy binding.
- [x] Preserve reusable resource guards, process-footprint sampler and reload harness.
- [x] Add plain Base-model tokenization and bounded oracle placement options to the
  existing reference tool; full 4B execution of these options remains pending.
- [x] Add Python and Linux CPU CI workflow; hosted execution remains pending.
- [x] Run lightweight Python and native tiny-model checks. See the
  [implementation validation](validation/footprint-policy.md).

## GPU-independent work — September 30

- [x] Review the existing telemetry, policy and tooling changes and package focused commits.
- [x] Publish the independent CPU CI draft [PR #17](https://github.com/anthonylu23/hLLM/pull/17).
  Hosted results are tracked in the [September 30 report](validation/cpu-readiness-20260930.md); a queued/building job is not a pass.
- [x] Test TERM/INT/HUP and exited-leader process-group cleanup, including a descendant
  that ignores TERM. Keep timeout and sampling-failure tests.
- [x] Add an explicit CPU/F32 harness rehearsal and register it with CTest under a
  resource guard. Exercise reload/cancellation/unload and rejection of stale or
  mismatched evidence; retain partial tokens and cleanup observations after worker loss.
- [x] Write the [concurrency-two evidence design](concurrency-qualification.md).
  The concurrency-one profile restriction remains in force; new capacity is unqualified.
- [ ] Complete hosted CI and review before merging independently scoped infrastructure.
- [ ] Resume P0–P2 with fresh profiles when both hosts meet resource gates. Do not
  reuse September 28 profiles past the freshness limit.

Larger context, reverse order and concurrency two remain follow-up experiments;
they are not prerequisites for reviewing the single-request policy implementation.

## Resource gates for every heavy step

The September 27 daytime check saw about **3.9 GiB available on the Mac**, using
the repo's conservative free-plus-inactive calculation, and **5.6 GiB Linux
MemAvailable**. CUDA had no model compute process. These are historical snapshots;
the user's expected 11 GB free is not a substitute for a new check.

1. Record Mac `vm_stat`, pressure, swap usage, load, disk and top processes; record
   Linux `free -h`, `nvidia-smi`, load, disk and relevant processes. Do not stop other
   applications. Use `host_available_bytes()` for the same availability definition
   as profiling and native activation.
2. Aim for at least 8 GiB Mac availability for the baseline; 8–9 GiB is a planning
   window for 512/256, not its acceptance threshold. Native configured caps plus
   10% safety, 256 MiB extra and headroom must pass live preflight. Host headroom is
   1 GiB; CUDA device headroom is 512 MiB. Never reduce allowances to pass a run.
3. Use `resource_guard.py` on **both** machines for every model-bearing worker or
   probe. Stop on the existing swap/pressure conditions, sampling failure or time
   limit, and the 1 GiB available-memory floor. Protect remote work independently
   of SSH/controller lifetime. No unbounded model process.
4. One substantial GPU workload at a time. Never run the independent oracle and
   hLLM CUDA workers together. Respect Linux host memory as well as free VRAM.
5. On any gate failure, preserve the rejected report, clean up owned processes and
   stop that expansion. Missing evidence is `unknown`, not passing. Do not retry
   unchanged resource conditions repeatedly.

## P0 — freeze source identity and verify the runnable checkout

- [x] Create a new exclusive evidence directory: `build/4b-fit-20260928-042730/`.
  The remote snapshot is in
  `/home/anthonylu/Projects/experiments/hllm-4b-fit-20260928-042730/source`.
- [x] Capture base commit, dirty diff, hashes of all native/protocol/Python/tool
  sources and the exact worker/profiler binary hashes. Rebuild if sources changed.
  Keep the Python packages synchronized for cross-host helper execution; do not
  assume a matching base commit identifies uncommitted changes.
- [x] Recheck the pinned checkpoint on both hosts: `Qwen/Qwen3-4B-Base`, revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`; full manifest hashes and three shards
  must agree. No checkpoint downloads if a valid complete copy is present.
- [ ] Run the changed Python suite and Linux/CUDA memory-profile/worker smoke
  checks against the actual binaries. Run broader regression tests if these fail.
  September 28: 156 Python tests passed on each host, plus focused Mac native
  checks. Existing Linux CPU profiler tests detected an old schema-1.0 binary;
  a clean build with the pinned dependencies passed all 80 Linux CTest entries.
  The CUDA memory profiler was rebuilt with its previous binary preserved;
  CUDA runtime checks await GPU availability.
- [x] Exercise the new Linux CPU CI dependency/build recipe on a clean hosted
  runner or isolated Linux build when convenient. Workflow syntax alone is not a
  successful hosted CI run; this is independent of 4B capacity qualification.
  September 28: the isolated Fedora build/test recipe passed after adding the
  CMake 4 compatibility setting for c-ares. The native CI timeout is now 60 minutes
  to accommodate a cold dependency build. Hosted Actions still has not run.

Known paths, to verify before use:

| Artifact | Last known location |
| --- | --- |
| Mac checkout | `/Users/anthony/Documents/Projects/hLLM` |
| Mac model | `build/models/Qwen3-4B-Base` |
| Mac worker/profiler | `build/native/m6-mlx/cpp/hllm-worker-mlx`, `hllm-profile-memory-mlx` |
| Full manifest / 128-token independent oracle | `build/4b-qualification-20260918/manifest.json`, `reference-f16.json` |
| Prior raw footprint evidence | `build/4b-footprint-20260923/` |
| CUDA source/build | `/home/anthonylu/Projects/experiments/hllm-m6-serving`, `build/cuda/cpp/` |
| Linux model | `/home/anthonylu/Projects/experiments/hllm-4b-20260918/model` |
| Previous remote diagnostics | `/home/anthonylu/Projects/experiments/hllm-4b-footprint-20260923/` |
| Previous Linux build/runtime environment | `/tmp/hllm-pr-review-env.sh` (inspect; may be stale) |

September 28 discovered that the original `build/cuda` memory-profiler executables
predated the process-telemetry changes. The CUDA memory profiler at that path is
now rebuilt and hashed in the session report; its predecessor is preserved under
the new remote run directory. The old CPU profiler remains stale: use the newly
built `source/build/native/ci-cpu/cpp/hllm-profile-memory-cpu` under the new remote run directory.
Do not infer executable freshness from matching adjacent source files alone.

Use a new remote evidence directory and explicit paths, not these old result
directories. If the remote source is dirty, preserve it and use a separate snapshot
directory. Reuse existing large dependencies/model files without mutating them.

## P1 — fresh baseline profiles and policy qualification

Target: uniform F16 weights/execution/KV/wire; MLX layers 0–14 and CUDA layers
15–35; 128 prompt tokens, 256 outputs, concurrency one, 384-token reservation.
Historical admission caps: MLX 4.375 GiB (`4697620480`), CUDA host 3.75 GiB
(`4026531840`), CUDA device 6.125 GiB (`6576668672`). Reconfirm live preflight.

- [x] Write a fresh explicit plan using `scripts.validation.checkpoint_run.make_plan`
  with names `['mlx', 'cuda']`, `DType.F16`, split `15`; preserve the pinned manifest.
  Write a `WorkloadProfile` with 128/256/384, `kv_dtype=F16`, concurrency 1. The
  default execution dtype in other examples may be F32: do not reuse it accidentally.
- [ ] Under the local guard, run `hllm profile-memory` on MLX for three cycles,
  adding `--mlx-fit-policy footprint-v1`. Under the remote guard, run the matching
  CUDA profile for three cycles. Run serially and preserve stdout, native JSONL,
  preflight, artifact and `.fit.json` files. Include the new source snapshot digest.
  September 28: **MLX completed and passed** in the new run directory; CUDA is
  pending because another job occupies the device. Preserve this profile and use
  a new filename if freshness or changed conditions require another run.
- [ ] Require completed schema-1.3 profiles, all ordered phases, coherent lifetime
  peaks, clean retirement and `safe` fit. MLX must actually report
  `policy: mlx-footprint-max-v1`; a fallback is not qualification of this policy.
  Retain a separate conservative assessment of the same evidence for comparison.
  The Mac result uses `mlx-footprint-max-v1` with no fallback; its conservative
  comparison is unsafe. This checkbox stays open until CUDA also passes.

MLX command template, after the new run directory and inputs exist:

```bash
uv run python scripts/validation/resource_guard.py \
  --seconds 1200 --record "$HLLM_RUN/mlx-profile.guard.jsonl" -- \
  uv run hllm profile-memory \
  --manifest "$HLLM_RUN/manifest.json" --plan "$HLLM_RUN/mlx-cuda.plan.json" \
  --workload "$HLLM_RUN/workload-128-256.json" --stage-index 0 \
  --model-root build/models/Qwen3-4B-Base \
  --binary build/native/m6-mlx/cpp/hllm-profile-memory-mlx --backend mlx \
  --unified-bytes 4697620480 --cycles 3 --timeout 1100 \
  --mlx-fit-policy footprint-v1 --source-revision "$HLLM_SOURCE_ID" \
  --concurrent-load "$HLLM_OBSERVED_LOAD" --output "$HLLM_RUN/mlx-memory.json"
```

Here `HLLM_RUN`, `HLLM_SOURCE_ID`, and `HLLM_OBSERVED_LOAD` are task-specific values populated by
the overnight Codex from that session. On Linux use the same inputs and the CUDA
profiler with `--backend cuda --stage-index 1 --host-bytes 4026531840
--device-bytes 6576668672`; use paths on that host and copy its full evidence back.
Omit the footprint helper around Python profiler commands; see the tools README.

## P2 — longer same-process reload soak

- [ ] Only after P1 passes, start a guarded worker per host, each with a 30-minute
  limit. Compile/use `process_footprint.c` for the direct Mac worker child; retain
  independent OS samples during load. The controller does not own these workers,
  so the outer supervisor must stop the guards at the end.
- [ ] Use bidirectional SSH forwarding if needed, after checking ports are unused:
  local MLX `127.0.0.1:50291`, CUDA `127.0.0.1:50293`, with
  `ssh -o ExitOnForwardFailure=yes -L 50293:127.0.0.1:50293
  -R 50291:127.0.0.1:50291 -N anthonylu@anthonypc`. Record transport identity;
  this is not M5 WAN acceptance.
- [ ] Construct `soak-workers.json` as in the tools README, using fresh profiles,
  independently checked executable digests and the reported runtime driver API.
  Keep worker caps identical to P1. Use `--max-cached-tokens 384` and default
  concurrency/batch limits of one.
- [ ] Run `reload_soak.py --cycles 6 --requests 4 --idle-seconds 10`. Reuse the
  pinned independent 128/256 oracle only after validating its payload identity.
  This is 24 exact continuations and six cancellation/unload cycles in the same
  worker processes, roughly 20 minutes based on previous timings.
- [ ] Inspect fresh gates, every continuation, reservation cleanup, Mac pressure,
  swap-outs, Linux availability and sampling errors. Compare **serving** peaks
  against prelaunch availability with unchanged allowances; a passing isolated
  profile alone is insufficient. Record maximum OS footprint/RSS and allocator
  active+cache/peak separately, plus the selected-policy envelope.
- [ ] Compare unloaded footprint at each cycle. Capture `vmmap -summary` after the
  final unload if useful. Report the measured trend; six cycles do not prove an
  indefinite plateau. About 0.9 GiB retained after unload was previously observed;
  process exit is the reliable complete release. Do not add allocator flushing here.

## P3 — expand context, only after the baseline serving-fit result passes

- [ ] Generate a fresh **independent** 512-prompt/256-output F16 oracle on Linux,
  with hLLM GPU workers stopped. Use the pinned oracle environment and the guarded
  `checkpoint_reference.py --plain --dtype f16 --gpu-layers 26 --prompt-tokens 512
  --output-tokens 256` path documented in the tools README. Check host/VRAM fit
  first; the previous mapping may need a separately justified adjustment.
- [ ] Create a 768-token workload and fresh isolated profiles. The MLX formula
  reserves about 5.007 GiB including resident weights for this split, versus the
  old 4.375 GiB cap. A starting candidate cap is 5.125 GiB, subject to native load
  and live physical preflight. Calculate the CUDA requirement independently.
  Do not just reuse 384-token caps/profiles or claim safety from an extrapolation.
- [ ] If profiles and fresh gates pass, run one guarded 256-token request and its
  repeat, then three reload cycles. Stop on exact-token, cleanup, telemetry or
  physical-fit failure. Record whether the new workload actually fits, even if its
  failure is an admission limitation rather than a regression.

## P4 — reverse-order live serving

- [ ] Return to 128/256 first. CUDA owns layers 0–20 and MLX 21–35: order
  `['cuda', 'mlx']`, split `21`. The final MLX stage owns normalization/head/sampling,
  so forward-order profiles cannot qualify it.
- [ ] Reprofile both assignments, pass fresh physical gates, then run the same
  independent oracle through live RPC serving and the reload/cancel sequence.
  The September 18 reverse-order evidence was offline numerical stage replay,
  not live reverse-order serving.

## P5 — concurrency is a separate gated follow-up

- [ ] Start with two requests at 128/256, only after single-request acceptance.
  Check combined reservations and actual active request overlap; do not double
  the model weight allocation in the estimate. Each extra Mac request adds about
  0.669 GiB logical reservation at this workload, before batching effects.
- [ ] `ProfileKey` currently supports concurrency one. Add/justify a separate
  combined-memory qualification path before treating these profiles as evidence
  for concurrency two. If that needs design work, leave P5 pending this night.
- [ ] Then use `serving_soak.py` for bounded HTTP levels 1 and 2, matching worker
  `--max-active-requests`, aggregate `--max-cached-tokens`, batch/chunk settings and
  the actual tokenized prompts. Its built-in prompts are not exactly 128 tokens;
  record the real workload and gate its capacity rather than relabeling it.
  Keep batch-dependent sampling limits explicit. Stop before any higher level
  that lacks physical evidence. Do not mix this with a performance-speedup claim.

## Final cleanup and report

- [x] Stop only the recorded owned worker/guard/tunnel PIDs. Verify no owned model
  workers remain on either host and no owned CUDA compute process remains.
  September 28 cleanup passed on both hosts; no tunnel was started. Repeat after
  every subsequent run.
- [x] Save source, binary, checkpoint and raw-evidence hashes; all failures and
  fallbacks; exact scope, caps, actual policies and fresh budgets; token comparisons;
  guard summaries; reload residuals; and cleanup evidence.
- [x] Write a concise tracked Markdown report and machine-readable summary in
  `docs/validation/`, leaving bulky raw artifacts under the new ignored run directory.
  Update this checklist with links and outcomes. Update README/milestone next steps
  only for gates that actually passed.
  September 28: [report](validation/qwen3-4b-fit-20260928.md) and
  [machine-readable result](validation/qwen3-4b-fit-20260928-results.json). The
  unrun CUDA/serving/expansion gates remain open above.
- [ ] Leave M5's exhaustive WAN acceptance, 32K context, runtime offloading,
  quantization and additional backends outside this night's acceptance claims.

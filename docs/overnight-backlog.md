# Two-host qualification log (September 27 – October 13, 2026)

This is the historical record of the guarded two-host 4B qualification steps P0–P5 and
what each found. Validation reports cite these step names. It is not a run book: to run
a two-host experiment, use the [two-host launchers](../scripts/validation/README.md#two-host-launchers),
which also hold the resource gates and run checklist. Current status, open items and
next steps are on [the status page](STATUS.md).

## Supporting work

### Low-memory session (September)

- [x] Implement `footprint-v1` with complete schema-1.3 telemetry, RSS/allocator
  cross-checks, unchanged allowances and recorded conservative fallback.
- [x] Propagate explicit policy through profiling CLI, measured bundles, fresh
  activation, native sweep execution and independent exclusion validation.
- [x] Add regression tests for physical gates, old hashes, fallback and policy binding.
- [x] Preserve reusable resource guards, process-footprint sampler and reload harness.
- [x] Add plain Base-model tokenization and bounded oracle placement options to the
  existing reference tool.
- [x] Add the Python and Linux CPU CI workflow; hosted execution completed September 30
  (one help-text test failure, fixed in [PR #19](https://github.com/anthonylu23/hLLM/pull/19)).
- [x] Run lightweight Python and native tiny-model checks. See the
  [implementation validation](validation/footprint-policy.md).

### GPU-independent work, September 30

- [x] Publish the independent CPU CI draft [PR #17](https://github.com/anthonylu23/hLLM/pull/17).
  Hosted results are in the [September 30 report](validation/cpu-readiness-20260930.md).
- [x] Test TERM/INT/HUP and exited-leader process-group cleanup, including a descendant
  that ignores TERM.
- [x] Add an explicit CPU/F32 harness rehearsal under a resource guard, registered with
  CTest: reload, cancellation and unload, rejection of stale or mismatched evidence,
  partial tokens and cleanup observations after worker loss.
- [x] Write the [concurrency-two evidence design](concurrency-qualification.md).
- [x] Land the stacked CI fixes: #19 → #17 → `main` and #20 → #18 → `main`.

### October 6–8 review follow-up

- [x] Reject any deployment entry before validation in the invalid-evidence tests.
- [x] Test busy-worker refusal with a real two-worker CPU deployment and its exact
  continuation afterwards.
- [x] [PR #21](https://github.com/anthonylu23/hLLM/pull/21): the October 6
  [hosted run](https://github.com/anthonylu23/hLLM/actions/runs/37494224305) passed
  169 Python tests and all 81 CPU CTests. Scope and local evidence are in the
  [review follow-up report](archive/review-followup-20261006.md).

### GPU-independent work, October

- [x] Deps-only CI job on `push` to `codex/**`, so stacked PRs stop rebuilding gRPC cold.
- [x] Linux AddressSanitizer/UBSan CTest job using the `asan` preset. Container-overflow
  and leak detection are disabled because the dependency prefix is uninstrumented (see
  the workflow comment).
- [x] `ConcurrentServingEvidence` model, validator, immutable writer and CPU rejection
  tests, plus opt-in request-lifecycle observations (`--request-observations on`);
  `scripts/validation/concurrent_soak.py`. October 11.
- [x] Promotion criterion for `footprint-v1`, written in the
  [profiling workflow](milestone-5-profiling.md#promotion-criterion-for-footprint-v1).
  The evaluation default flipped to `footprint-v1` on October 10 when PR #22 merged.
- [x] Reconcile `SPEC.md` with the implementation (PR #26, October 12).

## P0 — source identity and a runnable checkout

- [x] Exclusive evidence directory per session (first: `build/4b-fit-20260928-042730/`),
  with the remote snapshot in a matching directory on the CUDA host.
- [x] Base commit, dirty diff, hashes of all native/protocol/Python/tool sources and the
  exact worker and profiler binary hashes; rebuild if sources changed.
- [x] Pinned checkpoint on both hosts: `Qwen/Qwen3-4B-Base`, revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`; full manifest hashes and three shards agree.
- [x] Python suite and Linux/CUDA smoke checks against the actual binaries.
  September 28: 156 Python tests on each host; the old Linux CPU profiler was a stale
  schema-1.0 binary, and a clean build passed all 80 Linux CTest entries. October 8:
  `main` at `a2b62a9` rebuilt on both hosts (source digest `abadd3fd…`); 7 of 7 CUDA
  CTest entries passed.

Lesson from September 28: do not infer executable freshness from matching source files.
The original `build/cuda` memory profilers predated the process-telemetry changes.

## P1 — fresh baseline profiles and policy qualification

Uniform F16 weights, execution, KV and wire; MLX layers 0–14 and CUDA layers 15–35;
128 prompt tokens, 256 outputs, concurrency one, 384-token reservation. Admission caps:
MLX 4.375 GiB (`4697620480`), CUDA host 3.75 GiB (`4026531840`), CUDA device 6.125 GiB
(`6576668672`).

- [x] Explicit plan from `scripts.validation.checkpoint_run.make_plan` (`['mlx', 'cuda']`,
  `DType.F16`, split 15) and a 128/256/384 `WorkloadProfile` with `kv_dtype=F16`. The
  default execution dtype in other examples is F32; do not reuse it by accident.
- [x] Three-cycle guarded profiles on each host. September 28: MLX passed; CUDA waited for
  the device. October 10: both passed on the same binaries (MLX footprint envelope
  5.28 GiB; CUDA 4.09 GiB host, 5.72 GiB device). Earlier October 8–9 profiles expired
  before a soak could run and are retained separately.
- [x] Completed schema-1.3 profiles, all ordered phases, coherent lifetime peaks, clean
  retirement and `safe` fit; MLX reported `policy: mlx-footprint-max-v1` with no
  fallback. The conservative replay of the fresh MLX profile is `unsafe` at 9.06 GiB.

## P2 — longer same-process reload soak

- [x] Guarded worker per host, Mac worker sampled by `process_footprint.c`, controller
  over an SSH tunnel (October 10 topology: Mac as controller; the launchers now run the
  controller on Linux). This is not M5 WAN acceptance.
- [x] `reload_soak.py --cycles 6 --requests 4 --idle-seconds 10`, caps identical to P1,
  `--max-cached-tokens 384`. October 10: 24 of 24 exact continuations and 6 of 6 exact
  cancellation prefixes in 1,174 s. The first full attempt was refused at cycle 1 by the
  fresh CUDA gate because the worker's own cached allocator hid 4.78 GiB; the tool now
  credits only inactive cached bytes and records both values (PR #22).
- [x] Serving peaks compared against prelaunch availability with unchanged allowances;
  OS footprint/RSS and allocator active+cache/peak recorded separately. See the
  [serving report](validation/qwen3-4b-serving-20261010.md).
- [x] Unloaded footprint per cycle: MLX residual 0.88 GiB for two cycles, then
  0.08–0.15 GiB; CUDA keeps 4.78 GiB cached until exit. Six cycles do not prove an
  indefinite plateau; process exit is the reliable complete release.

## P3 — larger context (512/256, 768 cached tokens)

- [x] Fresh independent 512/256 F16 oracle on Linux with hLLM GPU workers stopped
  (`checkpoint_reference.py --plain --dtype f16 --gpu-layers 26`).
- [x] 768-token workload and fresh isolated profiles. October 10: MLX cap 5.125 GiB, CUDA
  caps unchanged; both `safe`, with native reservations at 5.01 of 5.125 GiB (MLX) and
  6.10 of 6.125 GiB (CUDA device).
- [x] One guarded request, its repeat, then three reload cycles: 6 of 6 exact, 3 of 3
  cancel prefixes, serving envelopes inside prelaunch availability on both hosts; see the
  [serving report](validation/qwen3-4b-serving-20261010.md#p3--512-prompt256-output-context-768-cached-tokens).

## P4 — reverse-order live serving

- [x] 128/256 with CUDA owning layers 0–20 and MLX 21–35 (`['cuda', 'mlx']`, split 21).
  The final MLX stage owns normalization, head and sampling, so forward-order profiles
  cannot qualify it.
- [x] Fresh profiles on both assignments (MLX final stage 5.28 GiB under `footprint-v1`,
  where the September 18 conservative sum had refused it) and a six-cycle live reverse
  soak: 24 of 24 exact, 6 of 6 cancel prefixes. See the
  [serving report](validation/qwen3-4b-serving-20261010.md#p4--reverse-order-live-serving-cuda--mlx-split-21).
  The September 18 reverse-order evidence was offline stage replay, not live serving.

## P5 — concurrency two

- [x] Two requests at 128/256 after single-request acceptance, recorded as a separate
  `ConcurrentServingEvidence` record; single-request profiles enter it only as freshness
  preconditions, and `ProfileKey` stays at concurrency one.
- [x] October 11: not measured. The CUDA profile was fresh and safe and the launch staged,
  but the MLX preflight refused twice (4.66 and 5.25 GiB against 6.06 GiB; a watch
  reading of 6.44 GiB fell 1.2 GiB at launch). See the
  [October 11 report](validation/qwen3-4b-concurrency-20261011.md).
- [x] October 13: **measured and accepted** with a user-approved 5.125 GiB MLX cap (at
  4.375 GiB the MLX stage refuses the second reservation, recorded as attempt 1). Three
  cycles, 27/27 exact, overlap and cancellation proven on both workers, physical
  envelopes safe each cycle (MLX 5435–5443 MiB, CUDA device 5932 MiB). PR #27 fixed two
  tooling defects found on the way; the accepted record was measured with an MLX worker
  whose allocator read was later corrected, so a re-run is open on
  [the status page](STATUS.md). See the
  [October 13 report](validation/qwen3-4b-concurrency-20261013.md). No throughput claim
  and nothing above concurrency two follows.

## Cleanup and reporting

- [x] Only recorded owned worker, guard and tunnel PIDs were stopped, and no owned model
  workers or CUDA compute processes remained on either host after each run.
- [x] Reports and machine-readable summaries in `docs/validation/`: the
  [September 28 fit report](validation/qwen3-4b-fit-20260928.md), the
  [October 10 serving report](validation/qwen3-4b-serving-20261010.md) and the
  October [11](validation/qwen3-4b-concurrency-20261011.md) and
  [13](validation/qwen3-4b-concurrency-20261013.md) concurrency reports. Bulky raw
  artifacts stay in the ignored run directories.

These runs make no claim about M5's WAN acceptance, 32K context, runtime offloading,
quantization or additional backends.

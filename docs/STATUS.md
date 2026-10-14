# Project status

Updated October 14, 2026. This page is the single source for what is qualified, what is
open and what comes next. Milestone documents describe design and usage and link here
for status instead of restating it. When a gate passes or a decision is made, update this
page in the same change as its evidence.

## What is qualified

| Area | Result | Evidence |
| --- | --- | --- |
| Milestones 0–4 | Complete: model preparation and planning, native CPU pipeline, CUDA worker, MLX worker, full Qwen3-0.6B across MLX and CUDA over Tailscale | [README milestone table](../README.md#milestone-status) |
| Milestone 5 | Implemented (M5.1–M5.6). WAN performance acceptance is deferred | [WAN closeout](validation/milestone-5-wan-closeout.md) |
| Milestone 6 | Implemented; qualified only for the recorded workloads | [M6 report](validation/milestone-6.md), [sweep](validation/milestone-6-sweep.md) |
| 0.6B precision target | F16 resident weights, F32 execution and KV, F16 wire (user-approved for the M5 0.6B qualification) | [precision decision](milestone-5-precision-proposal.md) |
| Qwen3-4B execution | MLX→CUDA matched an independent F16 reference for 256 tokens | [September 18 report](validation/qwen3-4b.md) |
| Qwen3-4B baseline fit (P1, P2) | 128/256, forward order, fresh profiles on both hosts, six-cycle reload soak, 24/24 exact | [October 10 report](validation/qwen3-4b-serving-20261010.md) |
| Qwen3-4B reverse order (P4) | CUDA→MLX at split 21, six cycles, 24/24 exact | [October 10 report](validation/qwen3-4b-serving-20261010.md#p4--reverse-order-live-serving-cuda--mlx-split-21) |
| Qwen3-4B larger context (P3) | 512/256 at 768 cached tokens, MLX cap 5.125 GiB, three cycles, 6/6 exact | [October 10 report](validation/qwen3-4b-serving-20261010.md#p3--512-prompt256-output-context-768-cached-tokens) |
| Qwen3-4B concurrency two (P5) | Accepted October 13 at 128/256 with a user-approved 5.125 GiB MLX cap, three cycles, 27/27 exact. **Caveat:** measured with an MLX worker whose allocator read was a data race, corrected afterwards in PR #27. The MLX envelope was decided by OS footprint, not the allocator, so the numbers stand, but a re-run on the corrected worker is open | [October 13 report](validation/qwen3-4b-concurrency-20261013.md) |
| MLX fit policy | `footprint-v1` is the evaluation default since October 10, 2026; stored bindings keep "omitted means `conservative-v1`" | [profiling workflow](milestone-5-profiling.md#mlx-footprint-policy) |

All Qwen3-4B evidence uses **uniform F16** weights, execution, KV and wire. The 0.6B
mixed-precision target has not been applied to 4B; see the open decisions below.

## Open decisions (user)

These need a decision before the dependent work can start.

| Decision | What is open | Blocks |
| --- | --- | --- |
| Canonical Qwen3-4B precision target | Uniform F16 (all evidence so far) or the 0.6B target of F16 weights with F32 execution and KV. Profiles bind precision into their identity | Any new 4B profiling beyond re-running recorded configurations; contexts above 768 tokens; 32K |
| MLX per-request workspace | Each active 384-token request reserves about 663–685 MiB on the MLX stage. Whether that workspace can be shared across active requests is a design choice | The MLX cap needed for concurrency two and above and for larger contexts. At 4.375 GiB the MLX stage refuses the second request |
| MLX cap increases | The design forbids automatic cap expansion. 1024/256 needs an MLX cap near 5.5 GiB | The larger-context run |
| Milestone 7 gating | [SPEC §26](../SPEC.md#26-milestones) orders ROCm and more stages after M5's WAN acceptance, which is deferred indefinitely. Decide whether M7 stays gated on it | Milestone 7 scope |
| Persistent reloadable MLX workers | Only worthwhile if long-lived workers are wanted; a fresh worker per deployment is the qualified path | The MLX unload-retention study |

## Open gates and follow-ups

| Item | Owner | Dependency |
| --- | --- | --- |
| Re-run P5 on the corrected MLX worker with fresh same-day profiles on both hosts | Agent, with the user freeing Mac memory | Mac preflight needs about 7 GiB free plus inactive at every cycle gate, lid open, `caffeinate -dis`; idle CUDA host |
| P5 follow-up: bounded HTTP `serving_soak.py` at levels 1 and 2 | Agent | After the P5 re-run. The tool lacks plan binding and a fit gate; harden it first |
| Harness: hash guard records only after the guards stop (the October 13 hashes had to be reconstructed) | Agent | None |
| Harness: bind the CUDA device watcher to the worker's PID. The worker already reports it through the control API (`QualificationState.process_id`, process-memory and observation records); only the launcher still uses `pgrep` | Agent | The launcher scripts, which are not yet tracked in the repository |
| Track the qualification launchers and result-assembly scripts used for the October 11 and 13 records | Agent | Redact host names, addresses and home paths |
| Launcher fails fast when a remote worker does not start, instead of waiting out its loop | Agent | Launchers tracked |
| Planner and legacy controller ignore chunked prefill: any two-stage prompt above the 8 MiB boundary payload (1638 tokens for 4B with an F16 wire) is rejected, and serving refuses chunked prefill with measured bundles | Agent | Workload and profile identity must carry the chunk size. Blocks prompts above about 1.6K tokens and the 32K target |
| Contexts above 768 cached tokens: 1024/256 next. A pinned 1024/256 F16 oracle exists in the October 11 run directory. Needs split 16 (the CUDA device reservation is within 27 MiB of its cap at 768) | Agent | Precision decision, MLX cap approval, Mac memory |
| 32K context | Agent | Chunk-aware planning, profiles, memory |
| M5 WAN performance acceptance: the 54-candidate exhaustive sweep, at most 10% reference drift, regret with a seeded 95% bootstrap upper bound at or below 15%, and the final audit | Deferred | A justified transport improvement or a measurably stable link; see the [restart gates](validation/milestone-5-wan-closeout.md#deferred-gates-and-restart-point) |
| Unified-memory headroom per Mac memory tier (SPEC §30) | Deferred | Access to other Mac tiers; only the 18 GiB machine is qualified |
| Native gRPC acceptability after conversion costs (SPEC §30) | Deferred | M5 WAN acceptance |
| M6: refresh measured profiles with sampling, chunk and batch settings; longer throughput, latency and memory soaks; concurrency 8, pinned CUDA batching, full model context, batch-independent seeded sampling | Agent | Measured bundles currently reject sampling and chunking. See [M6 next steps](milestone-6.md#next-steps) |
| M6 review follow-ups: conservative CUDA load admission, bounded or streamed soak evidence storage, direct-backend API preconditions | Agent | None; see the [M6 review](validation/milestone-6-pr-review.md#follow-ups) |
| Order footprint observations by a steady clock instead of `observed_at_unix_ns` | Agent, low priority | None; the current fallback direction is safe |
| A stage stream closing before `Terminate` reports `INTERNAL`; `ABORTED` would be more precise | Agent, low priority | Process suites pin the current status |
| Apple AddressSanitizer hangs before `main`; Linux ASan and UBSan run in CI, so MLX code has no sanitizer coverage | Deferred | Toolchain |
| MLX cache limit is `min(limit / 8, 64 MiB)` and has not been profiled against full-checkpoint workspace (`TODO` in `cpp/src/mlx/backend.cpp`) | Agent | Mac time |
| CI never compiles the CUDA or MLX backends | Agent | Needs CUDA and LibTorch toolchains; MLX needs a macOS runner |

## Next steps, in order

1. Fix the worker's retirement reasons and the serving streaming-deadline truncation found
   by the October 14 audit. The P5 cancellation evidence relies on the retirement reasons.
2. Track the launchers and result-assembly scripts, and close the harness follow-ups above.
3. Decide the canonical 4B precision target.
4. Re-run P5 on the corrected worker, then the HTTP serving soak at levels 1 and 2.
5. Make the planner and controller chunk-aware, then run 1024/256.
6. Decide the MLX per-request workspace design and Milestone 7's gating.

Qualification runs follow the [overnight backlog run book](overnight-backlog.md) and the
[tools README](../scripts/validation/README.md). Profiles expire after 24 hours, so every
GPU run needs fresh same-day profiles on matching binaries.

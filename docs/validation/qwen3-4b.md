# Qwen3-4B-Base bounded qualification — 2026-09-18

Status: real MLX→CUDA generation and fault recovery passed, including exact
256-token agreement with the independent F16 oracle. **Broader physical-fit
qualification is blocked by the full serving run's conservative MLX envelope.**
Numerical stage replays passed in both orders. All experiment workers stopped;
cleanup was verified at 01:58:56 Eastern on September 18, before the 08:30 cutoff.
No model workers remained when rechecked on September 19.

[Machine-readable results](qwen3-4b-results.json) include the original rejected
profiles, fresh assessments, numerical comparisons, worker identities, cleanup
evidence, and hashes of the raw artifacts.

## Scope and reproducibility

The user authorized a real 4B qualification pass, with Mac memory returned before
08:30 Eastern on September 18. Each heavy model experiment had an independent
timeout and a 12:20 UTC hard stop, leaving ten minutes for cleanup. Guards stopped
owned process groups on sustained Mac pressure, more than 256 MiB of new swap-out
in a minute, or Linux available memory below 1 GiB. Other applications were left
running.

- Complete `Qwen/Qwen3-4B-Base` revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`, downloaded on both machines.
- All three shard hashes match between machines and the pinned download metadata.
  Preparation hashes full payloads: 36 layers, 398 tensors, 8,044,936,192 bytes.
- Runtime source is merged M6, `66227433182b48f1d79fffd5ec112d33e47d44fc`.
  All 121 compared native, protocol, and Python source files match the Linux copy.
- F16 resident weights, execution, KV cache, and stage boundaries. Plain text
  completion tokenization is used for this Base checkpoint.
- Explicit feasible plans; native gRPC over bidirectional SSH forwards over
  Tailscale. This is not measured automatic placement or a WAN acceptance sweep.
- Raw plans, scripts, profiles, traces, independent oracle, guards, and process
  samples are retained under ignored `build/4b-qualification-20260918/`.

The independent oracle uses Transformers 4.57.6 eager attention and Torch
2.13.0+cu130, F16, TF32 disabled, with Accelerate 1.15.0 keeping layers 26–35's
weights on the CPU for dispatch. This memory arrangement applies only to the
oracle; hLLM stages retain their assigned weights on their accelerator. The oracle
is independent of hLLM but is not the F32 oracle used for M5's separate contract.

## Memory findings and placement changes

The existing physical-fit rules are retained: 1 GiB host/unified headroom,
512 MiB device headroom, 256 MiB additional overhead, and 10% safety. MLX's
conservative envelope adds peak process RSS and allocator residency; these are
overlapping accounting views, not a measured additive physical footprint.

The first 14-layer MLX assignment completed two 128-prompt/64-output cycles with
normal pressure and no new Mac swap-outs. The second load increased process RSS
enough to fail the conservative physical-fit assessment. A cold process passed.
Subsequent pipeline experiments therefore used a fresh worker for every deployment;
repeated deployment loading in the same Mac process remains unqualified.

The initial 22-layer CUDA cap failed the physical preflight because NVIDIA driver
reservations reduce usable free VRAM. A 16/20 split passed CUDA checks, but the
Mac envelope was slightly above the available-memory snapshot. A 15/21 split
reduced Mac demand and admitted CUDA loading under the unchanged allowances:

| Worker | Assigned layers | Admission cap |
| --- | ---: | --- |
| MLX | 15 | 4.375 GiB unified |
| CUDA | 21 | 3.75 GiB host; 6.125 GiB device |

MLX→CUDA splits at layer 15; CUDA→MLX splits at layer 21. Both endpoint copies of
the tied embedding are accounted for. Fresh physical availability is reassessed
before each pipeline or numerical experiment. Original preflight/fit rejections
are preserved; a later assessment does not overwrite an earlier failure.

## Initial execution

MLX→CUDA completed the initial 128-prompt/64-output workload. All 64 continuation
tokens matched the independent F16 oracle and a repeated native request. Three
short prompts matched the oracle's first three decisions each. Cancellation after
three tokens, prefill deadline expiry, and token-capacity rejection each released
request reservations and recovered with the expected four-token prefix. Unload
returned both workers' logical weight/cache/workspace reservations to zero.

A second fresh deployment completed 128-prompt/256-output generation and its exact
repeat. Both continuations matched all 256 oracle tokens. Cancellation, prefill
deadline, a five-second decode deadline, and token-capacity rejection recovered
correctly. The decode deadline interrupted generation after 35 tokens. Both
deployments unloaded all logical model and sequence reservations.

| Native workload | First continuation | TTFT | Mean subsequent token interval |
| --- | ---: | ---: | ---: |
| 128 prompt / 64 output | 9.680 s | 1.376 s | 0.131 s |
| 128 prompt / 256 output | 34.725 s | 0.396 s | 0.134 s |

These are observations with background applications and uncontrolled WAN
conditions, not controlled speedup or planner-acceptance measurements. The repeated
prompt was chosen for exact token capacity, not as a model-quality benchmark.

## Serving exposed a physical-fit qualification gap

The isolated profiles passed fresh prelaunch reassessment. The multi-request
serving runs subsequently reached higher RSS than those profiles captured.
Reapplying the existing conservative rule to the observed serving counters fails:

| Workload | Sampled MLX RSS peak | MLX allocator peak | Conservative requirement including headroom | Available before launch |
| --- | ---: | ---: | ---: | ---: |
| 128 / 64 | 3.506 GiB | 3.624 GiB | 9.093 GiB | 7.687 GiB |
| 128 / 256 | 4.012 GiB | 3.635 GiB | 9.662 GiB | 7.852 GiB |

The requirement is `(RSS peak + allocator peak) × 1.10 + 0.25 GiB + 1 GiB`.
It intentionally overcounts overlapping views; the table does **not** report a
9–10 GiB measured footprint. Mac pressure remained normal and no new swap-outs
were recorded during these workloads. Sampled CUDA process-GPU peaks were
4.949 GiB and 4.971 GiB respectively. Sampling can miss transients.

Functional success therefore does not establish a passing physical-fit result.
The 512/256 expansion, persistent reverse-order serving, and concurrency tests
were stopped at this gate. Smaller, independently guarded numerical stage replays
investigated correctness without qualifying those serving modes.

A supplementary read-only Mac sampler uses `proc_pid_rusage(RUSAGE_INFO_V4)` to
record OS physical footprint, lifetime footprint peak, and resident size for an
owned process. An isolated numerical probe reported approximately 3.84 GiB physical
footprint while holding its assigned weights. This is evidence for investigating
the accounting overlap, not a replacement acceptance rule. It was added after the
serving runs, so it cannot retrospectively supply their physical-footprint peaks.
Across the five instrumented numerical probes, its largest recorded lifetime
physical-footprint peak was 3.939 GiB.

## Numerical and regression checks

Three plain-text prompts were replayed through each stage order, with a prefill
and two teacher-forced decode steps per prompt. The real F16 boundary bytes from
the first native stage were consumed by the second. These were offline stage
replays, not reverse-order live RPC serving. All **18 token decisions** matched
the independent F16 oracle. Layer and full-vocabulary logit comparisons covered
the final teacher-forced step of each prompt: six snapshots total.

| Stage order | Maximum absolute logit error | Maximum relative logit L2 | Maximum relative layer L2 |
| --- | ---: | ---: | ---: |
| MLX → CUDA | 0.03125 | 0.000931 | 0.002065 |
| CUDA → MLX | 0.03125 | 0.000762 | 0.001786 |
| Declared limit | 0.25 | 0.01 | 0.015 |

The existing five CPU Qwen3 CTest entries, MLX numerical parity entry, and CUDA
numerical parity entry all passed: seven CTest entries total. No production
runtime code changed in this pass.

No resource guard fired. One isolated Mac profile had a single non-normal pressure
sample at the two-second sampling cadence; it did not reach the sustained-pressure
stop threshold. Both serving windows had normal pressure throughout their samples.
All guarded Mac phases recorded zero new swap-out bytes. These observations do not
cover checkpoint download and build preparation.

Logical request and model reservations retired after both live deployments. MLX
allocator-active bytes returned to zero; CUDA retained approximately 8.125 MiB of
allocator-active state and a larger cached pool until process exit. The workers
were then terminated, releasing those process-owned allocations. The final GPU
compute-process query was empty. No experiment or follow-up automation was left
running.

## Next decision

The [September 23 footprint soak](qwen3-4b-footprint.md) subsequently completed
three same-process deployment cycles and twelve exact 256-token continuations.
It measured a 4.576 GiB lifetime footprint peak and identified retained empty host
allocator regions after unload. Its evidence supports an explicit footprint-aware
fit policy; the production fit gate remains unchanged.

The [physical-footprint instrumentation](../milestone-5-profiling.md#process-physical-footprint)
is now implemented in native profile phases and worker metrics. These additions
do not retroactively change this run's evidence or its fit decision. Use the new
same-process load/request/unload observations to implement a justified physical-fit
rule, preserving headroom and safety allowances, and qualify it before expansion.

Only after that should we choose among quantization, runtime offloading, or more
stages for capacity. This pass establishes that resident 4B inference works; it
does not establish that offloading is necessary. Longer contexts also need a
separate assessment of the current dense-attention workspace reservation, which
grows quadratically with total reserved sequence capacity even with chunked prefill.

## Remaining scope

These experiments do not establish 32K context, arbitrary concurrent workloads,
batch-independent seeded sampling, production throughput gains, M5 WAN acceptance,
or safe repeated deployment loading in a long-lived Mac worker. Loader conversion
and runtime weight offloading remain different features.

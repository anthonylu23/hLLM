# M5.3–M5.4 qualification — 2026-09-08

M5.3 native compute/conversion profiling and M5.4 directional native transport
profiling are implemented. Both endpoint roles of Qwen3-0.6B completed paired
512-prompt/256-output timing runs on Apple MLX and NVIDIA CUDA. Native transport
qualified in both directions at both the 0.6B and 4B payload widths, with consistent
observed direct Tailscale routes.

This qualifies the profiling tools and the recorded assignments/payloads. It does
not implement measured placement, qualify all splits, or establish the full M5
15% selection criterion. Qwen3-4B-Base compute and physical fit remain unqualified.

The [machine-readable summary](milestone-5-timing/summary.json) contains component
costs, paired phase distributions, physical preflight settings, software identities
and links to compressed raw artifacts. Every archive was decompressed and validated
against its schema and content digest after writing. The
[workflow and timing attribution](../milestone-5-profiling.md) describe how to repeat
these measurements and how to consume them without double counting.

## Compute experiment

The checkpoint is `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, split at layer 14 of 28. It uses F16
execution, KV and boundary buffers, concurrency one, and 768-token capacity. The
checkpoint content digest agrees across all four profiles. Input identity is the
same deterministic synthetic stage exercise used for memory profiling: token zero
for the first stage and zero-valued F16 boundary activations for the final stage.
This exercises real weights and shapes; it is not a distributed continuation test.

Each fresh process loads its assignment once and runs seven paired cycles: two
warmups plus five measured cycles. Each pair has separate ordinary and instrumented
sequence states, with pass order alternating. Prefill covers 512 tokens; all 255
decode positions (contexts 512–766) are measured. Boundary bytes or sampled tokens
match exactly between paired passes at every step, including warmups.

MLX ran on Apple M3 Pro with MLX 0.32.2. CUDA ran on RTX 3060 Ti with LibTorch
2.13.0 and NVIDIA driver 610.57.04. Existing desktop applications and the Fedora
Java server remained running. No competing inference or memory sampler ran during
these timings. Native numerical suites additionally check nonzero boundary values
and decode outputs with timing enabled on CPU, MLX and CUDA.

| Assignment | Ordinary prefill median | Ordinary 255-decode total median | Instrumented prefill difference | Instrumented decode-total difference |
| --- | ---: | ---: | ---: | ---: |
| [MLX first](milestone-5-timing/mlx-first.json.gz) | 137.318 ms | 3,721.106 ms | +19.12% | −2.45% |
| [MLX final](milestone-5-timing/mlx-final.json.gz) | 124.295 ms | 3,899.446 ms | +2.66% | +8.13% |
| [CUDA first](milestone-5-timing/cuda-first.json.gz) | 25.638 ms | 2,015.863 ms | +0.78% | +5.09% |
| [CUDA final](milestone-5-timing/cuda-final.json.gz) | 26.419 ms | 2,215.454 ms | +0.68% | +5.46% |

The decode total is the median of five complete per-cycle sums, excluding prefill
and sequence allocation. Differences compare medians of instrumented and ordinary
phase totals. The negative observed difference is run variability; it is not
negative instrumentation work or a reusable correction factor. In particular,
MLX first-stage prefill shows substantial synchronization overhead. Future planning
must retain ordinary whole-stage baselines and validate any component-based model;
these measurements do not justify assuming zero or constant profiling overhead.

Raw artifacts retain every global layer index, endpoint component, conversion/copy
region, validation region, sequence allocation and wall-time residual. Component
sums reconcile with instrumented wall time. Warmups remain in the raw evidence and
are excluded from the summary distributions. Full-checkpoint CUDA measurements use
pageable transport; pinned CUDA conversion is covered by native process regression
checks, not by a claimed full-checkpoint pinned performance result.

## Physical preflight

CUDA used 2 GiB host and 2 GiB device admission caps, with the same host/device
headroom and extra allowances as the earlier
[memory qualification](milestone-5-memory.md).

MLX retained the 2,936,012,800-byte load/run cap and 256 MiB physical headroom. Two
attempts were rejected before loading as Mac availability changed; their
[original allowance](milestone-5-timing/rejected-mlx-first.preflight.json) and
[compute-only allowance](milestone-5-timing/rejected-run-mlx-first.preflight.json)
remain recorded. No failed load or OOM was retried.

The successful compute-only passes reserved an additional 64 MiB for the bounded
native timing records, rather than the earlier 256 MiB allowance that also covered
transport/process overhead. They run without activation RPC traffic or an external
memory sampler. Current free-plus-inactive availability recovered to 3,689,594,880
bytes for the first stage and 4,593,614,848 bytes for the final stage, and each
preflight passed before loading. This does not replace the previous memory-envelope
measurements or relax the larger target's load gate.

## Directional transport experiment

The isolated native probes use the production F16 boundary codec and
`StageExecution.Execute` framing. They send one prefill activation and 255 one-token
activations on each persistent stream, with a deterministic sampled-token reply per
activation. All timed serialization, copying and exchange runs in C++; Python
requests the run and observes its route. There is no receiver model compute.

Each direction/width has two warmup streams and five measured streams: 1,792 total
exchanges, with five prefill and 1,275 decode samples retained for measurement.
The reported RPC round trip includes receiver boundary decode/validation and the
sampled-token reply. Sender encoding is separate, and GPU conversion belongs to
the compute profiles above. No measurement halves a round trip or adds feedback
again.

| Payload scope and direction | Prefill payload | Prefill RPC median | Decode payload | Decode RPC median |
| --- | ---: | ---: | ---: | ---: |
| [0.6B MLX→CUDA](milestone-5-timing/mlx-cuda-link.json.gz) | 1 MiB | 149.807 ms | 2 KiB | 46.063 ms |
| [0.6B CUDA→MLX](milestone-5-timing/cuda-mlx-link.json.gz) | 1 MiB | 208.324 ms | 2 KiB | 46.209 ms |
| [4B width MLX→CUDA](milestone-5-timing/mlx-cuda-4b-link.json.gz) | 2.5 MiB | 254.807 ms | 5 KiB | 46.695 ms |
| [4B width CUDA→MLX](milestone-5-timing/cuda-mlx-4b-link.json.gz) | 2.5 MiB | 268.939 ms | 5 KiB | 48.362 ms |

Sender prefill-encode medians were 0.336/0.537 ms at the 0.6B width and
1.160/1.229 ms at the 4B width, for MLX→CUDA/CUDA→MLX respectively. Full exchange
medians, raw samples, serialized message/feedback sizes, channel readiness and
stream setup/teardown distributions are retained in the artifacts and summary.
The setup measurements contain the probe's initial-metadata readiness barrier;
production does not wait at that point. They must not be added blindly to TTFT.

Source-host ping/status observations bracket every run, with status sampled at
0.5-second intervals during execution. The four runs retained 168, 179, 163 and
185 path observations respectively, all on their respective consistent direct
routes. These are sampled observations, not packet-by-packet route attestations;
shorter undetected transitions remain possible. No DERP or peer-relay performance
is claimed by this report. The 4B rows qualify payload transport only: no 4B
checkpoint was loaded, and no 0.6B compute/memory result is extrapolated to it.

The transport probes use IDs `mlx-host` and `cuda-host`; the summary records their
explicit association with the stage-profile worker IDs `mlx` and `cuda`. There is
no implicit planner aliasing or measured planner integration in this change.

## Evidence and regression checks

The [native build snapshot](milestone-5-timing/build-source-identity.json) identifies
base revision `5c205f6cdba5fa5d69d18de1cff93b320dc5d403` and its source-file digests.
Each profile independently records its native executable digest, compiler and
backend/driver or gRPC-core/Protobuf identity. The
[collector source manifest](milestone-5-timing/collector-source-identity.json) records
Python sources used for final collection separately from the native build snapshot.

- 71 Python tests passed, plus Ruff, Pyright and generated-Protobuf consistency.
- All 57 local CTest entries passed, including the full CPU/MLX numerical, lifecycle,
  pipeline and memory suites. Final compute/link process checks passed after the
  collector and semantic-validation updates.
- All 60 Fedora CTest entries passed, including full CPU/CUDA mixed-pipeline and
  failure qualification. An additional CUDA compute pass exercised both pageable
  and pinned transfer modes with paired output parity.
- Artifact validation rejects missing/duplicate components, wrong contexts,
  overlapping totals, invalid payloads, direction/build/workload mismatches, stale
  measurements and unknown or changing observed paths.
- Native process tests cover explicit peer resolution, payload/traffic bounds,
  malformed tensors, deadlines, cancellation and recovery.

The first cross-machine attempt hit its connection deadline because Fedora's
Tailscale zone allowed SSH only. Subsequent measurements used a temporary,
source-address-restricted rule for the dedicated probe port. Both owned native
probe processes were stopped and the rule removed after measurement; the
[cleanup record](milestone-5-timing/cleanup.log) preserves that result. The Mac
collector uses `posix_spawn`-compatible subprocess options, avoiding fork of live
gRPC threads during path sampling.

Next: M5.5 should resolve compatible measured inputs, preserve unknown candidates,
apply memory gates and explain predicted costs. M5.6 must then compare selected
plans against an independent exhaustive sweep and evaluate the 15% criterion.

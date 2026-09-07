# Full-checkpoint and cross-machine qualification — 2026-09-06

Milestone 4's execution/transport criterion passed on the complete Qwen3-0.6B
checkpoint. MLX→CUDA and CUDA→MLX each generated 256 tokens over both direct
Tailscale and Dallas DERP connections. All four split outputs exactly matched the
independent F32 Transformers reference. Each worker held 14 of 28 layers.

[Reproduction instructions](../milestone-4.md) describe the opt-in scripts and
native diagnostic targets. [Recorded results](full-checkpoint-results.json) preserve
the numeric summaries, token hashes, memory observations, and failure outcomes.
Large reference tensors and raw logs remain in the ignored `build/cross-machine/`
directory. Source was the uncommitted `codex/mlx-worker` working tree based on
`32d43419c82a988aaa4357377fe362a5233ca9b5`.

## Refresh on merged Milestone 3 — 2026-09-07

The qualification was repeated on native/tooling commit `0893129`, based on merged
main `fbbd31f`. [Refresh results](milestone-4-refresh.json) record the independent
oracle hash, all numerical probes, partition residency, allocator snapshots,
process-memory samples, faults, and observed Tailscale paths.

All 50 Mac and 53 Linux CTest entries passed, along with 42 Python tests, Ruff,
Pyright, generated-protobuf reproducibility, and whitespace checks. The MLX F16,
CUDA F16 and CUDA F32 full-checkpoint probes again passed all nine decisions and
the declared numerical bounds. The same pinned checkpoint and independent oracle
were used. The previous CUDA-only index-177 drift investigation remains historical;
that full-model long continuation was not repeated in this refresh.

| Split run | 256-token elapsed | TTFT | F32 reference prefix |
| --- | ---: | ---: | ---: |
| MLX → CUDA, direct | 32.295 s | 0.373 s | 256/256 |
| CUDA → MLX, direct | 25.326 s | 0.388 s | 256/256 |
| MLX → CUDA, DERP | 25.534 s | 0.384 s | 256/256 |
| CUDA → MLX, DERP | 26.259 s | 0.427 s | 256/256 |

All six DERP fault cases passed: cancellation after one token, deadlines during
decode (12 and 11 tokens), and admission rejection before output. Each recovered
with the expected four reference tokens. Model and sequence reservations retired.
MLX allocator active bytes returned to zero after each unload, retaining about
64 MiB of cache. CUDA process VRAM still grew across assignments; its allocator
telemetry and stream reuse are follow-up work.

DERP(dfw) was observed before and after the relayed runs, with peer-specific UDP
drop counters increasing. The temporary table and firewall exception were removed,
the rollback timer stopped, and direct connectivity restored (50–58 ms).
Timings remain observations under application load, not controlled performance
benchmarks; Linux C++ compilation overlapped DERP qualification. RSS and NVIDIA
process usage are separate observations, not quantities to add to reservations.

## Checkpoint, machines and workload

- Model: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
  revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- `model.safetensors` SHA-256:
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
  HF checksum verification passed for all seven required assets on both machines.
  Missing ancillary repository artifacts and local HF cache metadata
  explain the CLI's missing/extra-file warnings; no model assets were missing.
- BF16 checkpoint storage; F16 native execution, KV caches and boundary tensors.
  No quantization, checkpoint reduction, random replacement, or layer pruning.
  Hidden size 1024, 28 layers, 16 attention/8 KV heads, explicit head dimension 128,
  vocabulary 151936, tied embedding/head.
- Mac: Apple M3 Pro, 18 GiB unified memory, macOS 26.5.1, AppleClang 17,
  MLX/MLX-Metal 0.32.2, Debug native build.
- Linux: Fedora 44, Ryzen 7 5800X, 16 GB host RAM, RTX 3060 Ti 8 GB,
  driver 610.57.04, CUDA toolkit 13.3, LibTorch 2.13.0, GCC 14,
  RelWithDebInfo native build.
- Independent oracle: Torch 2.13.0 / Transformers 4.57.6, eager attention,
  TF32 disabled and FP16 reduced-precision reduction disabled.
- Three short chat prompts, each generating 16 tokens; a 33-token observatory-story
  chat prompt generates 256 tokens with stop IDs disabled. The reserved context is
  289 tokens, not the model's advertised 40960-token maximum. Greedy sampling and
  the checkpoint's non-thinking chat template are used throughout.

Existing applications, including the Linux Java workload, were left running. Linux
had approximately 6.7 GiB available host memory and 129 MiB GPU memory in use before
qualification. The Mac was under existing application load. Choosing 0.6B allowed
full and split inference with usable budgets of 4 GiB unified on MLX and 2 GiB
host / 6 GiB device on CUDA. These are admission budgets with headroom, not hard RSS
or framework-allocator caps. The larger 4B target was not downloaded or inferred.

## Full-checkpoint numerical checks

Before measuring, the manual probe declared these limits: F16 last-logit maximum
absolute error ≤0.25, relative L2 ≤0.01, and worst layer relative L2 ≤0.015;
F32 limits are 0.01, 0.001, and 0.001 respectively. All nine sampled argmax decisions
must match. The probe compares all 151936 last-row logits and the final hidden row
of every layer at prefill and two teacher-forced incremental steps for each of the
three prompts. These tolerances are specific to this checkpoint/workload.

| Native backend | Max absolute logit error | Worst logit relative L2 | Worst layer relative L2 | Argmax |
| --- | ---: | ---: | ---: | --- |
| MLX F16 | 0.054744 | 0.002733 | 0.003269 | 9/9 |
| CUDA F16 | 0.071867 | 0.003671 | 0.004033 | 9/9 |
| CUDA F32 | 0.000031 | 0.00000129 | 0.00000218 | 9/9 |

All declared short-probe criteria passed. The smallest F32 top-two margin in these
nine decisions was 0.279236. Full-model resident F16 weights were 1,192,099,840 bytes.
The checkpoint's separately serialized tied head was verified byte-for-byte and
deduplicated in resident storage. The new shared-loader regression failed before
the fix and passed afterward for F32/F16/BF16 checkpoint storage.

MLX-only generation matched all 256 F32 reference tokens. CUDA-only matched the first
177 tokens, then diverged at zero-based output index 177. A diagnostic replay of the
exact shared prompt and 177 preceding tokens reproduced that differing decision:
CUDA chose token 748; F32 chose 13970. At that point the F32 top-two margin was
0.029865, versus a maximum logit error of 0.045309, relative L2 0.001546 and worst
layer relative L2 0.003119. The numeric errors remain within the declared F16 bounds,
but the exact-argmax diagnostic correctly reports failure. This is consistent with
F16 rounding changing a close greedy decision; it is not claimed as exact parity.
Transformers' own F16 generation first diverged from F32 at index 122.

The four distributed outputs nevertheless matched all 256 F32 tokens. Their different
rounding paths happen to preserve this sequence; no guarantee of identical arbitrary
F16 continuations follows. Reproduce the drift probe with
`checkpoint_reference.py --continuation-reference reference-f32.json --at-index 177`,
then the CUDA F16 native probe; its exit status 1 is the expected argmax mismatch.

## Cross-machine results

The split is explicitly fixed at layer 14, using pageable CUDA boundary staging.
Both native workers maintain the stage stream and autoregressive loop; the controller
only submits requests, observes token events, and reads control reports. Model files
are complete on disk on each machine; resident layers are partitioned.

| Placement/path | Tokens | Total seconds | TTFT seconds | F32 matching prefix |
| --- | ---: | ---: | ---: | ---: |
| MLX only | 256 | 5.689 | 0.050 | 256 |
| CUDA only | 256 | 4.372 | 0.075 | 177 |
| MLX→CUDA direct | 256 | 26.175 | 0.397 | 256 |
| CUDA→MLX direct | 256 | 27.134 | 0.463 | 256 |
| MLX→CUDA DERP | 256 | 25.506 | 0.431 | 256 |
| CUDA→MLX DERP | 256 | 27.564 | 0.409 | 256 |

Totals cover the generation RPC through terminal cleanup and control-report calls
after the first token. They exclude stage loading and tokenization. These are single
observations with warm-up from the short prompts, different native build modes,
other applications running, and uncontrolled network conditions. The slightly faster
MLX→CUDA relay observation is not evidence that relaying improves throughput.

Direct pings measured 50–61 ms. During forced relay, all before/after pings reported
`DERP(dfw)`, measuring 58–111 ms. A dedicated temporary nftables table dropped direct
Tailscale UDP to/from the Mac peer's public IPv4 address and IPv6 prefix, while leaving
DERP TCP intact. An independent 600-second rollback timer was installed first. The
drop counters increased in both directions; both full runs and the fault checks
finished before the rules were removed. Direct pings then returned at 47–68 ms.
The temporary worker-port rule was restricted to the Mac's tailnet source address.
Both rules and the rollback timer were removed, and owned worker processes stopped.

## Memory and lifecycle evidence

| Per-stage F16 payload/reservation | First stage bytes | Final stage bytes |
| --- | ---: | ---: |
| Resident weights | 751,631,360 | 751,633,408 |
| KV cache at 289 reserved tokens | 16,572,416 | 16,572,416 |
| MLX unified workspace including transport | 225,219,344 | 225,219,344 |
| CUDA device workspace | 205,091,072 | 205,091,072 |
| CUDA host workspace | 1,845,776 | 1,845,776 |

Each split worker has fewer resident weight bytes than the 1,192,099,840-byte full
F16 model. The tied embedding is intentionally duplicated between the two stages.
Completed requests returned active requests, reserved cache and workspace to zero;
unloading returned model weight reports to zero. MLX allocator active bytes also
returned to zero; cached blocks remain separate from live model payloads.

The full MLX diagnostic measured peak allocator use of 1,204,869,436 bytes, maximum
RSS 1,541,685,248 bytes and macOS peak memory footprint 1,986,774,240 bytes (1.85 GiB).
Its diagnostic traces are outside production workspace accounting. The full MLX RPC
process was sampled at a maximum RSS of 1,585,332,224 bytes. Across the full/split RPC
sequence, MLX's recorded process-wide allocator peak was 1,235,214,348 bytes.

A fresh-process direct rerun in each order confirmed all 256 tokens again and sampled
both processes every 0.5 seconds:

| Rerun | MLX maximum sampled RSS | CUDA maximum sampled RSS | CUDA process VRAM |
| --- | ---: | ---: | ---: |
| MLX→CUDA | 1,584,201,728 | 1,342,533,632 | 998,244,352 |
| CUDA→MLX, next load in same processes | 1,573,568,512 | 1,965,477,888 | 1,795,162,112 |

Sampling can miss transient peaks. RSS, NVIDIA process memory, allocator allocations,
and logical reservations describe different things and must not be added together.
The original long qualification process accumulated CUDA process VRAM from about
1.36 GiB during CUDA-only inference to 4.33 GiB after the fourth split deployment,
and 5.81 GiB after failure qualification, despite zero model/request reports after
unload. CUDA's current per-load pooled streams retain allocator/BLAS caches; its tiny
lifecycle test explicitly warms all 32 streams. This run does not establish a small
steady-state physical footprint across arbitrary full-model reloads.

A separate F32 Transformers process subsequently encountered CUDA OOM while that
idle qualification worker retained its caches. Stopping the owned worker restored
headroom; the identical reference/probe rerun succeeded. This is a recorded coexistence
limitation, not an inference failure in the split pipeline. GPU sharing and measured
placement must account for cached allocations, rather than relying on zero model
reports. No production CUDA cache/stream behavior was changed in this milestone.

## Faults and regression

Over the real DERP path, each stage order passed cancellation after the first token,
deadline expiry during decode (12 and 11 tokens respectively), and oversized-context
admission rejection before output. Every case checked reservation cleanup and then
successfully generated four fresh tokens. Final unload released model reports.
This adds real-network evidence to the broader tiny-fixture failure matrices; it does
not repeat worker-kill, malformed-frame or partition injection with the full checkpoint.

| Final regression | Result |
| --- | --- |
| Mac MLX-enabled CTest | 47/47 entries, 180.60 seconds |
| Separate Mac CPU-only CTest | 45/45 entries, 33.84 seconds |
| Linux CUDA-enabled CTest | 50/50 entries, 324.30 seconds |
| Python unit/property suite | 38 passed |
| Ruff, Pyright, generated protobuf reproducibility, diff whitespace | Passed |

CTest entries group some native and pytest cases. The added redundant-head case
accounts for the increase in common-test counts. The optional full-checkpoint probes
are not registered as ordinary CTests. No new sanitizer or Metal-validation claim
is made. The MLX wheel's previously recorded SDK/deployment-target link warnings
remain, with successful execution on the tested macOS version.

The 4B checkpoint, full model context, pinned cross-machine mode, quantization,
batching, application mTLS, and automatic measured placement remain outside this
qualification. These results establish the tested 0.6B workload and actual direct/
DERP execution paths, not arbitrary-model or maximum-context feasibility.

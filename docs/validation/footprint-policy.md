# Footprint policy implementation — September 27, 2026

Status: implemented and tested on synthetic/tiny models. **Fresh 4B policy
qualification remains pending overnight.** The user deferred heavy model work
until Mac memory is available; no 4B checkpoint was loaded in this session.

September 28 follow-up: a [fresh Mac 4B isolated profile passed](qwen3-4b-fit-20260928.md)
using the opt-in policy. Two-host serving qualification remains pending because
the CUDA device is occupied by another workload. The checks below describe the
September 27 implementation session.

September 30 follow-up (review hardening): frozen sweeps now carry the bundle's
explicit policies and refuse executors or probes that applied a different one;
explicit policies are rejected on non-MLX workers; `profile-memory` reports the
applied policy and fallback notes; an MLX-bundle test shows the conservative sum
refusing and the footprint policy admitting the same evidence through placement
and fresh activation; historical fit reports are replayed under both policies.
See the [policy contract](../milestone-5-profiling.md#mlx-footprint-policy)
and the [CPU readiness report](cpu-readiness-20260930.md).

## Behavior

`footprint-v1` is an explicit MLX option in profiling, measured bundle worker
bindings and native sweep worker configurations. `conservative-v1` remained the
default at the time of this report (it became the profiling default on October 10,
2026, after the [two-host serving qualification](qwen3-4b-serving-20261010.md)). Complete, coherent schema-1.3 memory evidence permits the maximum of OS
footprint, RSS and allocator envelopes instead of their overlapping sum. Existing
safety fractions, overhead, headroom, native reservations and configured-cap
preflight remain in force. Missing or inconsistent footprint evidence records a
fallback to the conservative policy; incomplete execution/allocator evidence keeps
the existing unknown result. CPU/CUDA formulas are unchanged.

Policy selection is retained through fresh activation and independent sweep
exclusion validation. Explicit choices participate in bundle/executor identities;
historical bundles omit the default field, and historical fit reports still round
trip. Fit output records the actual formula, requested MLX policy and fallback notes.
See [the complete policy contract](../milestone-5-profiling.md#mlx-footprint-policy).

## Reproduction tooling

- `resource_guard.py` bounds an owned process group by time, availability, Mac
  pressure and new swap-out activity. Log files are opened before spawning and
  refuse overwrite; sampler errors and signals retire the owned processes.
- `process_footprint.c` is a separate macOS sampler for the direct native worker
  PID, covering load-time control-RPC gaps.
- `reload_soak.py` uses explicit endpoints and fresh matching profiles, checks
  current fit before every deployment, requires exact reference tokens, exercises
  cancellation and reloads, and saves phase snapshots to JSONL. It preserves partial
  failure reports and does not treat isolated profile fit as serving acceptance.
- The existing reference generator now exposes plain Base-model tokenization and
  explicit GPU-layer placement instead of relying on ignored custom scripts.
- GitHub Actions defines routine Python checks and a Linux CPU build/test job.
  Native gRPC/Protobuf dependencies come from one pinned gRPC source tree.

## Checks completed locally

- **156 Python tests passed**, including footprint cross-checks, all fallback
  cases, unchanged allowances/admission/backend behavior, historical fit artifacts,
  bundle hash/activation policy propagation and policy-specific sweep exclusions.
- **Three focused CTest entries passed:** `MemoryProfile-cpu`, `MemoryProfile-mlx`
  and `MeasuredPlannerIntegration`. Tiny MLX profiles exercised both uniform and
  mixed precision with the opt-in formula; CPU kept its existing formula.
- **One additional MLX process test passed:** the new harness completed two tiny
  reload cycles with exact regression tokens and clean cancellation/unload, then
  rejected an impossible fresh headroom requirement before loading any stage.
  This uses a synthetic native CPU token baseline, not a new independent 4B oracle.
- Resource-guard tests verify pressure/swap/availability stops, refusal to overwrite
  existing evidence, timeout cleanup and cleanup on sampling exceptions.
- The independent process-footprint helper compiled with `-Wall -Wextra -Werror`
  and reported counters for an owned lightweight process.
- Ruff, Pyright, generated-protobuf consistency, lockfile and whitespace checks passed.
- Actionlint 1.7.12 accepted the CI workflow; the native dependency shell script
  passed `bash -n`. The hosted workflow and clean Linux dependency build have not
  run in this session.
- The saved September 18 independent oracle passed the new harness's checkpoint
  and token-shape validation: 128 prompt and 256 generated tokens. Reuse still
  requires fresh checkpoint verification on the two hosts at overnight launch.

All test-owned local workers were stopped after the checks. No remote model worker
was started. The [overnight backlog](../overnight-backlog.md) defines the remaining
fresh profiles, six-cycle soak, serving-fit assessment and gated workload expansions.

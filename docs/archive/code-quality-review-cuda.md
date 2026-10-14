# CUDA stack review — PRs 5–9

Reviewed implementation, code quality, admission/accounting, numerical semantics,
resource ownership, cancellation/deadline propagation, build boundaries, and test
coverage on 2026-09-06. The PRs form a stack and must land in dependency order.

## Findings and fixes

| Severity | PR | Finding | Resolution |
| --- | --- | --- | --- |
| P1 | 8 | Pinned admission caps staging at 8 MiB, but execution copied the complete boundary through it. A valid larger prefill failed after admission. | Copy successive chunks through the same allocation; synchronize before reuse. Hardware regression covers 8 MiB + 1 byte, 16 MiB + 17 bytes, and subsequent small transfers. |
| P2 | 8 | Planner checked host-resident stage memory and host transport storage independently. Both could fit separately while their sum exceeded host capacity. | Include transport storage in the host stage workspace and resulting pressure/admission calculation. Exact-fit and one-byte-short regression added. |
| P2 | 9 | Mixed deadline tests used identical application and client deadlines, allowing client-side expiration to hide incorrect server status. | Keep the client deadline five seconds later and require the native generation handler's `DEADLINE_EXCEEDED`. |
| P3 | 9 | Mixed fault tests imported other test functions and monkeypatched their module's worker constructor. This made reuse depend on hidden globals and test-module internals. | Extract shared failure contracts with an explicitly typed worker factory. Both suites invoke the same checks directly. |
| P3 | 6–9 | Documentation still described mixed execution as future work and one-stage plans as the only qualified flow. | Align backend extension and usage documentation with the implemented test coverage. |

The large-transfer regression was run against the original implementation on the
RTX 3060 Ti and failed with `boundary exceeds pinned staging capacity`; it passed
after the fix. Allocation counters additionally check that chunking retains only
the configured 8 MiB staging allocation and releases it afterward.

## Per-PR assessment

- **5 — Backend integration:** factory capabilities match executable support at that
  point in the stack; common transport is independent of CPU/Torch types. Domain
  arithmetic checks overflow, pinned bytes count within host usage, and admission
  occurs before sequence allocation. No blocking finding specific to this PR.
- **6 — CUDA execution:** shared checkpoint validation avoids a duplicate model
  loader and uploads one tensor at a time. Inspected GQA, Q/K normalization, RoPE,
  causal masking, cache position checks, precision conversion, sampling, and stream
  completion/error paths. No additional blocking implementation finding.
- **7 — Mixed qualification:** covers both orders, all tiny-fixture splits, storage
  formats, tied heads, repeated requests, stop IDs, long decode, and the CLI path.
  Process-launcher extraction is useful reuse; no blocking finding specific to it.
- **8 — Pinned transfers:** the two accounting/transfer findings above required fixes
  before merge. Bounded staging now supports payloads above its allocation size.
- **9 — Failure qualification:** preserves stream draining and poisoned-buffer
  rejection when incorporating chunked transfers. Strengthened deadline validation
  and removed fragile test-to-test coupling.

## Quality and remaining limits

The primary “slop code” concern was hidden test coupling, along with stale status
claims. The backend factory, shared dense loader, opaque state, and separate Torch
translation units have concrete responsibilities and do not warrant a broad rewrite.
The dense CUDA attention path is intentionally conservative and quadratic in context;
repeated string lookups and per-operation validation are performance work, not a
claim of optimized serving. Avoid adding another abstraction layer before profiling.

Memory reports are payload/reservation estimates, not physical RSS or allocator
limits. Previously documented framework/context retention remains relevant. These
checks use tiny models and loopback transport; they do not establish full-checkpoint
fit, throughput, arbitrary-workload memory bounds, or cross-machine behavior.

## Validation

All 37 Python unit tests, 43 macOS CPU-only CTest entries, and 48 Linux
CUDA-enabled CTest entries passed. Ruff, Pyright, generated-binding checks, and
whitespace checks passed. See the [mixed qualification report](validation/mixed-cpu-cuda.md)
for case counts, environment details, and the strengthened deadline checks.

## Next steps

Assess physical memory fit for a concrete full-checkpoint workload, including CPU
F32 weights, dense attention workspace, and measured CUDA/framework overhead.
Profile before selecting attention or transfer optimizations. Cross-machine execution
and MLX remain the later milestones already described in the project specification.

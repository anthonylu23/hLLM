# Code quality review

The review covered handwritten Python preparation, domain models, planning, wire
mapping and CLI code, plus native buffers, numeric conversion, tensor envelopes,
Safetensors, CPU kernels, worker control, build files, and tests. Generated bindings
were checked for reproducibility rather than edited manually.

## Fixed findings

- **P1 — KV cache allocation overflow:** multiplying three unchecked dimensions could
  wrap the allocation size and allow later out-of-bounds writes. Validate dimensions
  and both products before allocating either cache vector.
- **P1 — load retries discard reservations:** a repeated `LoadStage` replaced the
  deployment state and emptied its active request set. Matching retries now preserve
  state; changed plan digests or stage indices require unloading first.
- **P2 — BF16 NaN corruption:** rounding certain NaN bit patterns produced infinity
  or zero. Preserve the sign and force a nonzero quiet NaN payload before rounding.
- **P2 — ambiguous shard ownership:** manifest preparation silently overwrote duplicate
  tensor names across shards. Reject duplicates before building the manifest.

Removed an unused JSON type alias, an unused digest wrapper, and stored weight byte
accounting that was never read. Removed the split-pipeline test that executed identical
calls on both sides of its assertion; it could not establish transport correctness.
The incremental-versus-one-shot causal execution test remains.

## Validation

Regression coverage exercises allocation overflow, NaN payload extremes, reservation
preservation on load retries, changed deployment digests, and duplicate shard tensors.
Validation on 2026-09-05: all 21 Python tests and 26 normal native tests passed,
as did Ruff, Pyright, and binding reproducibility. The ASan/UBSan executable built
but hung before entering the tests, so sanitizer validation is incomplete. A macOS
process sample showed recursive AddressSanitizer initialization waiting in
`StaticSpinMutex::LockSlow` while initializing shadow memory; CMake test discovery
therefore timed out. Recheck with a working sanitizer toolchain before claiming
sanitizer coverage. Use the commands in the README to reproduce the suites.

## Next steps and current limits

The worker control service currently validates selected tensor metadata and tracks one
request reservation. It does not construct executable stages, allocate request KV
caches, or implement execution RPCs. Complete those features with shape/architecture
validation and real memory admission before treating a successful load or reservation
as execution readiness. Manifest digests are identifiers here, not verified signatures.

Add a real transport integration test when execution RPCs exist, comparing received
activations and final output to the CPU reference. GPU backends and measured memory
and performance qualification remain future milestones.

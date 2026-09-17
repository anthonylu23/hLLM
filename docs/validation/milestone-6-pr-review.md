# Milestone 6 PR review fixes — 2026-09-17

This follow-up addresses the actionable findings from the [PR #16 audit](https://github.com/anthonylu23/hLLM/pull/16#issuecomment-5702752849).
The earlier [full-checkpoint sweep](milestone-6-sweep.md) remains historical evidence
for its recorded binaries; its 4B, WAN, sampling-reproduction and sanitizer limits
remain in force.

## Changes

- **Scheduler ownership under allocation failure:** claiming batch members now uses
  a fixed-capacity array. An allocation failure cannot strand a ticket between
  removal from its queue and insertion into the dispatch. Error construction for
  expired/cancelled members is caught, and expiry retains its deadline category.
  Input/output allocations remain inside the dispatch failure handler. Result
  delivery uses move construction with a compile-time non-throwing requirement.
- **Streaming retention:** response probability records and text pieces accumulate
  only for non-streaming responses. Incremental decoding still retains the history
  it needs for final verification; this is not a claim of constant total memory
  regardless of context length. An unreachable exception handler was removed.
- **Focused coverage:** an isolated allocation-fault executable checks claimed-member
  retirement and subsequent dispatch recovery. A weak-reference test checks release
  of emitted streaming records/fragments and preservation of non-stream output.
  HTTP tests cover default EOS, explicit stop IDs, disabled EOS, cross-token string
  stops, finish reasons and consumed-token usage in both response modes. Forced
  shutdown asserts client-visible failure and native cleanup.
- **Scope of sampling checks:** tiny-fixture token equality remains a regression
  assertion, explicitly distinguished from full-model reproducibility. Ragged-batch
  backend tests now run both greedy and positive-temperature cases.
- **Usage documentation:** added the missing output default, request-size/message/
  stop-ID bounds and deadline ceiling to the serving instructions.

## Regression checks

Both new regressions were checked against the original PR code:

- The allocation test fails because a claimed caller never retires; with the fix,
  callers finish and a subsequent dispatch succeeds. Additional injected allocation
  positions exercise input preparation and output collection.
- The retention test fails for the original streaming handler and passes after the
  fix; its non-stream case verifies complete text and probability output.

The first shutdown assertion expected only an interrupted HTTP transport. Execution
showed that forced shutdown can instead deliver an SSE error followed by `[DONE]`.
The corrected test accepts either failure outcome, rejects successful finish reasons,
checks the queued request returns a server error, and verifies native unload. An
initial broad run had already loaded the older assertion; the corrected suite was
rerun afterward.

## Validation

The checks below cover the changes and their surrounding runtime paths.

- **All 83 CPU/MLX CTest entries passed across the broad run and corrected rerun.**
  The broad run passed 82 entries; its CPU integration entry had loaded the initial
  shutdown expectation described above. The corrected CPU integration and six
  scheduler/allocation entries then passed together. MLX numerical, mixed-pipeline
  and concurrent-serving checks passed.
- **86 CPU/CUDA CTest entries passed** in the Linux broad run, including numerical,
  mixed-pipeline, failure qualification, concurrency and allocation-fault checks.
- **138 focused Python tests passed**, including the new retention and HTTP stop cases.
- **6 scheduler/allocation CTest entries passed on Linux with full ASan/UBSan/leak
  and container-annotation checks enabled.**
- **72 native tests and 19 serving/chunking/harness/cleanup cases passed under
  ASan/UBSan with only `detect_container_overflow=0`.** This retains the previously
  documented prebuilt-Protobuf annotation limitation; it is not unrestricted
  sanitizer coverage of that dependency.
- Ruff, Pyright, generated protobuf consistency, lockfile and diff checks passed.

Raw build/test logs and negative regression checks are retained under ignored
`build/pr16-review/` locally and `build/pr16-fix-*` in the isolated Linux validation
checkout. This follow-up does not repeat the full-checkpoint 15-minute soaks or claim
new 4B fit, WAN acceptance, performance gains or batch-independent seeded sampling.

## Follow-ups

Conservative CUDA loading admission, bounded/streamed soak evidence storage, and
additional direct-backend API preconditions remain separate improvements. No changes
to runtime weight streaming/offloading or the outstanding milestone gates are included.

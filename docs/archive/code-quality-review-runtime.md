# Runtime and controller review

Reviewed the worker services, backend contract, shared loader, memory accounting,
tensor envelopes, Safetensors parsing, CPU kernels, the Python preparation, planner,
controller and CLI code, and both test harnesses on 2026-09-06. The audit followed the
CUDA stack review and covered everything on `main` at that point.

## Findings and fixes

| Severity | Area | Finding | Resolution |
| --- | --- | --- | --- |
| P2 | prepare / loader | A tied checkpoint that still ships `lm_head.weight` prepared and planned cleanly but failed every worker's tensor ownership check at load time, after the plan had been accepted. | Treat the record as redundant: the loader ignores it when the config is tied and the planner no longer budgets it. Regression at unit, native loader, and two-worker process level, asserting identical tokens with and without the copy. |
| P2 | prepare | `llama.v1` accepted `rope_scaling` although workers reject any scaled RoPE, so preparation produced manifests the runtime could not execute. | Reject during preparation with the same rule the Qwen3 adapter already applied; remove the dead parser. |
| P2 | transport / planner | Prefill crosses a stage boundary as one message capped at 8 MiB. A prompt within the model context could still fail inside the first worker, nothing upstream knew the ceiling, and the cap also applied to single-stage deployments that never cross a boundary. | Expose the limit to Python (`MAXIMUM_RPC_BYTES`, `maximum_boundary_tokens`), reject two-stage candidates with `BOUNDARY_PAYLOAD_EXCEEDED`, validate in the controller before issuing a call, and scope the native check to split deployments. |
| P3 | services | Wire error codes and gRPC statuses were inferred from whichever standard exception type escaped a backend (`length_error`, `invalid_argument`, or a catch-all). The contract was implicit and easy to break from backend code. | Introduce `hllm::runtime::Error` with an explicit `ErrorCode`; the loader, accounting, envelopes, Safetensors, CPU and CUDA stages and both services throw it. Uncategorized exceptions report as `BACKEND_ERROR` / `INTERNAL`. |
| P3 | services | An already-expired deadline was rejected as an invalid request, indistinguishable from a malformed one; the controller's transport deadline coincided with the application deadline, so clients synthesized their own `DEADLINE_EXCEEDED`. | Report expired deadlines as `DEADLINE_EXCEEDED` from reservation and both execution RPCs; the controller's transport timeout trails the application deadline by two seconds. |
| P3 | services | The watchdog kept polling after a request completed, so a cancellation arriving between lease release and the final acknowledgment could turn a finished request into `CANCELLED`. | Stop and join the watchdog once the request is terminal, before terminal events are written. |
| P3 | tests | The process harness started workers with an undrained stderr pipe: a chatty worker could block, and a worker that exited during startup surfaced only as a 30-second readiness timeout. | Log stderr per worker under the test root, poll the process while waiting for the channel, and fail fast with the captured output. |

## Quality assessment

The split between preparation, planning, wire mapping, and native execution is sound
and no component warrants a rewrite. The recurring theme was gates applied in the
wrong place: checks the worker enforced that the planner or preparer did not know
about. Those are now enforced once, upstream, with the worker check retained as a
defensive backstop. Error categorization was the other systemic weakness; the typed
error makes the mapping explicit rather than relying on standard exception hierarchies.

## Remaining limits

- A stage stream that closes before `Terminate` still reports `INTERNAL`; `ABORTED`
  would be more precise. Both process suites pin the current status and the mixed
  CPU/CUDA suite could not be run for this review, so the status was left unchanged.
- Chunked prefill remains future work; the boundary ceiling is enforced, not lifted.
- `cpp/src/cuda/stage.cpp` and its test received exception-type substitutions only.
  They were not compiled here and should be exercised on a CUDA host.
- Memory reports remain payload/reservation estimates; see the CUDA review for the
  physical-fit caveats, which still apply.

## Validation

Development and ASan/UBSan native builds passed all 45 CTest entries, including the CPU
pipeline process suite. All Python unit and integration tests, Ruff (check and format),
Pyright, and the generated-binding check passed. Each change was verified as a
separate commit before the next was staged.

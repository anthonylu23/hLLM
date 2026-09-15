# PR #15 review fixes — 2026-09-14

This follow-up implements the four confirmed bugs from the
[comment audit](pr15-comment-audit.md), adds downstream restart coverage, and
corrects link-probe diagnostics. The changes are on `codex/m5-profiles-memory`,
following reviewed commit `ff0ad9242fe84e1744358f59675335092fb5f23a`.
M5 WAN acceptance remains deferred.

## Implemented behavior and regression coverage

| Area | Result | Regression coverage |
| --- | --- | --- |
| Interrupted sweep | A fresh interruption saves the attempt and summary, then stops that invocation. Resume validates and retains the saved attempt and proceeds to remaining jobs. | Interrupt during a reference or candidate; bounded resume; full resume; byte-for-byte retention of saved results/evidence; no duplicate execution. A failed candidate remains incomplete; a failed reference keeps the stability gate from passing. |
| CLI entry point | The module invokes Typer after registering every command. | Subprocess checks for all eleven commands under the module and installed entry point, plus help dispatch for the three previously missing commands. |
| Completion events | A final deadline/cancellation check commits the outcome before sequence cleanup. Usage and terminal writes follow release without reapplying the generation deadline. | A private backend delays sequence destruction past the application deadline; exactly one token, usage event and completed terminal event are delivered, reservations are clean, and the next request/unload succeeds. Existing cancellation, slow compute unwind and blocked client-write tests remain required. |
| Plan errors | Invalid versioned identities, incompatible schemas and hash mismatches report `ERROR_CODE_INCOMPATIBLE_WORKER`. The feasibility-schema diagnostic is separate from measured-identity diagnostics. | Six native rejection scenarios, unloaded state after rejection, subsequent valid load, and a wire-code assertion in the native measured-plan process test. |
| Downstream restart | Existing cached-channel and fail-fast serving policies are retained. | Native cached-channel connection, peer shutdown, transient failure and recovery without replacement. A real two-worker generation test kills/restarts/reloads the downstream stage twice while retaining the upstream process/deployment, then verifies exact tokens, reservations, controller-channel recovery and final unload. |
| Link diagnostics | Source and target contention report `RESOURCE_EXHAUSTED`; bad inputs remain `INVALID_ARGUMENT`; downstream RPC status/details are preserved; malformed/truncated replies report `DATA_LOSS`; local application deadlines report `DEADLINE_EXCEEDED`. | Native source/target contention and half-close tests; absent-peer application timeout; injected peer identity-RPC denial, transport failure, permission denial, missing/malformed feedback, missing/bad acknowledgment, stalled exchange and explicit cancellation; successful source reuse after each injected failure. |

The completion fix separates ordinary event emission (guarded I/O plus a running
check) from post-completion delivery. It reuses `ActiveIo` instead of the duplicate
inline guard. The client transport can still fail independently; this is not a
guarantee that terminal events can be delivered after a client disconnect or its
own RPC deadline.

The link probe's source watchdog cancels downstream I/O and lets the handler
return the precise application-deadline status. Direct target stream I/O retains
the server transport cancellation fallback so blocked synchronous operations
remain bounded. Failed stream operations inspect `Finish()` to preserve the peer
status; a successful transport ending before the expected protocol response is
classified as data loss. Neither failure is substituted for a timing sample.

## Restart policy and limits

Serving requests still fail fast during a transient connection failure. The
regressions use bounded retries after explicit downstream stage reload; no
automatic generation replay, keepalive tuning, or reconnect-backoff changes were
added. The controller has its own control channels and performs best-effort unload.
The process test waits for those channels to recover as well as the native serving
channel before verifying context-exit cleanup. Restarting the worker does not
restore its in-memory deployment automatically.

These local CPU tests establish recovery of the exercised workflow; they do not
promise a ten-second recovery bound on every network or qualify the NYC–Texas WAN.

## Validation

- Full Python suite: **105 passed**.
- Ruff lint and formatting: clean. Full Pyright: **0 errors, 0 warnings**.
- Generated protobuf bindings: current (`scripts/generate_proto.py --check`).
- Focused link-probe and process-restart suite: **15 passed**.
- Full rebuilt native CTest suite: **69/69 passed** in 240.36 seconds, including
  native runtime regressions, CPU pipeline integration, MLX numerical and
  failure/recovery integration, CPU/MLX memory and compute profiling, link profiling
  and measured-planner integration.

Reproduction commands:

```bash
uv run pytest tests/python -q
uv run ruff check .
uv run ruff format --check .
uv run pyright --pythonpath .venv/bin/python
uv run python scripts/generate_proto.py --check
uv run cmake --build build/native/m5-stage-deadline -j 4
uv run ctest --test-dir build/native/m5-stage-deadline --output-on-failure
```

The local CTest evidence is retained at
`build/validation/pr15-audit/ctest-fixes.log`. The native build directory is the
existing macOS configuration with CPU and MLX enabled.

The native build uses the existing strict warning flags, including conversion and
sign-conversion warnings as errors. MLX linking emits existing SDK-path/deployment
target warnings; linking succeeds. Full-checkpoint CUDA/WAN acceptance is outside
this fix validation.

## Workflow changes and next steps

The README and placement/profiling workflows now document interrupted resume,
matching CLI entry points, completion semantics, restart behavior, serialized probe
measurements and the new diagnostic codes. The CLI's two existing formatting
differences were also normalized.

Rebuild/deploy the updated workers and link probes before relying on these fixes.
Changed serving/probe executables and Python package sources change qualification
identities, so retain historical evidence and collect fresh matching profiles,
selection and sweep inputs when qualification resumes. An interrupted sweep can
continue collecting evidence, but its retained failed attempt cannot be erased to
obtain a pass; start a new frozen sweep after correcting the cause.

Hash-helper consolidation, profile caching, persistent F32 output-head storage,
probe target consolidation and additional historical F16 qualification were not
part of this implementation. No acceptance thresholds were relaxed.

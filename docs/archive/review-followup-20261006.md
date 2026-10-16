# Review follow-up — October 6, 2026

The requested CI and policy hardening was already merged when this session began:
PRs #19 and #20 landed through #17 and #18. Current `main` is `1571eea` and its
[hosted run](https://github.com/anthonylu23/hLLM/actions/runs/36797199218) passed both
Python and CPU jobs. The native dependency cache exists. PR #1 is closed; its old
annotation change was not reapplied because current tied-head accounting already
matches native loading.

Verified merged fixes include ANSI-independent CLI help checks, cache saving after
successful dependency builds, removal of duplicate feature-branch push checks,
longer native test budgets, sweep policy binding and probe mismatch rejection,
meaningful MLX placement tests, cancellation-prefix evidence, preflight skip
classification, guard exit-record preservation, CLI policy reporting and historical
fit replay. The conservative memory policy remains the default.

## Remaining refusal-test gaps closed

The invalid-evidence tests now intercept deployment entry and fail if any stage
load is attempted before rejecting stale/future profiles, changed binary identity,
changed capacity or workload. Previously they only inspected after-load log records,
which could miss a deployment started before the fit check.

A new real two-worker CPU test keeps an existing deployment loaded, invokes the
rehearsal, and verifies refusal before fit assessment or new deployment entry. It
checks the failed report, retained weights and an exact continuation through the
original session after refusal, then verifies normal owner-driven unload. It uses
the real bounded 15-second busy-worker wait.

## Validation

- 169 Python unit tests passed on the merged code.
- The CLI help regression passed with `GITHUB_ACTIONS=true`.
- Ruff, Pyright, generated-protocol consistency and whitespace checks passed.
- The targeted CPU worker/profiler rebuild succeeded with one compiler job.
- Guarded CPU rehearsal: **9 passed in 41.30 seconds**; guard exited zero.
  Evidence: `build/cpu-rehearsal/20261006-161318-5u49eeyk/` (JUnit, inputs, profiles,
  worker logs, reports and guard samples).

The October 6 [hosted run](https://github.com/anthonylu23/hLLM/actions/runs/37494224305)
passed 169 Python tests and all 81 CPU CTests, including `CpuReloadRehearsal` in
40.07 seconds. The dependency cache was restored successfully.

The October 2 backlog/readiness edits were initially preserved separately. They
are included in a separate documentation commit in PR #21 for the October 8
repository synchronization.

## Next steps

Review the refusal tests and accompanying backlog updates. GPU qualification still needs
fresh profiles and the two-host baseline soak; these CPU tests do not qualify 4B
capacity. Steady-clock ordering remains a documented lower-priority follow-up;
backward wall-clock observations currently cause a safe conservative fallback.

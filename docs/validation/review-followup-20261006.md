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

Existing uncommitted October 2 updates in the overnight backlog and September 30
readiness report were preserved separately from this follow-up.

## Next steps

Review this test-only follow-up and its hosted checks. GPU qualification still needs
fresh profiles and the two-host baseline soak; these CPU tests do not qualify 4B
capacity. Steady-clock ordering remains a documented lower-priority follow-up;
backward wall-clock observations currently cause a safe conservative fallback.

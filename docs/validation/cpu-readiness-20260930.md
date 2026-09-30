# GPU-independent readiness work — September 30, 2026

The existing telemetry/policy changes were reviewed and packaged, CPU failure-path
coverage and a guarded rehearsal were added, and hosted CI was started. No full
model or CUDA job was launched. The conservative MLX fit policy remains the default;
fresh two-host 4B qualification is still pending GPU availability.

## Review and changes

- Reviewed OS counter presence/lifetime semantics, schema-1.3 validation and historical
  digest preservation, explicit policy binding through placement/activation/sweep,
  unchanged physical allowances, and guard/harness process ownership.
- Fixed a harness reporting gap: a failed generation now retains partial tokens,
  elapsed time and its error. After session teardown, surviving-worker memory and
  unreachable-worker errors are saved separately. Failure reports do not certify
  cleanup of a lost worker. Atomic report updates use unique temporary files.
- Added explicit CPU/F32 rehearsal mode with labeled synthetic reference provenance
  and a separate report scope. Default F16 qualification is unchanged. The same CPU
  implementation supplies rehearsal tokens; these are not an independent oracle.
- Added a reusable 180-second guarded runner and `CpuReloadRehearsal` CTest entry.
  Each invocation preserves a fresh evidence directory. A resource-guard refusal
  to start exits 77, which the CTest entry declares as a skip; any other preflight
  or wrapper exception produces a machine-readable failure result with exit code 125.
- Added a [concurrency-two design](../concurrency-qualification.md) covering shared
  weights, independent KV reservations, per-worker overlap, combined physical fit,
  cancellation and immutable evidence. No concurrency-two implementation or capacity
  acceptance is claimed; `ProfileKey` still rejects concurrency greater than one.

## Local validation

- Full Python suite: **160 passed** before the final preflight-result regression was
  added. The final targeted validation-tools suite: **18 passed**, including that
  new test and the four real process-tree interruption cases. Ruff, Pyright,
  generated-protobuf consistency, lockfile and whitespace checks passed.
- Guard tests exercise TERM, INT, HUP and a leader that exits while its descendant
  ignores TERM. Timeout, sampling failure and evidence-overwrite tests remain.
- Initial guarded CPU rehearsal: **8 passed in 22.14 seconds**. Two CPU stages
  completed two reload cycles, two exact requests per cycle, cancellation and unload.
  Fault cases rejected stale/future profiles, changed binary/cap/workload, mismatched
  tokens and a lost worker. Worker-loss evidence retained one token and confirmed
  surviving-worker release. Guard exited zero, minimum Mac availability was
  **3.85 GiB**, pressure stayed normal, and no new swap-outs were observed.
- Full serial CPU CTest selection: **80 passed; the new rehearsal entry was blocked
  at guard preflight**. macOS reported pressure level 2, despite 4.37 GiB available.
  No rehearsal worker launched in that attempt. This is a resource refusal, not a
  passing 81-test run. The refusal is preserved; no pressure threshold was relaxed.
- The default-mode/F32 rejection assertions added after the initial rehearsal still
  require a fresh process run (hosted CI includes them). Other final wrapper changes
  were exercised by the targeted unit suite.

Local raw evidence:

- `build/cpu-rehearsal/20260930-044938-pma08hha/`: successful guard, JUnit, per-case
  profiles/inputs/worker logs/reports and observation streams.
- `build/cpu-rehearsal/20260930-045507-_kltd3gg/`: rejected Mac-pressure preflight.
- `build/native/m6-mlx/Testing/Temporary/LastTest.log`: full serial CPU CTest run.

The native worker/profiler sources did not change in this session; a targeted build
of their current checkout completed with no native compilation needed. Local test
workers were cleaned up. No work was run on the occupied remote GPU host.

## Review packaging and hosted CI

CI is isolated in draft [PR #17](https://github.com/anthonylu23/hLLM/pull/17), based
on `main`. The remaining work is grouped into separate profiling/policy, validation
harness, and documentation/evidence commits on `codex/4b-qualification`.

The [initial hosted CI run](https://github.com/anthonylu23/hLLM/actions/runs/36670443273)
passed Python checks; the native dependency/build/test job was still running when
this report was first written. The isolated Fedora result from September 28 remains
historical evidence for that recipe.

**Hosted results (completed later on September 30).** Both hosted runs finished red
on one test. On `codex/cpu-ci` the native job passed 77 of 78 CTest entries; on this
branch's head (`6b06e1e`, [run 36671209116](https://github.com/anthonylu23/hLLM/actions/runs/36671209116))
it passed 80 of 81, including **`CpuReloadRehearsal` in 22.07 seconds**, which is the
hosted execution of the default-mode/F32 rejection assertions. The single failure in
both runs was `CpuPipelineIntegration`: `test_module_cli_registers_serve` compared
`serve --help` text that Typer renders with terminal escape codes under
`GITHUB_ACTIONS`. The native dependency build took about 48 of the job's 52 minutes
and its cache was never saved because the job failed. These are fixed in the stacked
CI PR [#19](https://github.com/anthonylu23/hLLM/pull/19) (escape-free comparison,
cache saved after the dependency build, pull-request-only branch triggers, 90-minute
job and 300-second CTest budgets).

**Hardening after review.** The stacked hardening branch adds: frozen sweeps record
the bundle's explicit MLX fit policies and refuse an executor or probe that applied a
different one; explicit policies are rejected on CPU/CUDA workers; `profile-memory`
prints the applied policy and fallback notes; the cancellation probe keeps its
observed prefix in failed reports; a guard refusal is a CTest skip rather than a
failure; and historical-evidence tests replay the fit formula instead of passing
vacuously. See the [footprint policy notes](footprint-policy.md).

## Next steps

1. Land the stacked CI fixes ([PR #19](https://github.com/anthonylu23/hLLM/pull/19))
   and confirm a green hosted run with a populated native dependency cache.
2. Review the qualification draft separately; preserve its conservative default.
3. Use the hosted `CpuReloadRehearsal` result as the rehearsal evidence; rerun locally
   only when Mac pressure is normal. A guard refusal now reports as a skip.
4. When both hosts have capacity, refresh the profiles and run the six-cycle baseline
   serving soak in the [overnight backlog](../overnight-backlog.md). September 28
   profiles are historical and exceed the 24-hour freshness limit.
5. Keep larger contexts, reverse order and concurrency-two execution as follow-ups.

# PR #15 comment audit — 2026-09-14

Audited all ten inline comments and the review summary attributed by the user to
Fable 5.1, posted through `cursor[bot]`, on PR #15 at
`ff0ad9242fe84e1744358f59675335092fb5f23a`. The local checkout matches that commit.
This is an audit, not an implementation or M5 acceptance qualification.

The subsequent [implementation and validation](pr15-review-fixes.md) addresses
the four confirmed bugs, restart coverage and link-probe diagnostics. Findings
and test results below describe the original reviewed commit.

Four concrete bugs hold up: interrupted sweep resume, module CLI registration,
post-generation terminal delivery, and invalid-plan error classification.
Other comments identify useful follow-up work, but several overstate the behavior
or propose changes that would weaken existing guarantees.

## Findings and dispositions

### 1. Interrupted sweep cannot resume — confirmed, P1

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008365871).
`qualification/sweep.py:447` checks `detail.startswith("interrupted")` after both
new execution and loading a persisted result. `qualification/native.py` emits
exactly that prefix for `KeyboardInterrupt`.

Using the existing sweep fixtures, an interruption on job two produced:

| Invocation | Newly launched jobs | Decision |
| --- | ---: | --- |
| Initial | 2 | incomplete |
| Resume 1 | 0 | incomplete |
| Resume 2 | 0 | incomplete |

This blocks the advertised resume workflow for potentially expensive sweeps.
Limit the immediate interruption return to a result created by this invocation,
and add an interrupt/resume regression test.

**Qualification:** skipping the persisted interruption permits remaining work to
run; it does not make the sweep eligible to pass. Its immutable `unknown` result
still prevents complete coverage. Preserve that failure. A successful later
qualification needs a new sweep or an explicitly designed immutable retry policy,
not deletion or relabeling of the failed attempt.

### 2. Three commands missing under module execution — confirmed, P2

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008367015).
`cli.py:340` invokes `app()` before the final three command decorators run.
Reproduced eight commands under `python -m hllm_control.cli --help` versus eleven
under `hllm --help`. The missing commands are `qualify-sweep`,
`seal-profile-bundle`, and `freeze-sweep`.

Move the module entry block to the end. **Correction to the suggested test:**
`CliRunner` importing `app` would not catch this particular bug, because import
skips the `__main__` block and completes registration. Test the module in a
subprocess as well as the installed entry point.

### 3. Cached channel reconnect behavior — confirmed risk, regression unproven

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008369594).
`execution_service.cpp:17` retains one channel per deployment. Its arguments only
set message sizes; the downstream RPC does not enable wait-for-ready.

A native experiment using `LoadedDeployment::downstream_channel()` established a
connection, stopped the peer, forced `TRANSIENT_FAILURE`, then restarted the peer
on the same port. An immediate RPC on the cached channel returned `UNAVAILABLE`;
the same channel recovered within the five-second test bound.

**Correction:** an immediate RPC on a newly created channel with the same target
and arguments also returned `UNAVAILABLE`. Fresh C++ channel objects do not
necessarily isolate the underlying gRPC connection state. This experiment proves
the recovery window exists, not that the previous per-request implementation
always avoided it. It is not an old-versus-new production benchmark.

The documented defaults include exponential reconnect backoff with a 120-second
maximum; that is not a measured 120-second outage from one transient blip.
Without wait-for-ready, an RPC started in transient failure fails immediately.
See [gRPC backoff](https://grpc.github.io/grpc/core/md_doc_connection-backoff.html)
and [wait-for-ready semantics](https://grpc.io/docs/guides/wait-for-ready/).

Add a production generation restart/recovery test and define the desired deadline
and reconnect policy. Do not treat smaller backoff or keepalive settings as an
established fix for the WAN spikes. Packet loss alone does not prove a channel
entered transient failure, and a restarted worker also needs deployment reload.

### 4. Completed generation can lose its terminal event — confirmed, P2

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008371468).
`GenerationService::Generate` stops the watchdog and releases the reservation,
then sends usage and terminal events through `emit`, whose `check_running()`
still consults the application deadline. `Execute`'s final acknowledgment does
not perform this check.

Reproduced deterministically against unchanged production code with a private
test backend: one output token, a 300 ms application deadline, a 500 ms sequence
destructor, and a separate five-second client RPC deadline. The client received
one token, zero usage events, zero terminal events, and gRPC status 4
(`DEADLINE_EXCEEDED`). Reservations were clean afterward. The delay occurs in
`control_.release()` after `watchdog.stop()`; no production timing hook was added.

Make the terminal-outcome boundary explicit and avoid rechecking the generation
deadline after committing that outcome. Retain transport write-error handling.
This does not imply delivery can succeed after the client's own RPC deadline.
The duplicate inline `Writing` guard is a valid, separate cleanup nit.

### 5. Invalid measured plans reported as backend failures — confirmed, P2

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008373696).
Three `std::invalid_argument` paths in `validate_plan` fall through to
`LoadStage`'s generic backend-error handler. A native test of missing measured
identities returned `accepted=false` with `ERROR_CODE_BACKEND_ERROR`.

The plan is safely rejected; this is incorrect diagnostics/error categorization,
not an admission bypass. Use the appropriate typed request/compatibility error
and assert its wire code. The feasibility-schema error text also conflates schema
version and identity problems.

### 6. Plan canonicalization duplication — maintenance concern; Unicode claim rejected

This is the design note in comment 5 and item 6 of the summary. The hand-built
planner payload, model serialization, and C++ reconstruction must evolve together.
A common Python helper and cross-language golden vectors would reduce drift.

However, the claimed non-ASCII mismatch is not present: Python uses
`ensure_ascii=False`, and nlohmann `dump()` also defaults to `ensure_ascii=false`.
See the [nlohmann dump API](https://json.nlohmann.me/api/basic_json/dump/).
A Python-hashed measured plan containing accented text, Japanese, Chinese, an
emoji, and U+2028 in identity strings was accepted by the actual C++ worker.

A future divergent digest would be rejected by the worker's existing hash check;
it would not silently activate a different plan. Existing native process tests
already cover a normal measured hash and rejection after digest-bearing data is
changed, although broader golden vectors remain useful.

### 7. Link probe serialization and errors — partly confirmed

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008375232).
One `active_` flag covers `QualifyLink` and `Execute`, so the same probe cannot
serve concurrent source and target work. Document the single-measurement
constraint. Concurrent opposite-direction measurements would also introduce
cross-traffic, so supporting them is not automatically desirable for isolated
profiles.

Actual native RPC results while holding a target stream open were:

| Operation | Returned status and detail |
| --- | --- |
| Start qualification on the busy probe | `FAILED_PRECONDITION: probe already active` |
| Qualify toward the busy target | `INVALID_ARGUMENT: activation exchange failed` |

The inline comment reverses the source-side activation-failure classification.
The underlying complaint is valid: `require()` maps failed transport exchanges
to invalid arguments and loses the underlying stream status. Classify input
errors, contention, cancellation, deadlines, and transport failures separately.
This diagnostic issue is P2; concurrency support itself is a design decision.

### 8. Disk bundle parsing — real repeated work, overstated complexity

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008376635).
`profiles_for()` filters `ProfileReference.assignment` **before** loading a file.
It does not parse every artifact for each candidate. Instrumenting the existing
fixture yielded 48 artifact loads for six candidates and 24 profiles: each
artifact was loaded twice during planning, excluding initial binding.

There is still repeated assignment parsing for memory and compute selection, and
activation does three whole-bundle traversals plus selected-stage evaluation.
Scanning reference metadata has a candidate-by-profile cost; full JSON/Pydantic
parsing does not have the claimed all-profiles-per-candidate behavior.

**Do not apply the proposed cache unchanged.**
`test_disk_bundle_equivalence_and_integrity` explicitly requires an already-open
bundle to reject artifacts tampered with afterward. A digest-keyed per-open cache
would bypass that check. The validation history also records that materializing
the full bundle exhausted a 5 GiB address-space cap; assignment loading reduced
planning peak RSS to roughly 243 MiB.

Prefer selected-assignment iteration in activation and bounded reuse within a
clearly defined validation operation. Preserve freshness/integrity semantics and
measure memory before introducing a cache. This is optimization work, not a
demonstrated correctness bug.

The secondary unguarded `next(...)` finding is confirmed: passing an assignment
whose worker is missing from the bundle raises bare `StopIteration`. Normal
`create_plan` checks this upstream, so this is a lower-priority robustness gap in
direct evaluation/imported inputs; return an explicit unknown reason or error.

### 9. Duplicate probe targets and missing includes — valid maintenance nits

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008378189).
Both executable targets compile `memory_probe.cpp`; the JSON `mode` selects the
operation. Separate names do not enforce separate modes. The usage string itself
lists both alternatives. Missing direct `<filesystem>`, `<limits>`, and `<memory>`
includes are confirmed.

“Byte-identical” is too literal: the two local CPU binaries differ, despite using
the same implementation. Consolidation or executable aliases are optional
build/packaging cleanup, not a runtime correctness fix.

### 10. Repeated mixed-precision weight casts — intentional measured tradeoff

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008380097).
CUDA `weight()` converts to the execution dtype at each call, and MLX constructs
the corresponding conversion. The tied-head path uses the full embedding
matrix. This is a real cost worth measuring if optimizing throughput.

It is already documented: F16 resident weights, F32 execution/KV, transient cast
workspace, and whole-stage profiles including the head operation. Caching a full
F32 head changes residency and requires new memory evidence and profiles. There
is no evidence here that the cost was accidentally omitted from whole-stage
timing, nor does workspace accounting alone independently qualify physical fit.
No fresh GPU throughput claim is made by this audit.

### 11. MLX SiLU affects legacy F16 execution — scope confirmed, validation overstated

[Original comment](https://github.com/anthonylu23/hLLM/pull/15#discussion_r4008381429).
The new F32 SiLU intermediate changes the ordinary F16 path as well as mixed-mode
code. Historical M3/M4 reports are not fresh qualifications of this binary.

However, this PR already documents the rounding change, preservation of old
evidence, 58 native checks and a targeted 256-token serving check after the fix,
followed by refreshed profiles and further numerical investigations. It is not
an undocumented or wholly untested numerical change. Explicitly cross-link those
limits from older F16 reports, and rerun any F16 qualification being claimed for
the current binary. Full M5 acceptance remains deferred.

## Summary-only hygiene and test observations

- New C++ sections exceed the surrounding line width, and no `.clang-format` was
  found. These are style observations, not demonstrated defects.
- The validator-local `import re` and the test-only `check_compatibility` helper
  with a parallel evaluator filter are confirmed maintenance concerns. No new
  compatibility bypass was demonstrated by this audit.
- Added validation JSON/gzip artifacts under `docs/validation` total approximately
  2.96 MB, 117 files, and 46,875 uncompressed JSON lines under this counting scope;
  the review's size estimate is broadly reasonable. `watchdog-repro.cpp` exists.
  Raw failures and evidence are intentionally retained, so deletion is not a
  correctness fix. Any relocation must preserve references and provenance.
- The seven new CLI commands lack dedicated CLI tests in `test_cli.py`. The
  resume, post-completion deadline, and channel-restart gaps are real. The new
  audit reproductions demonstrate why existing passing suites do not settle them.
- `ruff format --check .` reproduces the two formatting differences in `cli.py`.
- The Ubuntu Protobuf observation is pre-existing: `find_package(Protobuf CONFIG
  REQUIRED)` is present on the base branch too. This Mac audit did not reproduce
  Ubuntu package installation or endorse the suggested shim as a portable fix.

## Verification performed here

- `pytest tests/python -q`: **102 passed**.
- CPU integration, native link, and native measured-sweep process tests:
  **21 passed**. CPU compute/memory process tests: **5 passed, 5 skipped**
  (unsupported CUDA/pinned/mixed cases). Combined: **128 passed, 5 skipped**.
- Existing downstream-channel and deadline/control tests: **8 passed**.
- Three separate native audit reproductions confirmed terminal loss, wrong plan
  error code, and transient reconnect failure/recovery. These assert the observed
  current behavior; they are not permanent regression tests for a future fix.
- Python reproductions confirmed interrupted resume and missing-worker errors,
  and counted disk artifact loads. Native RPC checks verified Unicode plan hashes
  and the actual link-probe contention statuses.
- Ruff lint and full Pyright: clean. Ruff formatting: one file differs as above.
- The selected native build reported no work needed for the worker, runtime
  tests, and link-probe targets. No fresh Ubuntu, ASan/UBSan, full MLX/CUDA,
  full-checkpoint, or WAN acceptance run was performed here.

The initial process-test invocation omitted `HLLM_PROFILE_BACKEND`, causing ten
setup `KeyError`s. After setting it to `cpu`, the affected tests produced the
five passes/five skips above. These were invocation errors, not product failures.

Local reproduction sources/results are under `build/validation/pr15-audit/`.
Production code and checked-in tests were not changed, and no GitHub comments
were posted.

## Recommended next steps

1. Fix interrupted resume with an immutable-result policy and a regression test.
2. Fix module command registration and add subprocess coverage.
3. Define and fix post-generation terminal semantics with deterministic slow
   cleanup coverage; correct typed plan rejection errors and assert wire codes.
4. Add generation restart/recovery coverage before choosing channel policy.
   Correct link-probe diagnostics and document serialized qualification.
5. Treat hash consolidation, bounded profile reuse, probe target consolidation,
   and persistent F32 head storage as separately reviewed refactors/optimizations.
   Preserve tamper detection, memory limits, historical evidence, and the existing
   deferred M5 acceptance status.

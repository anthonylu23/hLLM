# M5.5–M5.6 implementation validation — 2026-09-09

Measured planning and the independent sweep runner are implemented and tested.
**Full checkpoint M5 acceptance is pending.** There is no qualified automatic
0.6B selection or <=15% regret result yet. The 4B target remains unqualified.
The [machine-readable status](milestone-5-planner/status.json) records the latest
measured selection and partial sweep coverage; regret remains unqualified.

## Channel reuse experiment — 2026-09-13

The timestamped [TCP diagnostic](milestone-5-planner/deadline-timing-tcp.json)
reproduced 21.8% drift with exact outputs. Slow prefill requests showed incomplete
boundary transfers after one to two seconds. The previous worker created a new
downstream channel for every generation, so request warmups could lose connection
and flow-control state before the timed request.

An implementation experiment now retains one downstream channel per loaded
split deployment. Initialization is thread-safe; each request still owns a fresh
RPC stream, context, deadline and cancellation. Outstanding deployment leases
keep the channel alive through cleanup, and unloading releases the deployment's
ownership. No global endpoint cache or unbounded retained history is introduced.

The isolated `m5-channel` Mac build passed all 65 CTest checks (213.45s);
the `cuda-channel` build passed all 68 checks (355.73s), including channel
ownership tests, cancellation/recovery integration and backend numerical parity. The existing `m5-deadline`
artifacts remain historical evidence for their exact binaries. This experiment
has not established a timing improvement or M5 acceptance. New binary validation,
exact-token timing comparisons and fresh qualification identities are required
before any acceptance claim.

## Implemented behavior

- Exact-assignment memory/compute and directional transport matching, with explicit
  missing, incompatible, stale and physical-fit rejection paths. Unknown candidates
  do not receive optimistic costs or ranks.
- Whole-stage prefill and every decode-context cost, production request setup,
  allocation and encode-plus-RTT attribution; components and profile identities
  remain visible. Full-workload ranking can change both stage order and split.
- Schema 1.1 measured plans bind workload and bundle digests. Both Python and native
  workers validate the measured hash; old-mode hashes retain their existing contract.
- Fresh activation capability, binary/device/allocator/transfer-mode and physical
  checks before load. Definitive rejection and an uncertain load response have
  distinct cleanup handling, without a fallback retry loop.
- Frozen, resumable, seeded all-split sweeps with independent fresh-process memory
  qualification, two warmups per job, five initial timing repetitions per candidate,
  drift references, client/native token timing, exact-reference correctness,
  cancellation/deadline recovery, and model/sequence cleanup.
- Complete-coverage, uncertainty and drift gates on the 15% result; separate timing
  and conservative memory-envelope prediction errors. Unknown failures cannot be
  turned into unsupported memory exclusions or a winning speed result.

The [workflow](../milestone-5-placement.md) documents the contracts, commands,
attribution assumptions, remote process ownership and remaining hardware work.

## Verification

- 80 Python tests pass, including synthetic winner changes, context-dependent
  prediction, physical rejection despite admission fit, stale/missing evidence,
  native/Python wire round trips, activation refresh, frozen/resumable sweep inputs,
  raw-evidence tampering, divergence and drift gates.
- Ruff and Pyright pass; generated protobuf bindings are checked against sources.
- The native CPU/MLX build passed the existing 57-test CTest suite during integration.
  The affected CPU/MLX pipeline tests and the new measured-planner integration test
  were then rerun successfully after subsequent changes.
- The new native integration test runs two actual CPU workers, verifies executable
  identity and measured-plan hash acceptance/rejection, performs fresh memory probes,
  checks output against the independent tiny Transformers fixture, records native
  and client timings, and verifies unload. Its one-token smoke job correctly reports
  an incomplete sweep; it is not hardware performance acceptance.
- CUDA rebuild, numerical parity, CPU/CUDA memory-profile checks, mixed pipeline
  integration and mixed failure/recovery qualification pass. The new measured-plan
  integration test also passes on Linux.
- A [cross-machine SSH executor smoke](milestone-5-planner/ssh-runner-smoke.json)
  passed with a five-token tiny fixture and one generated token, two warmups and one
  timed request. It verified both fresh memory probes, host/package/executable/PID
  identities, exact reference output, native/client timing and unload across macOS
  and Linux. Its manually constructed test placement is not an automatic-selection
  result. The temporary source-specific firewall rule was removed, and no owned
  serving listener remained.

## Independent acceptance reference

The [compressed reference](milestone-5-planner/reference-512-f32.json.gz) contains the
new exact 512-token input and all 256 F32 greedy outputs for pinned Qwen3-0.6B. Its
checkpoint content digest matches the earlier profiles. The producer is Transformers
4.57.6 with PyTorch 2.13.0+cu130, eager attention and TF32 disabled. Tokens are generated
without stop IDs. The sweep's declared policy is exact agreement with this continuation
for every warmup and timed run; a divergent F16 continuation remains a correctness
failure regardless of latency.

The fixture repeats/truncates the story prompt's token IDs to exactly 512 tokens.
This is a controlled token workload, not a claim about natural 512-token prompts.
The exported artifact omits unrelated short-prompt numerical arrays and retains the
full source artifact's SHA-256. Decompression and the published content hash were
verified. The full raw reference is retained in the private build output.

## Acceptance pilot and all-split collection — 2026-09-09 evening

The [512+256 pilot](milestone-5-planner/pilot-512-256.json) passed on the pinned
Qwen3-0.6B checkpoint at split 14 in both MLX→CUDA and CUDA→MLX orders. Each
direction ran two warmups and one timed request in fresh serving processes after
independent native memory probes. All six requests matched every one of the 256
F32 reference tokens. Sequence retirement and model unload passed. Last-token
times ranged from 29.28–32.35 seconds for MLX→CUDA and 27.38–28.52 seconds for
CUDA→MLX. These are pilot observations, not automatic-selection or regret results.
The source hashes match the implementation-validation snapshot and both native
builds were up to date; controller/remote helper package digests matched.

[Maximum-layer stage probes](milestone-5-planner/extreme-assignment-memory.json)
also passed for both first/final roles on both backends. To cover these larger
assignments, subsequent collection uses 3.25 GiB MLX admission, 2 GiB CUDA host
admission and 3 GiB CUDA device admission. Safety remains 10%; host headroom is
256 MiB on MLX and 1 GiB on CUDA, CUDA device headroom is 512 MiB, and extra
overhead is 256 MiB. A maximum-layer memory pass does not establish all-split
correctness or performance.

All-split collection started on both hosts. MLX completed memory and compute for
split 1 in both roles and split 2 in the first-stage role. Its next assignment,
split 2 in the final-stage role, was rejected by the preflight in three separately
preserved attempts. [Available-memory observations](milestone-5-planner/all-split-resource-gate.json)
ranged from 3.81–4.32 GB against the 4.38 GB required for the larger cap. Interim
readings exceeded 5 GB, but that headroom did not persist through launch checks.
Spotlight briefly consumed approximately 2 GB; no existing workload was stopped.
The rejected preflights are not independent infeasibility evidence.

After memory was freed, a [fresh launch check and memory probe](milestone-5-planner/mac-resume-20260909.json)
passed for the previously blocked MLX assignment: 5,646,876,672 bytes were available
at preflight, and the conservative host envelope was 3,533,593,713 bytes. MLX
collection resumed in `profiles-05` with the same capacity and physical allowances.
An additional rejected preflight in `profiles-04` remains preserved.

The subsequent [profile audit](milestone-5-planner/profile-coverage-20260910.json)
on September 10 verified all 54 CUDA memory and 54 CUDA compute artifacts. Their
sealed identities, completion flags and recorded fit assessments passed; the raw
archive was copied locally and its SHA-256 verified. MLX has 30 memory and 29
compute artifacts, covering 29 complete assignments. Its batch stopped at the
split-15 final-stage compute preflight. Neither batch is currently running.
An exact compute-only restart is prepared, followed by the remaining 24 assignment
pairs. At the latest inspection, League of Legends was active on the Mac; resume
timing collection in an idle GPU window to avoid mixing load conditions.

At 01:42 local on September 10, the game was no longer observed and the
[compute preflight passed](milestone-5-planner/mac-resume-20260910.json) with
5,229,068,288 available bytes. MLX collection resumed at split 15 final compute
in `profiles-06`, keeping its completed memory artifact and all safety allowances.
That batch is running; the earlier paused-state observations remain historical.

The next inspection found that batch had completed 39 of 54 MLX assignments
before a preflight rejection at split 20 final memory (3,675,832,320 available
bytes versus 4,375,497,932.8 required). Those 39 assignments have both memory and
compute artifacts. CUDA remains complete. The remaining 15 MLX assignments,
setup/link calibration and the independent sweep are still pending. The user has
authorized pushing the current work to a branch and continuing qualification with
30-minute checks; no acceptance result is implied by that checkpoint commit.

The 02:57 local heartbeat on September 10 found both profile batches complete.
The [complete profile audit](milestone-5-planner/all-profile-coverage.json) verifies
all 216 memory/compute artifacts, unique assignment coverage and consistent scope
and environment identities. Production request-setup calibration is now running:
five fresh measured jobs per direction, each preceded by two reference-checked
warmups, with independent memory probes and unload checks. Directional links,
fresh physical/path snapshots, frozen selection and the independent sweep remain
pending. Calibration is separate from acceptance timing.
Private run logs and the exact restart instructions are in
`build/m5-acceptance/README.md`. Pilot workers were retired and the temporary
source-specific firewall rule was removed. No acceptance sweep has been frozen
or started. Restore stable Mac headroom, finish the missing assignment profiles,
then collect five production setup samples after warmup per direction and refresh
directional links before selecting and freezing the plan. Finally run the full
54-candidate independent sweep and its correctness, health, drift and regret gates.

## Earlier hardware gate

The [recorded Mac gate](milestone-5-planner/resource-gate.json) observed
3,509,518,336 available bytes. Even the previously qualified split-14 admission cap
requires 3,766,484,992 bytes with 10% safety, 256 MiB headroom and 256 MiB extra
transport/runtime allowance. This conservative preflight fails. An active user game
was also observed during resource inspection; no existing workload was stopped.
This was an availability constraint, not a claim that the checkpoint can never
fit, and it is not independent exclusion evidence for all 54 candidates.

Next steps are to refresh compatible current-build profiles across splits, collect
production request-setup calibration in both directions, freeze automatic selection,
and run all 54 candidates once resource headroom permits. Publish the complete
coverage, correctness, drift, cleanup, uncertainty and prediction-error report before
claiming M5 acceptance. 4B and the separate 32K capacity workload require their own
physical-fit and inference qualification.

Production setup calibration and both directional link profiles now pass; the
[calibration audit](milestone-5-planner/calibration-20260910.json) records five setup
samples per direction and 30 exact-reference requests including warmups. The first
full planner attempt exhausted its 5 GiB address-space cap while serializing the
216-profile collection. Disk-backed bundle support preserves all raw evidence and
validation while loading profiles by assignment. The real bundle indexed at about
215 MiB peak RSS; measured selection and independent acceptance remain pending.

The disk planner completed all 54 candidates with measured evidence and no unknown
or infeasible candidates. It selected `mlx--cuda-m001` (MLX → CUDA, split after layer
1), using a peak RSS of 248,880 KiB (243 MiB) across assembly and evaluation. The
selection, bundle, reference and executor are frozen and the independent sweep has
started. The first reference job passed fresh memory probes and entered inference.
Acceptance is still pending the complete sweep, correctness/health checks, drift and
bootstrap regret gates. The 30-minute monitor tracks progress and preserves failures.

The first sweep stopped on an exact-token correctness failure at MLX → CUDA split
25, in the first warmup of its twelfth job. Eleven jobs succeeded beforehand. The
first mismatch is output token 146; failed evidence and frozen selection are retained,
and owned processes/firewall rules were cleaned up. The
[failure audit](milestone-5-planner/sweep-01-failure.json) records a small independent
F32 logit margin at that position and a tie after F16 rounding. Mixed-backend tracing
is still required to establish the cause. This is a failed correctness gate, with
full acceptance and statistical comparison incomplete.

[Mixed-boundary controls](milestone-5-planner/boundary-trace-01.json) reproduce the
failure with exported native F16 payloads. The same CUDA suffix chooses the correct
token when given the independent F32 prefix rounded to the same F16 wire format.
Increasing only the native CUDA suffix to F32 does not restore the reference token.
This localizes the decisive difference to prefix numerical values for this history;
it does not establish a faulty individual MLX operation. An F32 MLX-prefix diagnostic
was rejected by the unchanged admission cap. No acceptance margins were relaxed.

A [SiLU precision experiment](milestone-5-planner/silu-precision-experiment.json)
restores all 256 reference outputs in the split-25 boundary replay. MLX previously
rounded sigmoid to F16 before multiplying by its input; computing SiLU in F32 and
rounding the activation once removes that avoidable rounding. A control that rounds
only model weights through F16 also retains the correct token. The correction is
being validated in a separate production build; original binaries and sweep evidence
remain intact. Changed-kernel profiles and a new sweep are required for acceptance.

The production fix passes all 58 native tests and the targeted serving regression:
two warmups plus one timed request, each matching all 256 reference tokens, followed
by clean unload. New MLX profile collection stopped before its first native run because
available Mac memory was about 50 MB below the unchanged physical preflight threshold.
Changed-binary profiles, fresh selection and full independent acceptance remain pending.

Corrected MLX and refreshed CUDA profile collection is complete. The
[new coverage audit](milestone-5-planner/all-profile-coverage-silu.json) verifies all
216 assignment/kind artifacts, including safe independent memory assessments and
consistent scope identities. Production setup calibration with the corrected binary
is running five fresh jobs per direction. New link/snapshot evidence, measured
selection and the independent sweep are still required; the failed sweep is preserved.

## Corrected sweep and numerical follow-up — 2026-09-11

Refreshed profiling covers all 216 artifacts. The [new calibration audit](milestone-5-planner/calibration-silu-20260911.json)
records 30 exact setup requests and qualified links in both directions. A new measured
plan selected MLX → CUDA after layer 3 and was frozen before independent timing.

The [corrected sweep](milestone-5-planner/sweep-silu-02-failure.json) stopped after
seven successful jobs (one selected reference and six candidates). CUDA → MLX after
layer 1 failed its first warmup at output index 145: expected 2487, observed 4034.
Owned processes exited and the firewall rule was removed; normal unload was not proven
for the failed job. Both failed sweeps and their original evidence remain intact.

A replay of the exact boundary history reproduces the mismatch. A separate
[F32 final-projection control](milestone-5-planner/head-f32-control.json) also chooses
4034, with unchanged last-layer activations. Final-logit rounding alone therefore does
not explain this failure. No additional production kernel or acceptance-policy change
has been made. The [completed diagnostic matrix](milestone-5-planner/correctness-matrix-01.json)
checked all 54 placements against all 256 teacher-forced reference steps: 51 were
exact, with one mismatch each at reverse split 1 and forward splits 17 and 19.
These diagnostics do not count as independent serving or timing acceptance.

Next, use that matrix to guide a numerical fix or identify a precision-policy decision.
Any implementation change needs broad correctness validation and refreshed affected
measurements before a new frozen sweep. Full coverage, health checks, drift and
statistical regret acceptance remain outstanding; 4B and 32K remain unqualified.

The [independent oracle controls](milestone-5-planner/oracle-matrix-control.json)
match all 256 tokens in both F32 and F16. The two affected positions have F32
winning margins of about 0.005796 and 0.001921; both become ties in the F16 oracle.
Replaying [F32 oracle prefixes rounded to F16 boundaries](milestone-5-planner/oracle-boundary-matrix-01.json)
through the three failing suffixes recovers all 256 exact tokens in each case.
This localizes sensitivity to the prefix inputs, without proving a specific kernel
bug. Next, compare prefix intermediate values at these positions with the oracle
before selecting any further precision change. Strict acceptance remains unchanged.

The [first-layer rounding control](milestone-5-planner/prefix-precision-01.json)
reproduces all 256 native CUDA boundaries byte-for-byte. Separately applying the
F16 oracle's rotary or attention rounding restores exact suffix tokens, while each
variant slightly increases relative boundary error against F32 at the affected step.
This is evidence of rounding-path sensitivity, not a demonstrated faulty operation;
no production change was adopted. The next diagnostic compares MLX prefix layers
with oracle intermediates at output index 200 for splits 17 and 19.

The [MLX layer comparison](milestone-5-planner/mlx-layer-control-01.json) checks
layers 0–18 at output index 200. All shared prefix rows match between splits 17
and 19, and exported boundaries match the original matrix. At every checked layer,
MLX has lower relative L2 error against F32 than the independent F16 oracle does.
At the two split boundaries, errors are about 0.147% and 0.135%, versus 0.182%
and 0.175% for the F16 oracle. This does not establish a kernel fault or satisfy
exact-token acceptance. Any further precision experiment must demonstrate numerical
improvement and broad correctness, rather than merely fitting the three token choices.

An isolated [F32 residual control](milestone-5-planner/residual-f32-control.json)
retains F16 weights, KV, projection outputs and boundaries while keeping residual
sums in F32. It recovers all 256 tokens at reverse split 1 and lowers prefix error
at splits 17 and 19 from 0.147%/0.135% to 0.095%/0.098% at index 200.
The [completed 54-placement matrix](milestone-5-planner/correctness-residual-01.json)
rejects this control: only 44 placements are exact, with ten regressions despite
repairing the original three failures. The control is not adopted in
production; broad correctness, execution-contract review and fresh affected memory,
compute, calibration and serving evidence are required before acceptance.

Further sweeps of unchanged production cannot pass the current correctness gate.
The user confirmed continued engineering toward exact F32-reference tokens for
every placement. Numerical tolerances remain unapproved. An isolated control with
F32 layer intermediates and F16 stored weights, KV cache and wire boundaries is
being investigated; it is not a production change or an acceptance result.
The timing, memory, coverage, health and regret requirements remain unchanged.

The [complete internal-precision matrix](milestone-5-planner/correctness-internal-complete.json)
combines 54 unique placements across three preserved attempts, with all raw hashes
and the shared MLX binary identity verified. It yields 51 exact placements: every
CUDA → MLX split is exact, while MLX → CUDA splits 16, 18 and 21 diverge.
This variant remains unadopted. A matching isolated CUDA precision control is being
built for a new comparison with F32 intermediates on both stages; weights, KV storage,
wire format, memory caps and exact-token acceptance remain unchanged.

The [combined precision matrix](milestone-5-planner/both-internal-progress.json)
completed 51 placements before a resource stop, with seven index-200 mismatches.
The remaining three are prepared. A [CUDA KV-cache control](milestone-5-planner/cuda-kv-control.json)
replays the identical failing forward-split-4 boundary with F32 suffix KV storage
and recovers all 256 tokens. Cache allocation and memory accounting both use four
bytes per element under the same caps. This establishes a contribution from suffix
cache quantization for this case; F32 KV does not qualify the F16 target. Next,
isolate key versus value quantization before considering F16-preserving changes.

Separate [key and value controls](milestone-5-planner/cuda-key-value-controls.json)
each restore all 256 exact tokens at forward split 4. Both correctly account for
six bytes per key/value element pair under unchanged caps. This does not uniquely
identify one cache as faulty. A full F32-KV diagnostic matrix is prepared with
four-byte cache allocation and accounting on both backends. Its purpose is to
establish whether cache precision resolves the remaining sensitivity; it cannot
qualify the F16-KV target or be adopted without an explicit precision decision.

The [completed F16-KV/internal-F32 matrix](milestone-5-planner/both-internal-complete.json)
is 47/54 exact and remains unadopted. The [F32-KV matrix](milestone-5-planner/kv-f32-progress.json)
has 23/23 exact placements so far, with all raw hashes verified. It stopped at the
Mac resource guard; 31 remaining placements are prepared for a fresh continuation
without overwriting the first attempt. Full diagnostic coverage is still required,
and even a complete pass would require an explicit precision-contract decision
before implementation and fresh acceptance qualification.

While the Mac was constrained, [nine remaining CUDA prefixes](milestone-5-planner/kv-remote-prefix-preparation.json)
were prepared remotely and verified against binary, history, specification and payload
hashes. The continuation reuses those exact boundaries; they count toward correctness
only after suffix replay. The remaining 31 placements have resumed after the Mac
resource guard passed, preserving the earlier 23 exact results. F32 KV adds 84 MiB
of cache across 28 layers at 768-token capacity, before backend workspace overhead;
this allocation calculation is not a fresh physical-memory qualification.

The continuation launch immediately stopped at its resource guard (4.32 GB), before
any new placement completed. The 23 exact results and nine prepared prefixes remain
intact. The next continuation is prepared; prefer 4.9 GB available at launch to absorb
startup fluctuations while retaining the existing 4.6 GB runtime guard.

## Complete precision diagnostic — 2026-09-12

The [audited F32-KV matrix](milestone-5-planner/kv-f32-complete.json) passes all
54 placements and 13,824 teacher-forced outputs. Coverage, raw hashes, plan pairs,
input history, configured caps and nine reused prefixes were verified. These
copied-stage diagnostics override the load specifications' F16 arithmetic/cache
semantics; they are not production acceptance evidence for the existing target.

The [precision proposal](../milestone-5-precision-proposal.md) requests approval for
explicit F16 resident weights, F32 execution/KV and F16 wire activations, keeping
exact tokens and all other gates unchanged. Production implementation, fresh memory
and performance qualification, the acceptance sweep, and the user's separate final
M5 audit remain required before marking complete or creating the PR.


## Approved mixed-precision implementation — 2026-09-12

The user approved the [explicit precision target](../milestone-5-precision-proposal.md):
F16 resident weights, F32 execution and KV, F16 wire. This replaces the execution/KV
precision for the new qualification while preserving all exact-token, workload,
resource, timing, and audit requirements.

Schema 1.2 now carries resident weight precision through settings, plans, native
capabilities, canonical hashes, profiles, measured selection, activation and sweep
placements. Legacy serialization and profile identities remain unchanged when the
new field is absent. CPU workers and unsupported precision pairs reject the mode.
Both GPU backends retain F16 weights, cast to F32 for computation, and account for
F32 KV plus transient weight-cast workspace. Independent memory evidence must match
the requested resident and execution dtypes, including any sweep exclusion.

Initial verification: 88 Python tests pass; Ruff, Pyright and protobuf freshness
checks pass. The CPU/MLX 59-test CTest suite passes, with affected serving, profile,
measured-plan and numerical checks rerun after final changes. CUDA production build and its 62-test CTest suite
pass; affected numerical, serving and profile checks are rerun after final changes. Fresh full-checkpoint evidence is
being collected; the earlier 54/54 diagnostic does not satisfy production acceptance.

Both extreme assignments on each backend passed fresh full-checkpoint memory
qualification under the fixed caps. The [compact evidence](milestone-5-planner/mixed-extreme-memory.json)
records schema/precision, executable and artifact identities, accounted memory and
physical envelopes. These four memory checks do not establish exact-token serving,
all-placement coverage, selection quality, or M5 acceptance.


The [mixed-precision setup calibration](milestone-5-planner/mixed-setup-calibration.json)
passed all ten fresh jobs at split 14, five in each direction. Each job used two
warmups and one timed request: all 30 requests matched all 256 independent reference
tokens (7,680 generated tokens), with memory qualification and clean unload. The
raw evidence hashes, profile precision and executable identities, timing triples,
and temporary firewall cleanup were audited. Median request-setup estimates are
113.02 ms for MLX→CUDA and 161.16 ms for CUDA→MLX. These are calibration estimates,
not placement regret or full-workload latency claims. Both hosts are collecting
new all-assignment memory and compute profiles; selection and acceptance remain pending.

The [CUDA mixed-precision profile collection](milestone-5-planner/mixed-cuda-profiles.json)
completed all 54 memory and 54 compute profiles. An artifact-by-artifact audit
verified all assignments, hashes, workload/precision and executable identities,
measurement conditions, and recomputed all 54 safe memory-fit assessments.
Mac collection remains in progress; the completed CUDA profiles do not establish
an automatic selection or acceptance result.

The [MLX profile collection](milestone-5-planner/mixed-mlx-profiles.json) also passed
its 108-artifact audit. Both hosts now have all 54 memory and 54 compute profiles
for the approved precision, with every memory fit recomputed safe. Fresh link
measurements passed in both directions. New measured selection and activation
validation precede the independent acceptance sweep; M5 remains pending.

The [fresh measured selection](milestone-5-planner/mixed-selection.json) qualifies
all 54 candidate placements and selects MLX→CUDA at split 1. The actual measured
plan activated through its frozen disk-backed bundle, generated all 256 exact
reference tokens, and left both workers unloaded with no active requests or
reservations. Fresh link measurements passed in both directions. This establishes
selection and activation behavior; the independent five-round acceptance sweep
and separate final M5 audit remain required.

All 54 placements completed their first production sweep jobs with exact tokens
and clean unload: [initial coverage checkpoint](milestone-5-planner/mixed-first-coverage.json).
Each placement passed two warmups and one timed request, totaling 162 exact
512+256 requests and 41,472 generated tokens. This checkpoint checks completed
job results; it is not the final raw-evidence audit or statistical acceptance.
The remaining shuffled rounds, selected health checks, drift and regret gates,
and separate final M5 audit remain required.

## Mixed sweep failure investigation — 2026-09-12

**Acceptance is blocked.** The [investigation record](milestone-5-planner/mixed-health-investigation.json)
verifies the raw evidence digests for all 112 completed job attempts. Both first
placement rounds completed: all 54 placements passed twice. There are 111 measured
jobs with verified unload and one unknown job, `r001-reference-after`. All three
full requests in that failed job also matched the reference, bringing the total
to **336 exact 512+256 requests**. It failed during selected health with
`CANCELLED: Cancelled on the server side`; its evidence does not identify which
health phase failed. Cleanup was not verified within that job, though owned
processes subsequently retired and the temporary firewall rule was removed.

A separate diagnostic using the unchanged frozen worker binaries and sweep
placement passed three more exact full requests and ten consecutive health pairs.
Every pair verified cancellation after one token, a deadline during decode, empty
request reservations and exact four-token recovery. Final unload and firewall
cleanup passed. These are diagnostic results, not replacement acceptance samples.

An [isolated C++ reproduction](milestone-5-planner/watchdog-repro.cpp), linked against
the frozen worker runtime archive, confirms a status-loss mechanism. A synthetic
backend that unwinds promptly at its deadline returns `DEADLINE_EXCEEDED`. Delaying
its unwind by 250 ms crosses the watchdog's 100 ms fallback and returns the exact
`CANCELLED: Cancelled on the server side` message. Both cases release reservations,
recover and unload. The test deliberately asserts the current fallback behavior;
it is not a passing regression test for the stricter M5 acceptance contract.
This single-stage reproduction confirms the mechanism but does **not** establish
that it caused the original cross-machine failure, whose partial health trace was
not retained. Production runtime code and binaries remain unchanged.

Timing is a separate blocker. The valid reference samples span 21.652–24.287 seconds,
a **12.171%** max/min spread against the frozen 10% limit. The slow reference's
first-token time was 2.642 seconds versus 0.650 seconds for the fastest reference;
request setup remained approximately 0.106 versus 0.117 seconds. Native and client
timing agree on the slowdown. The cause of the prefill variability remains
unresolved; the current unrelated CPU workload was left running. Adding samples
cannot reduce a max/min spread, so resuming this sweep cannot make it qualify.

Next steps are to preserve precise deadline status while keeping blocked transport
operations bounded, add deterministic regression coverage, and retain partial
health phases and tracebacks. Requalify affected identities and cross-machine
health, investigate timing stability, then freeze a new complete sweep. Keep every
original attempt and the exact-token, memory, drift and regret gates. The separate
M5.1–M5.6 audit still precedes acceptance and PR creation.

## Deadline handling fix — 2026-09-12

The [deadline fix checkpoint](milestone-5-planner/deadline-fix.json) implements the
confirmed watchdog correction. A generation deadline still cancels the peer and
marks the request cancelled. The 100 ms client-transport fallback now applies only
while a client write remains active; slow compute unwind can return its precise
`DEADLINE_EXCEEDED` status. A write publishes its active flag before checking
cancellation, so the watchdog cannot exit just before a new blocking write starts.
Explicit cancellation and downstream idle-read interruption remain bounded.

A real gRPC regression with a private slow-decode backend failed against the old
runtime and passes after the fix. Separate tests verify control cancellation,
stalled-client write interruption, reservation retirement, recovery and unload.
All **62 Mac CTest checks** and **90 Python tests** pass, along with Ruff, Pyright
and the protobuf consistency check. The separate CUDA build also passed all
**65 CTest checks**. The [cross-machine health pilot](milestone-5-planner/deadline-health.json)
passed at split 1 in both stage orders: six full exact 512+256 requests, ten
cancellation/deadline pairs, twenty exact four-token recovery prefixes, and clean
unload. Raw evidence digests and worker binary identities were verified; the
owned firewall rule was removed. These are health diagnostics, not sweep samples.

Failed health attempts now retain partial fault phases, token counts, RPC status,
recovery output and tracebacks. Recovery must return the complete expected prefix;
a shorter matching prefix is rejected. New source, executable and package
identities are kept separate from the failed sweep. This fix does not establish
the original failure's exact cause or resolve timing drift. Production profile
refresh, new selection/sweep, and the final M5 audit still remain.

Fresh profiles for the new binaries are collecting on both hosts, with the largest
first/final assignments checked before all 216 memory and compute artifacts. The
failed sweep and its timing samples remain preserved. Setup/link calibration,
new measured selection, timing stability and a newly frozen sweep still precede
the final acceptance audit.

The [new-build CUDA profile audit](milestone-5-planner/deadline-cuda-profiles.json)
passed all 108 artifacts: 54 memory and 54 compute profiles, exact assignment and
workload coverage, source/executable identities, measurement conditions and sealed
artifact hashes. All 54 independent memory fits recompute as safe. Mac collection
is still running (46/108 artifacts at this checkpoint), with no resource pauses
or failures observed. The CUDA audit alone does not qualify selection or regret.

The [new-build MLX profile audit](milestone-5-planner/deadline-mlx-profiles.json)
also passed all 108 artifacts, including all 54 safe memory fits. Both collection
processes exited successfully: **216/216 profiles are complete and audited** for
the deadline-fixed binaries. Fresh production request-setup calibration is now
running five jobs per stage order, each with two exact-reference warmups and one
timed request. Link/environment refresh, new measured selection, timing stability,
independent acceptance and the final audit remain pending.

[New-build setup calibration](milestone-5-planner/deadline-setup-calibration.json)
passed all ten fresh jobs, five per stage order at split 14. All **30 full requests**
matched the independent reference, with verified unload. Raw evidence hashes,
worker identities, memory-profile scope and setup timing triples were audited;
setup residuals were independently recomputed. Median setup estimates are
107.04 ms for MLX→CUDA and 164.67 ms for CUDA→MLX. The owned firewall rule was
removed. Fresh directional links are running, with measured selection and exact
activation queued only after link success and cleanup. No regret or timing-drift
acceptance is implied by this calibration checkpoint.

The [refreshed measured selection](milestone-5-planner/deadline-selection.json)
measured all 54 candidates and selected MLX→CUDA at split 1. Its actual measured
plan activated through the new frozen bundle, matched all 256 reference tokens,
and unloaded cleanly. Both directional links passed and owned firewall rules were
removed. A separate five-job timing-stability pilot is prepared with two exact
warmups and one timed request per fresh job. It waits below the unchanged 4.9 GB
Mac startup guard. These diagnostic jobs do not replace the independent acceptance
sweep or its 10% drift gate.

The [five-job timing pilot](milestone-5-planner/deadline-timing-failure.json)
completed with 15 exact full requests and clean unload, but **failed the 10% timing
stability gate: 12.398% drift**, spanning 22.533–25.327 seconds. All samples and
raw evidence digests were audited and preserved. Variation occurs in prefill and
decode, including occasional long inter-token gaps; native and client timings
agree closely. The original prefill-only hypothesis does not explain all observed
variation. Host pressure and network latency remain possible contributors, not
established causes. An instrumented diagnostic is collecting RTT, Mac memory
activity and CUDA utilization/clocks alongside generation with unchanged binaries
and correctness requirements. It cannot substitute for acceptance samples, and
no full sweep will launch on this failed stability result.

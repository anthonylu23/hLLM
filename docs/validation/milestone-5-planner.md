# M5.5–M5.6 implementation validation — 2026-09-09

Measured planning and the independent sweep runner are implemented and tested.
**Full checkpoint M5 acceptance is pending.** There is no qualified automatic
0.6B selection or <=15% regret result yet. The 4B target remains unqualified.
The [machine-readable status](milestone-5-planner/status.json) records the latest
measured selection and partial sweep coverage; regret remains unqualified.

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

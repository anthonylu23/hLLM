# Measured placement and independent qualification

M5.5 implements explicit measured planning. M5.6 implements the independent sweep
runner. The full checkpoint acceptance run remains pending; see the
[implementation validation and hardware gate](validation/milestone-5-planner.md).

## Build and compatibility

Rebuild workers and regenerate protobuf bindings after updating this code. The
worker runtime now links OpenSSL Crypto to hash its executable and validate measured
plan hashes. Legacy feasibility/estimated plans retain schema 1.0 and their existing
plan hashing. Measured plans use schema 1.1 and hash the full workload digest and
profile-bundle digest. Python and native workers reject altered measured plans.

The approved M5 target uses opt-in schema 1.2: `weight_dtype: F16`,
`execution_dtype: F32`, workload `kv_dtype: F32`, and F16 wire activations.
Use `examples/profiles/milestone-5-mixed-objective.yaml` with
`examples/workloads/milestone-5-mixed.yaml`. Both workers must advertise
`supports_mixed_precision: true`; CPU workers reject this mode. With `weight_dtype`
absent, resident weights retain the execution dtype and legacy hashes are unchanged.
Schema 1.2 plans (including feasibility plans) bind the explicit precision in their
canonical digest and use `plan-<first 16 digest characters>` identifiers. Matching
memory and compute artifacts use schema 1.2 and include the resident weight dtype.
Legacy profiles cannot qualify the new target.

Native MLX/CUDA loaders keep F16 weights, cast selected weights for F32 computation,
and allocate F32 KV caches. Admission includes transient F32 weight casts, bounded
by the largest evaluated layer or output-head group. Embedding lookup casts only
its selected rows. The fixed memory caps still apply. Fresh profiles, selection,
independent acceptance sweep, and the separate final audit remain required; the
54-placement diagnostic is not production acceptance.

`GetQualificationState` reports fresh physical availability, executable SHA-256,
PID, a host/hardware fingerprint, backend/device/driver API identity, allocator configuration, and boundary
transfer mode. It does not reset allocator peaks. Measured activation refreshes
capabilities and this state before any load, requires unloaded idle workers, and
checks the independent physical envelope. A failed load unwinds acknowledged stages
and a stage whose load response was lost. A definitive rejection is not treated as
an acknowledged load. There is no automatic fallback to another placement.

## Assemble and use a bundle

`hllm_control.planner.measured.BundleContent` is the strict input schema. A bundle
contains the exact manifest/checkpoint/workload/input identities, two worker
bindings, memory and whole-stage compute artifacts, directional link artifacts,
and production request-setup samples. Worker bindings explicitly identify the
memory probe, compute probe and serving executable; these are different binaries.
Bindings include configured admission caps and timestamped physical budgets.
Memory and compute must agree on device, backend, allocator configuration and OS.
Missing, stale, ambiguous, incomplete or mismatched evidence cannot rank a candidate.

Store the assembled object as JSON, then seal and plan:

```bash
uv run hllm seal-profile-bundle --input build/m5/bundle-input.json \
  --output build/m5/bundle.json
uv run hllm plan --manifest build/m5/manifest.json \
  --workers build/m5/workers.yaml --workload examples/workloads/milestone-5.yaml \
  --mode measured --profile-bundle build/m5/bundle.json \
  --settings examples/profiles/milestone-5-objective.yaml \
  --output build/m5/measured-plan.json --report build/m5/measured-report.json
uv run hllm explain build/m5/measured-report.json
```

The worker profiles in `--workers` must exactly match the frozen bundle. `--links`
is optional in measured mode; configured link scalars do not substitute for native
link measurements. `hllm generate` requires `--profile-bundle` for measured plans and
rejects a different prompt/output length. Workload capacity must match the exercised
request; the initial sweep supports precisely prompt + output capacity.

Enumeration remains both orders and every interior split. Candidates are explicitly
`measured`, `unknown`, or `infeasible`. Ranking includes only fully measured feasible
candidates; therefore a selected plan with unknown alternatives is not an exhaustive
optimality claim. Exact ties use candidate ID. Explanations include the winner,
runners-up, TTFT/generation estimates and rejection reasons; JSON retains every
component, context, objective settings, artifact digest and conservative physical envelope.

## Timing attribution

The fixed workload objective is TTFT + 255 mean decode intervals for 512+256.
Prediction uses uninstrumented whole-stage medians at each exact context, sequence
allocation medians, production request setup, and directional sender-encode + RTT.
Whole-stage timings already include device boundary conversion. RTT includes
receiver decoding and token feedback; never halve it or add feedback again.
Instrumented per-layer costs and the link probe's diagnostic stream-readiness barrier
are not summed into the prediction. Load time and tokenization are outside the timed
generation interval.

Production `GenerationRequest.capture_timing` adds optional native elapsed time to
each token and setup time to the first token. Ordinary requests do not take these
extra timestamps. Native setup excludes stage-zero acquisition/allocation and model
execution. For each separate calibration request, retain client TTFT, native TTFT
and native setup. The setup contribution is:

```
native_setup + client_TTFT - native_TTFT
```

`DirectionEvidence.request_setup_raw` stores these raw triples; its samples and
content digest are validated against them. Collect at least five samples after
warmup, record method/time, and keep them separate from the later qualification
sweep. Backend allocation and downstream startup can overlap; the additive predictor
is an approximation to qualify, not a claim of perfectly disjoint wall-clock regions.
The saved M5.3–5.4 artifacts lack these production setup samples and cover only split
14. They cannot by themselves produce a qualified all-split measured plan.

## Freeze and execute the sweep

Generate a new independent reference with the isolated Transformers environment:

```bash
python scripts/validation/checkpoint_reference.py MODEL_ROOT reference-512.json \
  --prompt-tokens 512 --output-tokens 256
```

The output preserves exact prompt IDs, checkpoint hashes, producer versions and all
256 greedy tokens. Prompt IDs are deterministically repeated/truncated from the
story fixture; they are authoritative. Stop IDs are empty. The predeclared sweep
policy requires exact agreement for every output token in every warmup and timed
request. F16 divergence stays a correctness failure and cannot win on speed.

Create a `NativeExecutor` JSON configuration (schema in
`python/hllm_control/qualification/native.py`). Each of its two workers specifies:
worker ID, backend, endpoint, serving/memory-probe paths and SHA-256 digests, model
root, evidence root, admission capacities, transfer mode, headroom/overhead and
bounded timeout. The interpreter must have this checkout's Python package installed.
An optional `ssh_host` runs the same helper on the remote machine. Addresses must be
reachable between workers and equal their configured listen addresses. SSH helpers
lease native children through stdin and retire only their own child on EOF; the
controller verifies its recorded PID and executable digest before loading.

```bash
uv run hllm freeze-sweep --manifest build/m5/manifest.json \
  --report build/m5/measured-report.json --profile-bundle build/m5/bundle.json \
  --reference reference-512.json --executor build/m5/executor.json \
  --output build/m5/sweep-input.json
uv run hllm qualify-sweep --spec build/m5/sweep-input.json \
  --executor build/m5/executor.json --output build/m5/sweep
```

`--max-jobs N` provides a bounded first run; repeat the same command to resume.
The spec, result digests and raw evidence must remain unchanged. Failed or interrupted
attempts are retained. A subsequent run does not silently erase/retry failed jobs;
start a new frozen sweep after correcting the cause. Executor configuration and the
complete controller-package source hashes are part of the frozen identity. Remote
helpers verify that package fingerprint before running any native process.

Each round shuffles all `2*(L-1)` candidates with a saved seed and repeats the selected
reference before/after the round. Each job runs a fresh native memory probe for each
stage, starts fresh serving processes, performs two warmups and one timed request,
and verifies sequence retirement and unload. Thus five initial rounds provide five
independent measured samples per candidate. Model load and memory telemetry are
outside timing. Raw client/native token arrivals expose buffering. Selected-reference
jobs additionally exercise cancellation, a deadline during decode, cleanup and
reference-checked recovery. Warm reload measurements remain separate from this
fresh-process protocol.

Memory exclusions require a completed independent native memory profile with exact
assignment/checkpoint/workload scope and an unsafe physical/admission assessment.
A failed load or an unavailable/preflight-rejected probe stays unknown; planner
rejection alone is never an exclusion. Raw probe logs are retained, including unknown
failures. A correctness failure or changing feasibility leaves acceptance incomplete.

The report compares the selected median last-token latency with the best independent
feasible candidate, and includes a seeded 95% bootstrap interval over selected/best
across all measured candidates. A pass requires complete coverage, successful selected
health checks, reference drift <=10%, and an interval upper bound <=15% regret.
An interval crossing the threshold triggers more rounds, up to fifteen. Persistent
noise is inconclusive. This bootstrap is conditional on observed samples, not a
universal guarantee across changing machine load. TTFT, mean ITL, generation latency
and conservative memory-envelope prediction errors are reported separately when
predictions exist. Raw results remain private; publish redacted summaries and digest
references without editing the sealed input/evidence used for resume.

Large collections can use a content-addressed disk bundle. `DiskBundleContent`
contains the same workload, workers, links, directions and freshness policy as an
inline bundle, plus `profile_references` (artifact digest and exact assignment).
Store each original sealed artifact at `profiles/<artifact_digest>.json` beside the
index, then call `seal_disk_bundle(content, index_path)`. It validates every artifact
before publishing the index. `read_profile_bundle` accepts both formats; planning,
activation and `freeze-sweep` use this reader. The disk bundle digest seals the index,
whose artifact digests transitively bind all raw measurements. It intentionally
uses a different bundle identity from the inline representation.

The reader validates all referenced evidence once and validates each artifact again
when used, including its declared assignment. Planning loads only the profiles for
the current assignment. This bounds profile residency by candidate size instead of
the full collection; diagnostic records and all evidence gates remain intact.

For an isolated mixed-backend numerical investigation, build the opt-in
`hllm-boundary-trace-mlx` and `hllm-boundary-trace-cuda` targets. Invoke each as
`MODEL_ROOT LOAD_SPEC HISTORY OUTPUT`. A first-stage history contains `steps` with
`input_ids` and `position`; its output preserves each real F16 boundary payload and
can be replayed as the final-stage history. Both stages trace only the last step,
report last-row layer activations, and the final stage reports logits and sampled
IDs. The tool enforces the load specification's memory capacity and refuses existing
outputs. These diagnostics do not replace independent serving correctness, timing,
or fresh memory qualification, and are never registered as ordinary tests.

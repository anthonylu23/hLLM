# M5 preliminary acceptance audit — 2026-09-14

**M5 is implemented but not accepted.** This audit records engineering checks and
fixes before the final production qualification. It does not replace the separate
final audit required after a successful independent sweep.

## Fixed findings

| Finding | Correction | Commit |
| --- | --- | --- |
| Direct summaries could count duplicate jobs, omit after-reference health checks, or skip result validation | Require unique scheduled jobs, validated evidence and complete contiguous rounds | `8818c6c` |
| Link/setup evidence could expire after planning but remain eligible at activation | Check selected transport measurement ages against activation time | `3a3fbbc` |
| Direction labels could disagree with probe worker identities | Bind both declared worker IDs to their probes | `117e21a` |
| Imported measured plans could disagree with frozen workload precision | Check execution/KV and wire/workload precision before worker RPCs | `38021c3` |

Normal sweep execution already validated scheduled results, and normal planning
already enforced execution/KV agreement. The summary and imported-plan findings
concern additional entry points. Mixed-precision schema validation already
requires F16 weights and F32 execution; no demonstrated downgrade of that target
is claimed.

After these fixes, all **102 Python tests**, changed-file Ruff checks and full
Pyright checks with the project interpreter pass. Native code was unchanged by
these four fixes. The native stage-deadline builds previously passed 66 Mac and
69 CUDA CTest checks; their full-checkpoint health evidence remains historical,
not a qualification of newly generated Python executor identities.

## Reviewed boundaries

The preliminary review covered fresh memory scope and physical-fit checks,
package/binary identity enforcement, owned process leases, ordered memory phases
and cleanup/context coverage, exact measured-profile matching, activation checks,
and content-addressed disk bundles. Disk artifacts are revalidated on access.
This bounded review is not a claim that all M5 paths have been exhaustively audited.

## Remaining acceptance evidence

The approved target is pinned Qwen3-0.6B, 512 prompt plus 256 output tokens,
concurrency 1, cache capacity 768, F16 resident weights, F32 execution/KV and F16
wire. Exact independent tokens and existing memory caps/margins remain required.
The 4B and 32K targets remain separate and unqualified.

Production timing remains unresolved. Uninstrumented stage-deadline timing failed
the 10% reference-drift gate even with sleep prevented. Raw TCP tests reproduced
latency tails in both directions. A [clean endpoint UDP capture](milestone-5-planner/udp-endpoint-confirmation.json)
recorded all 600 probes leaving the Mac capture point, but only 595 at the PC and
in returned echoes. This supports a delivery-loss contribution; it does not locate
the physical fault or prove the cause of each TCP stall. There is no supported
production transport fix yet. Both machines remain on separate networks over
Tailscale; Ethernet is unavailable.

Before acceptance:

1. Establish an evidence-supported correction or stable operating condition,
   then pass production timing without changing thresholds.
2. Synchronize the updated Python package and create new executor identities.
   Preserve historical configurations and results.
3. Collect fresh memory/compute profiles, setup calibration and link/environment
   snapshots; make a new measured selection and freeze its independent sweep.
4. Cover all 54 placements or independently supported memory exclusions, with two
   warmups per fresh job and five shuffled rounds, extending to at most 15 when
   uncertainty requires it. Require exact tokens, cleanup, selected fault/recovery
   checks, reference drift at most 10%, and observed regret and seeded 95% bootstrap
   upper bound at most 15%. Unknown failures remain unresolved.
5. Complete the separate M5.1–M5.6 final audit, fix findings and revalidate affected
   behavior. Only then mark acceptance and create the PR.

Repeated identical network probes are not the next step. Further transport work
needs a concrete hypothesis and a bounded comparison; the existing loss evidence
is sufficient to keep the current acceptance claim open.

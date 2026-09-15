# M5 engineering closeout — WAN acceptance deferred

The user elected to move on from M5 once the bounded code investigation was
complete, rather than continue chasing cross-region performance acceptance.
M5.1–M5.6 are implemented. **Full M5 performance acceptance is deferred, not passed.**

## Code versus environment

The Mac is in NYC and the CUDA host is in Texas. Typical measured round-trip
latency is 52–54 ms. The current synchronous stage protocol requires a boundary
exchange per token, so approximately 13–14 seconds of network waiting across
256 steps is an architectural cost at that RTT, before compute and extra stalls.
A steady geographic latency is profileable; it does not itself explain packet loss.

The evidence supports a delivery-loss contribution to the observed timing spikes:

- [Raw TCP](milestone-5-planner/raw-tcp-comparison.json) reproduced spikes without
  model execution or gRPC. [Both payload directions](milestone-5-planner/raw-tcp-direction-comparison.json)
  were affected. A larger gRPC receive window did not give consistent improvement.
- [UDP probes](milestone-5-planner/udp-delivery-diagnostic.json) observed missing
  echoes without the long TCP recovery tails.
- [Clean endpoint captures](milestone-5-planner/udp-endpoint-confirmation.json)
  recorded all 600 probes at the Mac outgoing capture point, with the same five
  probes absent at the PC and from the returned echoes. Both capture helpers
  exited cleanly, and the temporary firewall rule was removed.

These observations do not locate a physical fault, prove Wi-Fi is responsible,
prove Tailscale is defective, or establish that all variability is unavoidable
because of geography. The unresolved boundary includes endpoint networking and
routing between the two regions. No global network settings were changed.

## Controlled runtime check

The new `test_delayed_boundary_preserves_tokens_deadline_and_recovery` runs two
real CPU workers through a local test relay. It separately delays activation
traffic and token feedback by 53 ms, with an additional 350 ms spike. Both cases:

- Preserve the exact four-token continuation from an unsplit baseline.
- Return `DEADLINE_EXCEEDED` when a 600 ms stall crosses a 200 ms application deadline.
- Release request/cache/workspace reservations and recover on the same deployment.
- Complete model unload with no remaining reservations.

This isolates runtime handling from the external WAN. It uses a small CPU fixture
and injected application delays, not packet-loss emulation or full-checkpoint
MLX/CUDA qualification. It found no additional runtime defect in these scenarios.
Existing native regression tests also cover blocked client writes, slow compute
unwind, explicit cancellation and downstream channel ownership.

The [preliminary audit](milestone-5-preacceptance-audit.md) records the four fixed
acceptance/activation evidence gaps. Historical native health fixes and full-model
results remain linked in the [validation report](milestone-5-planner.md).

Verification at closeout: all 118 Python/integration tests and seven targeted
native channel/deadline tests pass. Changed-file Ruff and full Pyright checks
with the project interpreter pass. No production native code changed in this
final pass.

## Deferred gates and restart point

The 10% reference-drift gate has not passed consistently on this WAN. No successful
fresh exhaustive sweep establishes observed regret and its seeded 95% bootstrap
upper bound at or below 15%. The separate final acceptance audit is also deferred.
No threshold was relaxed, failure discarded, or unsupported memory exclusion added.
The 4B and 32K targets remain unqualified.

Resume qualification when there is a justified transport improvement or a changed,
measurably stable operating condition. Use the updated package/executor identities,
fresh relevant profiles and setup/path snapshots, a new measured selection, and
five shuffled all-placement rounds (up to 15 for uncertainty), exact tokens and
fault/recovery checks. Complete the final audit before marking M5 accepted.

The recurring M5 investigation is stopped at this closeout. Further work should be
an explicit resumed qualification or a separately scoped transport investigation.

# Channel sweep deadline audit — 2026-09-13

The first channel-reuse sweep is not accepted. Its 54 placement jobs all
produced exact outputs. Across 56 jobs, all 168 full requests matched the
independent reference. The final reference job failed during the deadline
health phase after 124 tokens, returning CANCELLED rather than
DEADLINE_EXCEEDED. Explicit cancellation and its four-token recovery passed
immediately beforehand. Deadline cleanup and recovery were not verified in
that job; owned processes were subsequently retired and firewall cleanup passed.

The existing slow-compute regression covers deadline unwind with no active
client write. A separate backpressure regression deliberately expects CANCELLED:
the synchronous worker forces transport cancellation when the client-write flag
is still set after a roughly 100 ms grace period. These tests do not establish
whether a consuming client can encounter that branch during a transient delay.

The saved failure lacks watchdog branch telemetry. Therefore the blocked-write
fallback is a hypothesis, not a demonstrated cause of the observed failure.
Other paths to inspect include server transport cancellation and cancellation
of the active request before the deadline.

An isolated diagnostic source snapshot adds watchdog trigger and forced-cancel
logging, including request ID, cancellation flags, active-write state and time
relative to the application deadline. The qualification binaries remain frozen.
The diagnostic will first exercise the existing deadline regressions, then
repeat the bounded full-checkpoint health pilot. Its outputs cannot replace
acceptance evidence. Any resulting runtime fix must preserve bounded cleanup
for stalled consumers as well as precise deadline status for consuming clients.


## Downstream regression and fix

The isolated full-checkpoint diagnostic reproduced the deadline failure during
its first MLX-to-CUDA health iteration. The generation watchdog logged no
trigger for that deadline request, which points away from its blocked-write
fallback. This trace does not directly identify the uninstrumented CUDA branch.

A new downstream slow-decode regression reproduces a concrete defect: the old
ExecutionService watchdog forces CANCELLED immediately at its application
deadline while compute is still unwinding. Request resources remain active when
the client receives that status. This is consistent with the full-checkpoint
failure and exposes a gap left by the earlier generation-only fix.

ExecutionService now marks active stream reads and writes. At a deadline, its
watchdog lets compute unwind and return DEADLINE_EXCEEDED; blocked stream I/O
retains the bounded transport-cancellation fallback. Each I/O operation marks
itself active before checking cancellation, preventing a new blocking operation
after the watchdog decides no interrupt is needed. Explicit cancellation
continues to interrupt transport immediately.

Isolated m5-stage-deadline and cuda-stage-deadline builds passed all 66 Mac
and 69 CUDA tests (211.44s and 360.35s). Full-checkpoint health validation
passed six exact requests, ten cancellation/deadline pairs and twenty exact
recovery prefixes, with cleanup verified in both directions. Timing validation
is running and must pass before refreshed
qualification evidence or another acceptance sweep. The earlier failed sweep
remains immutable and unaccepted.

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

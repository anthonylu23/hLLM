# Model and backend boundaries

The runtime should support additional model families incrementally, without a general
model graph engine or a major redesign.

## Responsibilities

- Model adapters own configuration validation, tensor names and ownership, shapes,
  positional encoding, normalization, attention and feed-forward semantics.
- Backend implementations execute supported model semantics on CPU, CUDA, or MLX,
  own device tensors and request state, and report their memory requirements.
- The common runtime owns deployment identity, admission, request lifecycle, transport,
  ordering, deadlines, cancellation, and token streaming. Model-specific branches belong
  inside adapters/backends, not the controller or transport.

Use a small explicit architecture registry. Validate architecture ID, revision, required
features, tensor shapes, and execution dtype before accepting an executable stage.
Preparation support and configured placement profiles do not establish execution support.
Advertised execution capabilities must match the backend implementation.

Qwen3 demonstrates this boundary: per-head Q/K normalization and projection widths
independent of the residual width fit the existing manifest and activation protocol.
Keep shared dense operations reusable; do not turn a Llama class into an unbounded
collection of switches for unrelated model families.

## Constraints to revisit with a concrete model

The current configuration, contiguous-layer placement, and KV memory formulas describe
dense autoregressive attention. More dense text decoders are the closest extension.
Mixture-of-experts may require expert placement and routing; recurrent or hybrid models
may require state other than a conventional KV cache; multimodal models may require
additional input encoders and boundary representations. These are future design work,
not capabilities implied by an architecture string.

Keep sequence state opaque to the common runtime. Add model-specific memory descriptions
when a real architecture requires them rather than teaching the planner unrelated
formulas today. Framework objects must never cross the transport boundary.

## Stage contract

The common stage interface accepts token IDs or F16 boundary bytes and returns F16
boundary bytes or a sampled token. Embedding, layers, and sampling remain inside one
backend operation so GPU intermediates can stay on device. CPU reference inspection APIs
are separate from that contract. Factories supply truthful execution capabilities and
validate load-time allocations; runtime admission checks host, device and pinned-host
reservations separately, with a single unified-memory budget for MLX including transport.
Pinned memory also counts toward host memory. Backend state and
pending device operations must finish safely before buffers can be released.

## Regression cases

The tiny Llama and Qwen3 fixtures are regression cases for every backend:
[CPU](milestone-1.md), [CUDA](milestone-2.md) and [MLX](milestone-3.md) all implement this
stage contract against the shared dense checkpoint metadata. Select the next model family
with the user before widening the stage contract.

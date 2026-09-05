# Milestone 0 implementation

Milestone 0 produces a deterministic deployment plan without starting native workers.

## Implemented boundaries

- `proto/` defines the versioned model, profile, placement, control, execution, and
  telemetry wire contracts.
- `hllm prepare` strictly validates one Llama-compatible model and reads only Safetensors
  headers unless full hashing is requested.
- Worker profiles model CUDA device memory, ordinary host memory, pinned host allocations,
  and Apple unified memory as distinct domains.
- `hllm plan` enumerates every split in both worker orders, rejects candidates that exceed
  their primary memory budget, and retains all candidate diagnostics.
- `hllm explain` renders the persisted report without recomputing placement.

Artifact digests exclude local filesystem paths. A source model should always be identified
by a repository ID and immutable revision when a manifest is used as a golden artifact.

## Current estimation limits

Feasibility mode ranks maximum primary-memory pressure and then pressure imbalance.
Estimated mode currently incorporates directional boundary-transfer estimates but not
per-layer compute. Consequently, it must not be described as measured automatic placement.
Measured dry-load memory and performance profiles belong to Milestone 5.

The planner counts tied token embeddings on both boundary stages because stage zero needs
the embedding lookup and the final stage needs the LM head. That duplication is explicit in
the deployment plan.

The test suite includes a sparse Falcon3-shaped fixture with the checkpoint's real tensor
dimensions and logical size. It validates 22-layer classification and all 42 two-worker
candidates without committing or physically allocating the multi-gigabyte payload. It does
not replace validation against the pinned upstream weight bytes.

## Hardware test pair

The initial profiles describe:

- An 18 GiB Apple M3 Pro MacBook Pro using an MLX worker.
- An 8 GiB RTX 3060 Ti on `anthonypc` using a CUDA worker.

The Tailscale route was direct in both directions during the 2026-09-03 validation. The
checked-in link profile records point-in-time RTTs and conservative effective throughput
over SSH/Tailscale. These measurements are suitable for Milestone 0 planning exercises,
but are not a replacement for payload-specific stage-transport benchmarks.

The pinned Falcon3 checkpoint, generated manifest, planner result, and fresh-clone Fedora
test are recorded in [the Milestone 0 validation report](validation/milestone-0.md).

## Next steps

1. Add payload-specific link qualification for prefill and decode transfers.
2. Complete Milestone 1 by connecting the existing native CPU kernels and control
   service to stage execution; see [the quality review](code-quality-review.md).
3. As a stretch goal, add a `smollm3.v1` preparation adapter without changing the planner
   or artifact shapes.

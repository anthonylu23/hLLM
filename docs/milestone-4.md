# Milestone 4 — full-checkpoint MLX/CUDA qualification

The native pipeline has run the complete, unquantized `Qwen/Qwen3-0.6B` checkpoint
across Apple Silicon MLX and NVIDIA CUDA workers on separate Tailscale machines.
Both stage orders generated 256 tokens over direct and DERP connections. All four
split runs matched the independent F32 Transformers reference token for token.
Each process loaded only its assigned 14 transformer layers, plus its endpoint
tensors. The complete checkpoint files remain available on each machine's disk.

See the [qualification report](validation/full-checkpoint-cross-machine.md) for
the pinned revision, numerical thresholds, timings, memory measurements, faults,
and limitations. The [refresh on merged Milestone 3](validation/milestone-4-refresh.json)
repeats the probes, both transport paths, and fault recovery. This qualifies the Milestone 4 transport/execution criterion on
0.6B; the original Qwen3-4B-Base target still needs a separate physical-fit assessment.

## Checkpoint compatibility change

This real checkpoint serializes both the embedding and an identical `lm_head.weight`
despite declaring tied embeddings. The shared dense loader now accepts that form
only after checking both tensors' metadata and comparing their payloads in bounded
chunks. The final stage uses the embedding array for projection and retains no
second resident copy. Differing copies are rejected. The first stage does not read
the unowned head. CPU regression tests cover all supported checkpoint storage dtypes.

## Reproduce

Build the [MLX worker](milestone-3.md) on the Mac and the
[CUDA worker](milestone-2.md) on Linux. The validation scripts are opt-in and do not
download models or start workers themselves. The independent reference environment
is separate from the hLLM runtime; Python never computes or schedules native stages.

Download the same revision into a self-contained directory on each machine. Native
loading deliberately rejects snapshot symlinks that resolve outside the model root.
Use the HF CLI's local-directory mode, or materialize a cached snapshot with
dereferencing/reflink copying. Do not relax the loader's path checks.

```bash
CHECKPOINT_REV=c1899de289a04d12100db370d81485cdf75e47ca
uvx --from huggingface_hub hf download Qwen/Qwen3-0.6B \
  config.json generation_config.json model.safetensors tokenizer.json \
  tokenizer_config.json vocab.json merges.txt \
  --revision "$CHECKPOINT_REV" --local-dir build/models/Qwen3-0.6B
uvx --from huggingface_hub hf cache verify Qwen/Qwen3-0.6B \
  --revision "$CHECKPOINT_REV" --local-dir build/models/Qwen3-0.6B
uv run hllm prepare build/models/Qwen3-0.6B \
  --model-id Qwen/Qwen3-0.6B --revision "$CHECKPOINT_REV" \
  --output build/qualification/manifest.json
```

On the CUDA machine, with sufficient free GPU and host memory:

```bash
uv venv build/reference-env
uv pip install --python build/reference-env/bin/python \
  'torch==2.13.0' 'transformers==4.57.6'
build/reference-env/bin/python scripts/validation/checkpoint_reference.py \
  build/models/Qwen3-0.6B build/qualification/reference-f32.json
```

Copy that reference JSON and the prepared manifest to the controller machine.
The reference records exact chat-template token IDs, layer snapshots, full last-row
logits, and 256 greedy tokens with stop IDs disabled. It uses eager attention, disables
TF32 and reduced-precision FP16 reduction, and never loads remote model code.

For a standalone full-model numerical probe, generate a load specification with
`checkpoint_run.py --workers mlx --probe-spec-only` (or `--workers cuda`, optionally
`--dtype F32`). Then invoke the matching optional native target:

```bash
uv run python scripts/validation/checkpoint_run.py \
  build/qualification/manifest.json build/qualification/reference-f32.json \
  build/qualification/mlx-load.json --workers mlx --probe-spec-only
build/native/mlx/tests/cpp/hllm-checkpoint-mlx \
  build/models/Qwen3-0.6B build/qualification/mlx-load.json \
  build/qualification/reference-f32.json build/qualification/mlx-probe.json
```

Start one worker on each machine, binding to its own tailnet address. Set the shell
variables below to those addresses. These example budgets are the tested usable
budgets, not automatically calculated physical limits:

```bash
# Mac
build/native/mlx/cpp/hllm-worker-mlx --listen "$MAC_TAILNET_IP:50161" \
  --worker-id mlx --model-root build/models/Qwen3-0.6B \
  --memory-limit-bytes 4294967296
# Linux; use the library environment required by its CUDA build.
build/cuda/cpp/hllm-worker-cuda --listen "$CUDA_TAILNET_IP:50163" \
  --worker-id cuda --model-root build/models/Qwen3-0.6B \
  --memory-limit-bytes 2147483648 --device-memory-limit-bytes 6442450944
```

Permit only the required peer/port in the machines' firewalls. On the controller:

```bash
uv run python scripts/validation/checkpoint_run.py \
  build/qualification/manifest.json build/qualification/reference-f32.json \
  build/qualification/mlx-cuda.json --workers mlx cuda --split 14 --require-exact \
  --mlx-endpoint "$MAC_TAILNET_IP:50161" --cuda-endpoint "$CUDA_TAILNET_IP:50163"
```

Repeat with `--workers cuda mlx` and separate output files. The runner uses a fixed,
explicit split with a content-derived plan digest, checks sequence cleanup after
every request and model cleanup after unload, and records memory, TTFT, total elapsed
time, generated IDs, and reference common-prefix length. Completion and exact token
agreement are separate results; `--require-exact` also fails on a differing
256-token continuation after saving the report. This is not automatic placement.

`checkpoint_faults.py` takes the same manifest/reference/output and explicit endpoints
and checks cancellation after the first token, deadlines during decode, oversized
request rejection, and fresh-request recovery in both orders. `memory_watch.py PID
OUTPUT --cuda` samples one owned process's RSS and optional NVIDIA process memory;
omit `--cuda` on the Mac. Set its duration long enough to cover the complete run.

DERP qualification requires observing an actual relayed path for the entire run.
An application proxy alone is not evidence of DERP. The recorded run temporarily
blocked direct Tailscale UDP only to/from the peer's public addresses, with an
independent rollback timer installed first. It verified DERP pings before and after,
packet-drop counters, and restored direct connectivity. Network-specific fault rules
are intentionally not installed by these scripts.

## Remaining work

Milestone 5 adds measured placement and split selection. Account for allocator caches
and framework/context overhead as well as model reservations. CUDA currently acquires
a pooled stream per stage load; repeated assignments can retain substantial cached
VRAM across the stream pool. Plan GPU sharing accordingly or restart an idle worker
before allocating another large process. The qualification report records the actual
coexistence OOM encountered and the successful isolated rerun.

This milestone does not qualify the 4B checkpoint, long-context limits, quantization,
BF16 execution, pinned MLX/CUDA transport, batching, optimized attention, application
mTLS, or throughput under controlled idle-machine conditions.

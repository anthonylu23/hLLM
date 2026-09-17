# Milestone 6 — concurrent serving and native scheduling

Status: serving, configurable sampling, chunked prefill, bounded decode batching and
bounded weight conversion are implemented. Greedy decoding remains the default.
See [qualification results](validation/milestone-6.md) for the precise workloads and
remaining limits. The [PR review follow-up](validation/milestone-6-pr-review.md) covers
allocation-failure handling and streaming-retention regressions. M5 WAN performance
acceptance remains deferred.

## Serving

A single HTTP process owns one persistent deployment. It provides text/chat completions,
SSE streaming, model/deployment discovery, health and Prometheus-format metrics.
Each native request owns independent KV state and sampling state. Native admission sums
weights, every request's KV cache and workspace in each memory domain. HTTP admission
uses a bounded FIFO queue; queued HTTP requests do not reserve native KV buffers.

Start **each participating worker** with its existing identity, model root and memory
budgets, plus `--max-active-requests 4 --max-cached-tokens 4096`. The second limit is a
ceiling on the sum of reserved prompt-plus-output tokens; it does not add memory.
Workers default to one active request and memory-based token admission when the token
ceiling is omitted.

With a prepared manifest, explicit feasible plan, endpoints and matching local tokenizer:

```bash
uv run hllm serve \
  --manifest build/model.manifest.json \
  --plan build/model.deployment-plan.json \
  --workers build/workers.yaml \
  --tokenizer-root /models/Qwen3-0.6B \
  --model-name hllm \
  --max-active-requests 4 --max-queued-requests 16 --timeout 60
```

The server defaults to `127.0.0.1:8000`; host and port are configurable for the trusted
private deployment environment. Workers must already be running. Multiple HTTP processes
must not share stage assignments. Shutdown cancels calls and unloads stages. The command
does not download model or tokenizer assets.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hllm","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true,"temperature":0.8,"top_p":0.95,"seed":42}'
```

Use `/v1/completions` with a string `prompt` for models without a chat template. Chat
supports text-only `system`, `user` and `assistant` messages, using local
`chat_template.jinja` or the default template in `tokenizer_config.json`. Sandboxed
Jinja templates fail explicitly on unsupported extensions. Plain completion tokenization
adds special tokens; rendered chat is encoded without adding another set. Incremental
text decoding preserves UTF-8 boundaries.

## Sampling and probability metadata

The user approved greedy-first serving, then the sampling follow-up on 2026-09-15.
The current subset supports:

- `temperature`: 0–100, default 0. Zero preserves the original greedy path.
- `top_p`: greater than 0 and at most 1, default 1.
- `top_k`: 0 disables filtering; otherwise at most the model vocabulary size.
- `seed`: optional unsigned 64-bit integer. Without a seed, non-default sampling
  resolves one on the driver and sends it to the final stage.
- `n`: exactly 1.
- Completions: `logprobs` from 0 to 5, or null/omitted to disable.
- Chat: `logprobs: true`, with optional `top_logprobs` from 0 to 5.

Positive-temperature sampling applies temperature, top-k, then top-p and renormalizes.
Ties use the lowest token ID. A counter-based random draw depends only on the request
seed and generated-token index; interleaving and prefill chunk count do not consume
another request's random stream. Reproduction is qualified for a fixed backend and
configuration. Floating-point differences can change sampled continuations across
backends or execution shapes.

The [focused validation sweep](validation/milestone-6-sweep.md) reproduced this limit:
64-token sampled requests matched sequential references with serial dispatch, but
dynamic batching changed near-tied probabilities and produced different seeded tokens
in both MLX/CUDA stage orders. Use `--max-decode-batch 1` when comparing seeded
continuations within the same backend configuration. A seed does not guarantee identical
text when concurrent arrivals change batch composition.

Sampling runs on the final native worker. The initial non-greedy/probability path copies
only final logits to bounded host buffers and uses shared C++ selection code. All three
backends include these buffers in workspace admission. Greedy requests without probability
metadata retain native accelerator argmax. Device-resident sampling is a later optimization.

Log probabilities describe the **original model softmax, before temperature or filtering**.
Metadata covers consumed native tokens, including tokens consumed for stop detection;
text stop filtering affects rendered text. Completion offsets refer to the unfiltered
decoded stream. Token display strings are decoded individually and may contain replacement
characters for partial UTF-8 tokens; chat `bytes` is null rather than invented byte data.
No full-vocabulary logits leave the worker. Probability metadata is bounded to one selected
token and five alternatives per generated token.

Other supported options are `max_tokens` (default 32, range 1–1,000,000), `stream`,
`stream_options.include_usage`,
`stop` (up to four nonempty strings of at most 256 characters), and `stop_token_ids`.
At most 32 explicit stop token IDs are accepted. Prompts and individual chat messages
are limited to 1,000,000 characters, with at most 256 messages per chat request; the
1 MiB request-body bound can impose a tighter limit. Prompt plus requested output must
also fit the model context. Serving timeouts and native application deadlines are
bounded to one hour.
Native EOS IDs are the default; an explicit empty token-stop list disables them. Stop
strings are removed from text even across token boundaries. Usage counts tokens consumed
for stop detection. Finish reasons are `stop` and `length`. Unknown options, tools and
unsupported values return an OpenAI-shaped HTTP 400 error.

## Chunked prefill and decode batching

These optimizations are **opt-in**:

- HTTP server: `--prefill-chunk-tokens 128` divides prompts into bounded chunks.
  Default 0 preserves whole-prompt prefill. Request KV reservation still covers the
  complete prompt and maximum output; chunking does not claim reduced KV capacity.
- Native workers: `--max-decode-batch 4` combines already-ready decode work, up to 8
  requests and no more than `--max-active-requests`. Default 1 preserves serial dispatch.
  There is no extra batch-collection delay.

Each stage has FIFO prefill and decode queues. After at most eight decode work items,
a waiting prefill chunk gets a turn. Intermediate chunks update KV and send explicit
prefill acknowledgments. Only the final chunk invokes the LM head and selects the first
output token. Phase, frame order, context positions and acknowledgment identities are
validated separately from generated-token counts.

MLX and pageable CUDA batch embedding/projection/MLP/head operations across ready decode
requests. Attention and KV updates remain per sequence, with independent context positions;
there is no padded or paged KV layout. CPU and pinned CUDA retain serial execution.
One RPC caller performs a native dispatch; other batch members wait for their own results.
The stage scheduler releases compute ownership before any network writes.

An in-flight GPU batch is non-preemptible. Cancellation of a member does not abort its
peers; buffers stay owned until compute completes, and the cancelled member's output is
discarded. Corrupt input or a backend failure fails the affected native dispatch. Tests
cover differing sequence lengths, seeded sampling, queue fairness and cleanup.

New workers advertise sampling, probability and chunking support. Serving rejects enabled
features on incompatible workers. New binaries invalidate older binary-bound profile
evidence. Sampling/chunking are rejected with existing measured bundles, whose schemas do
not qualify these settings; use explicit feasible plans until profiles cover the new modes.

## Backpressure and cleanup

A full HTTP queue returns 429. Queue time counts toward the deadline; expiry returns 504.
Native capacity rejection returns 429 before headers or an SSE error after streaming
starts. Other native failures return 503. Streaming errors include an `error` object and
`[DONE]` when the connection remains writable.

JSON bodies are limited to 1 MiB before parsing. Streaming consumes native events directly,
with no unbounded producer queue. Emitted probability records and response text pieces
are only accumulated for non-streaming responses; incremental decoding retains its
own token/text history for final decoding verification. Slow clients retain their slot
and native deadline.
Disconnect cancels the generation RPC, asks each worker to cancel the request, and polls
its active request ID until retirement. This cleanup is shielded from repeated ASGI
cancellation before reusing the HTTP slot. Unconfirmed cleanup makes health and admissions
fail closed until restart, including requests already waiting in the queue.

`/v1/models`, `/v1/deployments`, `/health` and `/metrics` expose configuration, reachability,
request outcomes, token counts, active/queued requests, aggregate time-to-first-token and
inter-token intervals, native queue wait and depth, compute steps, actual decode-batch
counts/largest size, and cache/workspace reservations. Timing sums and counts are not
percentile estimates or M5 acceptance results.

## Bounded loading and the 4B blocker

The shared decoder reads at most 1 MiB of source bytes per chunk, converts that chunk,
and compares any redundant tied-head payload chunk by chunk. CPU/CUDA still assemble a
full F32 tensor for their existing loaders. MLX assembles the destination F16/F32 host
tensor directly and copies it into MLX without full-tensor F32 conversion intermediates.

MLX load admission includes all resident weights, the largest destination host tensor,
bounded conversion/comparison buffers and fixed scratch. Source metadata and capacity
are checked before payload loading. Tests cover conversion across chunk boundaries,
late tied-head corruption, overflow, numerical parity and load rejection.

The [recorded 4B rejection](validation/milestone-5-memory.md#larger-target-preflight)
describes the previous loader. Its endpoint-only bound was 5.796 GiB. The new F16 endpoint
estimate is about 1.453 GiB, **excluding transformer weights and inference workspace**.
This addresses that conservative loading term; it is not measured 4B fit. No 4B payload
was loaded in this follow-up. Runtime weight offloading/streaming remains a separate
candidate if resident-weight capacity is still insufficient. Headroom and M5 WAN gates
remain unchanged.

## Next steps

1. Refresh measured placement profiles with explicit sampling, chunk and batch settings;
   collect longer controlled throughput/latency comparisons and longer memory soaks.
2. Measure a real 4B assignment with the bounded loader when a complete checkpoint and
   a feasible memory budget are available. Qualify full context separately.
3. Optimize device sampling and attention/KV batching if measurements justify them.
4. Keep M5's exhaustive WAN comparison deferred until its acceptance measurements pass;
   then proceed with Milestone 7's additional backend/stage work.

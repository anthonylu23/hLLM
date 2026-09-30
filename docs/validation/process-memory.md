# Process-memory instrumentation validation — 2026-09-19

Native memory probes and serving workers now share OS process accounting. macOS
reports physical footprint and its lifetime peak alongside RSS and MLX allocator
counters. Linux reports RSS and leaves physical footprint unavailable. See the
[contract and interpretation limits](../milestone-5-profiling.md#process-physical-footprint).

This change adds measurement evidence only. The conservative MLX physical-fit rule,
headroom, safety fraction and admission checks are unchanged. The previous
[4B qualification result](qwen3-4b.md) remains bounded execution evidence with an
unresolved serving-memory fit gate.

## Validation

- Apple Silicon CPU/MLX build succeeded.
- All 127 Python unit tests passed. New coverage checks artifact hash compatibility,
  tamper detection, consistent RSS copies, single-process identity, unavailable
  versus zero counters, older worker responses and unchanged fit results.
- All 12 selected CTest entries passed: process-memory accounting, worker control,
  CPU pipeline integration (33 Python process tests), and CPU/MLX memory profiles.
  Completed native profiles contain 21 observations covering seven phases over
  three load/run/unload cycles. Uniform and mixed-precision MLX profiles use schema 1.3.
- All seven MLX worker tests passed. Metrics are checked before load, with request
  reservations, after request cleanup and after model unload over 12 cycles for
  each Llama/Qwen3 and precision configuration. OS lifetime peaks do not reset.
- The same native process-memory allocation test compiled and passed on Fedora
  with GCC (`-std=c++20 -O1 -Wall -Wextra -Werror`). It checks resident-page growth,
  lifetime-peak retention and unavailable physical-footprint counters. This was
  a standalone utility test, not a new CUDA-worker qualification.
- Ruff, Pyright, generated-protobuf consistency and `git diff --check` passed.

The native allocation test touches a 16 MiB allocation. Model tests use tiny
fixtures; no 4B checkpoint was reloaded and no model worker remains running.

## Next evidence

The [September 23 instrumented 4B soak](qwen3-4b-footprint.md) now covers three
deployment cycles, twelve exact continuations, system guards and an unloaded VM
breakdown. It supports an explicit footprint-aware fit policy with unchanged allowances.
The [September 27 implementation](footprint-policy.md) adds this policy as an opt-in
and passes tiny-model checks; fresh 4B qualification is in the
[overnight backlog](../overnight-backlog.md). RSS-plus-allocator remains the default.

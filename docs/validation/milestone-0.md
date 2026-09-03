# Milestone 0 validation report

Validated on 2026-09-03 against the public repository and the intended first hardware pair.

## Real checkpoint

- Model: `tiiuae/Falcon3-3B-Base`
- Immutable revision: `092c29e3114ff551bd06b93b85424cdb525d227c`
- Format: two indexed Safetensors shards
- Parsed result: 22 transformer layers, 201 tensors, 6,455,310,336 tensor bytes
- Manifest digest: `1340e5b5dfce2462a3b28739aa7da82a52c499a1af08c29782ed2acf48d69f41`

The preparation command used full payload hashing. The resulting local SHA-256 values and
sizes matched the Hugging Face metadata at the immutable revision:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00002.safetensors` | 4,989,378,032 | `8f8a63e9867330c7f211e00479c51d76ca2c24523d40571c651182f270e40363` |
| `model-00002-of-00002.safetensors` | 1,465,955,608 | `dea5f9d277c03a9f04d80b2c1d065763ffae625f13ceacc05d2429bbb59362ea` |

The installed Hugging Face CLI predates its cache verification command, so upstream
verification used the immutable revision's `x-linked-etag` and `x-linked-size` response
metadata. Model weights and generated build artifacts remain intentionally ignored by Git.

## Placement result

The feasibility planner enumerated all 42 contiguous two-worker candidates. Fourteen were
rejected for exceeding RTX 3060 Ti device memory. The selected plan was:

- Stage 0: `mac-m3-pro`, layers `[0, 16)`
- Stage 1: `rtx-3060-ti`, layers `[16, 22)`
- Mac memory estimate: 7.07 GiB of 12.20 GiB usable, 58.0% pressure
- RTX memory estimate: 3.48 GiB of 6.45 GiB usable, 53.9% pressure

The reverse worker order at layer 6 had the same feasibility score and was retained as the
runner-up. Milestone 0 does not yet have measured per-layer compute profiles, so this tie is
expected and is not a throughput claim.

## Cross-platform reproducibility

A fresh clone of public commit `5576821b3c89c2bbf56dd23243aa9ebcbf35f2b8` was tested on
the remote machine:

- Fedora Linux 44, x86-64, kernel 7.1.9
- NVIDIA GeForce RTX 3060 Ti, 8,192 MiB, driver 610.57.04
- `uv` 0.11.29 with managed CPython 3.12.13
- Ruff formatting and lint: passed
- Pyright: 0 errors
- Pytest: 17 passed
- All seven protobuf schemas compiled with `grpcio-tools`

The machine had 3,720,171,520 bytes of host memory available during the check. An unrelated
existing workload was left untouched.

## Link observation

Tailscale reported a direct path in both directions. Point-in-time RTT was 58 ms from the
Mac to Fedora and 72 ms in the reverse direction. A 64 MiB stream over SSH/Tailscale took
10.95 seconds Mac-to-Fedora and 6.31 seconds Fedora-to-Mac, or approximately 6.1 MB/s and
10.6 MB/s respectively. The checked-in profile rounds these down to 6 MB/s and 10 MB/s.

This is an application-level smoke measurement, not a controlled transport benchmark. It
includes SSH overhead and does not characterize p50/p99 behavior or small decode payloads.
Payload-specific link qualification remains a later milestone.

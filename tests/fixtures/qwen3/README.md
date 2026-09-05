# Qwen3 fixtures

`config.json` and `model.safetensors.index.json` are from the Apache-2.0-licensed
[Qwen3-4B-Base checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Base/tree/906bfd4b4dc7f14ee4320094d8b41684abff8539).
`tensor-metadata.json` contains name/shape/dtype entries extracted from its three
Safetensors headers. Retrieved 2026-09-05; no weight payloads are included.

`tiny-reference.json` is generated locally using the pinned dependencies in
`scripts/generate_qwen3_reference.py`. It contains a deterministic, randomly initialized
then overwritten tiny Qwen3 model and independent Transformers outputs, not pretrained
checkpoint weights. See `docs/qwen3.md` for validation scope and regeneration.

"""Write a tiny deterministic Qwen3 checkpoint for the CPU pipeline demo.

These are synthetic oracle weights, not a pretrained language model.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    oracle = json.loads((root / "tests/fixtures/qwen3/tiny-reference.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    config = oracle["config"]
    config.update(max_position_embeddings=512, eos_token_id=None)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    header = {}
    payload = bytearray()
    for name, tensor in sorted(oracle["weights"].items()):
        if name == "lm_head.weight":
            continue  # Tied embedding supplies the head.
        start = len(payload)
        payload.extend(struct.pack(f"<{len(tensor['values'])}f", *tensor["values"]))
        header[name] = {
            "dtype": "F32",
            "shape": tensor["shape"],
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (args.output / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    print(args.output)


if __name__ == "__main__":
    main()

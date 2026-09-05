# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.7.1", "transformers==4.51.3"]
# ///
"""Regenerate the tiny CPU float32 Qwen3 oracle; no checkpoint download required.

Run with `uv run --script scripts/generate_qwen3_reference.py`.
The runtime test consumes the checked-in fixture without Torch or Transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    torch.set_num_threads(1)
    config = Qwen3Config(
        hidden_size=6,
        intermediate_size=10,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=11,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        sliding_window=None,
        use_sliding_window=False,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).float().eval()
    # Serialize every input weight, so C++ does not have to reproduce RNG or sin.
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32)
            values = torch.sin(values * 0.37 + index * 0.29)
            values = 0.8 + 0.3 * values if "norm" in name else 0.2 * values
            parameter.copy_(values.reshape(parameter.shape))

    tokens = torch.tensor([[1, 4, 2, 8, 3]])
    layer_outputs = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _args, output: layer_outputs.append(output[0].detach())
        )
        for layer in model.model.layers
    ]
    with torch.no_grad():
        output = model(tokens, use_cache=True)
        embeddings = model.model.embed_tokens(tokens)
    for handle in handles:
        handle.remove()

    def tensor(value: torch.Tensor) -> dict[str, object]:
        return {"shape": list(value.shape), "values": value.flatten().tolist()}

    fixture = {
        "producer": {"torch": torch.__version__, "transformers": transformers.__version__},
        "config": config.to_dict(),
        "weights": {name: tensor(value) for name, value in model.state_dict().items()},
        "embeddings": tensor(embeddings),
        "layer_outputs": [tensor(value) for value in layer_outputs],
        "logits": tensor(output.logits),
        "keys": [tensor(pair[0]) for pair in output.past_key_values],
        "values": [tensor(pair[1]) for pair in output.past_key_values],
    }
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/qwen3/tiny-reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()

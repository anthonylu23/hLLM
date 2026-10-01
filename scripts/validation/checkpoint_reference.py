"""Produce a pinned Transformers eager-attention oracle for checkpoint qualification.

Run in an isolated torch/transformers environment. Never loads remote Python code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import torch  # pyright: ignore[reportMissingImports] -- isolated reference environment
import transformers  # pyright: ignore[reportMissingImports]
from transformers import (  # pyright: ignore[reportMissingImports]
    AutoModelForCausalLM,
    AutoTokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument(
        "--plain", action="store_true", help="Plain completion tokenization for Base models"
    )
    parser.add_argument(
        "--gpu-layers", type=int, help="Keep the remaining layers on CPU via Accelerate"
    )
    parser.add_argument("--teacher-reference", type=Path)
    parser.add_argument("--continuation-reference", type=Path)
    parser.add_argument("--at-index", type=int, default=177)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an independent reference")
    if args.output_tokens <= 0 or (args.prompt_tokens is not None and args.prompt_tokens <= 0):
        parser.error("prompt/output token counts must be positive")
    config = json.loads((args.model / "config.json").read_text())
    if args.gpu_layers is not None and not 0 <= args.gpu_layers <= config["num_hidden_layers"]:
        parser.error("gpu-layers must be within the model layer count")
    checkpoint_files = {}
    for path in [args.model / "config.json", *sorted(args.model.glob("*.safetensors"))]:
        with path.open("rb") as stream:
            checkpoint_files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    checkpoint_digest = hashlib.sha256(
        json.dumps(checkpoint_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    device_map: dict[str, int | str] | None = None
    if args.gpu_layers is not None:
        device_map = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": 0}
        device_map.update(
            {
                f"model.layers.{i}": 0 if i < args.gpu_layers else "cpu"
                for i in range(config["num_hidden_layers"])
            }
        )
    model = cast(
        torch.nn.Module,  # AutoModel returns a Module; its generated typing is loose.
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32 if args.dtype == "f32" else torch.float16,
            attn_implementation="eager",
            local_files_only=True,
            **({"device_map": device_map, "low_cpu_mem_usage": True} if device_map else {}),
        ),
    ).eval()
    if device_map is None:
        model = model.to("cuda")

    def encode(prompt: str):
        if args.plain:
            return tokenizer.encode(prompt, add_special_tokens=True)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    teacher = json.loads(args.teacher_reference.read_text()) if args.teacher_reference else None
    continuation = (
        json.loads(args.continuation_reference.read_text())["long_generation"]
        if args.continuation_reference
        else None
    )
    if continuation and (teacher or not 0 <= args.at_index < len(continuation["generated_ids"])):
        parser.error("continuation requires an in-range output index and no teacher-reference")
    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Calculate 17 plus 25. Give the result and a short explanation.",
        "Explain why the sky appears blue in simple language.",
    ]
    if continuation:
        prompts = [continuation["prompt"]]
    layer_rows = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        layer_rows.append(hidden[0, -1].float().cpu().tolist())

    handles = [
        layer.register_forward_hook(capture)
        for layer in model.get_submodule("model.layers").children()
    ]
    result = {
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_files": checkpoint_files,
        "producer": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": args.dtype,
            "attention": "eager",
            "tf32": False,
            "plain_completion": args.plain,
            "device_map": device_map,
        },
        "cases": [],
    }
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            ids = encode(prompt)
            if continuation:
                ids = continuation["token_ids"]
            entry = {"prompt": prompt, "token_ids": ids, "steps": []}
            if continuation:
                entry["warmup"] = []
            cache = None
            position = 0
            for step in range(args.at_index + 1 if continuation else 3):
                if teacher:
                    ids = teacher["cases"][index]["steps"][step]["input_ids"]
                if continuation and step:
                    ids = [continuation["generated_ids"][step - 1]]
                layer_rows.clear()
                output = model(
                    input_ids=torch.tensor([ids], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                logits = output.logits[0, -1].float()
                best = torch.topk(logits, 2)
                if continuation and step < args.at_index:
                    entry["warmup"].append({"input_ids": ids, "position": position})
                else:
                    entry["steps"].append(
                        {
                            "input_ids": ids,
                            "position": position,
                            "logits": logits.cpu().tolist(),
                            "layers": list(layer_rows),
                            "argmax": int(logits.argmax()),
                            "top2_margin": float(best.values[0] - best.values[1]),
                        }
                    )
                cache = output.past_key_values
                position += len(ids)
                ids = [int(logits.argmax())]
            result["cases"].append(entry)
        for handle in handles:
            handle.remove()
        if continuation:
            result["continuation_index"] = args.at_index
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
            print(json.dumps({"output": str(args.output), "index": args.at_index}), flush=True)
            return
        prompt = (
            "Write a detailed story of at least 500 words about a robot exploring "
            "an abandoned observatory."
        )
        ids = encode(prompt)
        if args.prompt_tokens is not None:
            # Exact token IDs are authoritative. Repetition is deterministic and
            # avoids tokenizer-dependent string padding or approximate lengths.
            ids = (ids * ((args.prompt_tokens + len(ids) - 1) // len(ids)))[: args.prompt_tokens]
        if len(ids) + args.output_tokens > model.config.max_position_embeddings:
            parser.error("reference workload exceeds model context")
        initial = ids
        cache = None
        generated = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.output_tokens):
            output = model(
                input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
            cache = output.past_key_values
            ids = [token]
        torch.cuda.synchronize()
        result["long_generation"] = {
            "prompt": prompt,
            "token_ids": initial,
            "generated_ids": generated,
            "text": tokenizer.decode(generated),
            "seconds": time.perf_counter() - start,
            "stop_ids": [],
        }
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dtype": args.dtype,
                "tokens": len(generated),
                "seconds": result["long_generation"]["seconds"],
                "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Exercise the qualification harness against real workers before hardware runs."""

import asyncio
import json
from pathlib import Path

import pytest
from hllm_control.models import DType
from hllm_control.serving.tokenizer import TextTokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from scripts.validation.serving_soak import qualify
from tests.process_helpers import Workers, write_model


@pytest.mark.parametrize(
    "sampling",
    [
        None,
        {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 5,
            "seed": 42,
            "logprobs": 2,
        },
    ],
)
def test_serving_qualification_report(tmp_path: Path, sampling: dict | None) -> None:
    manifest = write_model(tmp_path)
    native = Tokenizer(
        models.WordLevel(
            {f"t{i}": i for i in range(manifest.config.vocabulary_size)}, unk_token="t0"
        )
    )
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    native.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    native.save(str(tmp_path / "tokenizer.json"))
    tokenizer = TextTokenizer(tmp_path, manifest.config.vocabulary_size)
    args = ("--max-active-requests", "2", "--max-cached-tokens", "1024")
    with Workers(tmp_path, extra_args=(args, args)) as workers:
        output = tmp_path / "result.json"
        result = asyncio.run(
            qualify(
                manifest,
                tokenizer,
                workers.endpoints,
                list(workers.endpoints),
                ["t1 t4 t2", "t3 t2 t1 t4"],
                rounds=2,
                output_tokens=256,
                levels=[1, 2],
                timeout=30,
                output=output,
                split=1,
                dtype=DType.F32,
                stagger_seconds=0,
                observation_interval=0.002,
                sampling=sampling,
            )
        )
        assert result["completed"]
        assert json.loads(output.read_text()) == result
        assert [level["peak_active_requests"] for level in result["levels"]] == [1, 2]
        assert all(level["after_unload"] for level in result["levels"])

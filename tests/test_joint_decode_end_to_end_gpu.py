from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch.distributed as dist

from joint_decode_gpu.config import JointDecodeConfig, JointDecodeModelConfig, JointDecodeSamplingConfig
from joint_decode_gpu.coordinator import Coordinator, run_joint_decode


@dataclass(frozen=True)
class StringCase:
    prompt_a: str
    chunks_a: tuple[str, ...]
    prompt_b: str
    chunks_b: tuple[str, ...]


@dataclass(frozen=True)
class EndToEndSetting:
    name: str
    microbatch_size: int
    top_k_a: int
    top_k_b: int
    max_num_batched_tokens: int | None
    cases: tuple[StringCase, ...]


SETTINGS = (
    EndToEndSetting(
        name="two_chunks_microbatch_two",
        microbatch_size=2,
        top_k_a=4,
        top_k_b=5,
        max_num_batched_tokens=None,
        cases=(
            StringCase("A prompt 0:", (" red apple", " on table"), "B prompt 0:", (" gato azul", " salta alto")),
            StringCase("A prompt 1:", (" blue stone", " near river"), "B prompt 1:", (" perro verde", " corre lejos")),
            StringCase("A prompt 2:", (" gold coin", " under sand"), "B prompt 2:", (" luna blanca", " brilla hoy")),
            StringCase("A prompt 3:", (" black bird", " above trees"), "B prompt 3:", (" sol rojo", " cae tarde")),
            StringCase("A prompt 4:", (" green leaf", " after rain"), "B prompt 4:", (" mar frio", " sube lento")),
        ),
    ),
    EndToEndSetting(
        name="three_chunks_microbatch_three",
        microbatch_size=3,
        top_k_a=3,
        top_k_b=3,
        max_num_batched_tokens=None,
        cases=(
            StringCase(
                "Case A0:",
                (" first", " middle words", " final bit"),
                "Case B0:",
                (" uno dos", " tres", " cuatro cinco"),
            ),
            StringCase(
                "Case A1:",
                (" small", " bright object", " lands"),
                "Case B1:",
                (" norte", " sur este", " oeste"),
            ),
            StringCase(
                "Case A2:",
                (" quiet", " silver path", " opens"),
                "Case B2:",
                (" alpha beta", " gamma", " delta"),
            ),
            StringCase(
                "Case A3:",
                (" warm", " copper light", " fades"),
                "Case B3:",
                (" piedra", " agua clara", " fuego"),
            ),
        ),
    ),
    EndToEndSetting(
        name="explicit_batched_tokens",
        microbatch_size=2,
        top_k_a=2,
        top_k_b=2,
        max_num_batched_tokens=256,
        cases=(
            StringCase("Explicit A0:", (" north wind", " returns"), "Explicit B0:", (" cielo gris", " vuelve")),
            StringCase("Explicit A1:", (" south star", " rises"), "Explicit B1:", (" bosque seco", " duerme")),
            StringCase("Explicit A2:", (" east road", " bends"), "Explicit B2:", (" rio largo", " canta")),
        ),
    ),
)


@pytest.mark.gpu
@pytest.mark.parametrize("setting", SETTINGS, ids=[setting.name for setting in SETTINGS])
def test_two_tokenizer_forced_string_chunks_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    setting: EndToEndSetting,
) -> None:
    model_a = _required_env_or_skip("JOINT_DECODE_GPU_TEST_MODEL_A")
    model_b = _required_env_or_skip("JOINT_DECODE_GPU_TEST_MODEL_B")

    try:
        from transformers import AutoTokenizer

        tokenizer_a = AutoTokenizer.from_pretrained(model_a, trust_remote_code=True)
        tokenizer_b = AutoTokenizer.from_pretrained(model_b, trust_remote_code=True)
        if tokenizer_a.get_vocab() == tokenizer_b.get_vocab():
            pytest.skip("test requires two models with different tokenizers")

        prompts_a = [case.prompt_a for case in setting.cases]
        prompts_b = [case.prompt_b for case in setting.cases]
        scripts_a = _scripts_by_rid(tokenizer_a, [case.chunks_a for case in setting.cases], setting.microbatch_size)
        scripts_b = _scripts_by_rid(tokenizer_b, [case.chunks_b for case in setting.cases], setting.microbatch_size)
        expected_text = [
            _decode_chunks(tokenizer_a, case_chunks)
            for case_chunks in [case.chunks_a for case in setting.cases]
        ]
        positions = {rid: 0 for rid in scripts_a}

        def scripted_select_for_rid(
            self: Coordinator,
            rid: str,
            entry_a: object,
            entry_b: object,
        ) -> tuple[list[int], list[int]]:
            del self, entry_a, entry_b
            position = positions[rid]
            positions[rid] = position + 1
            return scripts_a[rid][position], scripts_b[rid][position]

        monkeypatch.setattr(Coordinator, "_select_for_rid", scripted_select_for_rid)

        max_chunks = max(len(chunks) for chunks in scripts_a.values())
        config = JointDecodeConfig(
            model_a=JointDecodeModelConfig(
                model_path=model_a,
                gpu_index=0,
                max_model_len=128,
                gpu_memory_utilization=0.8,
                enable_prefix_caching=False,
                enforce_eager=True,
            ),
            model_b=JointDecodeModelConfig(
                model_path=model_b,
                gpu_index=1,
                max_model_len=128,
                gpu_memory_utilization=0.8,
                enable_prefix_caching=False,
                enforce_eager=True,
            ),
            sampling=JointDecodeSamplingConfig(
                max_tokens=max_chunks,
                top_k_a=setting.top_k_a,
                top_k_b=setting.top_k_b,
                microbatch_size=setting.microbatch_size,
                max_num_batched_tokens=setting.max_num_batched_tokens,
                barrier_timeout_s=60.0,
                seed=0,
                stop=(),
            ),
        )

        outputs = run_joint_decode(config, prompts_a, prompts_b, select_token=lambda *_args, **_kwargs: 0)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    assert [output.text for output in outputs] == expected_text
    assert [
        case.prompt_a + output.text
        for case, output in zip(setting.cases, outputs, strict=True)
    ] == [
        case.prompt_a + expected
        for case, expected in zip(setting.cases, expected_text, strict=True)
    ]
    assert all(output.finish_reason == "stop" for output in outputs)
    assert positions == {rid: len(chunks) for rid, chunks in scripts_a.items()}


def _required_env_or_skip(name: str) -> str:
    import os

    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"set {name}")
    return value


def _scripts_by_rid(
    tokenizer: Any,
    all_chunks: list[tuple[str, ...]],
    microbatch_size: int,
) -> dict[str, list[list[int]]]:
    scripts: dict[str, list[list[int]]] = {}
    for case_index, chunks in enumerate(all_chunks):
        chunk_index = case_index // microbatch_size
        index_in_chunk = case_index % microbatch_size
        rid = f"jd-c{chunk_index}-r{index_in_chunk:06d}"
        scripts[rid] = [_encode_nonempty(tokenizer, chunk) for chunk in chunks]
    return scripts


def _encode_nonempty(tokenizer: Any, text: str) -> list[int]:
    token_ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    if not token_ids:
        raise ValueError(f"chunk tokenized to an empty list: {text!r}")
    return token_ids


def _decode_chunks(tokenizer: Any, chunks: tuple[str, ...]) -> str:
    token_ids = [
        token_id
        for chunk in chunks
        for token_id in tokenizer.encode(chunk, add_special_tokens=False)
    ]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

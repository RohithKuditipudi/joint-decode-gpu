from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch.distributed as dist

from joint_decode_gpu.config import JointDecodeConfig, JointDecodeModelConfig, JointDecodeSamplingConfig
from joint_decode_gpu.coordinator import Coordinator, JointDecoder, run_joint_decode


@dataclass(frozen=True)
class StringCase:
    prompt_a: str
    chunks_a: tuple[str, ...]
    prompt_b: str
    chunks_b: tuple[str, ...]


@dataclass(frozen=True)
class EndToEndSetting:
    name: str
    max_microbatch_size: int
    top_k_a: int
    top_k_b: int
    max_num_batched_tokens: int | None
    cases: tuple[StringCase, ...]


@dataclass
class ScriptedCase:
    model_a: str
    model_b: str
    tokenizer_a: Any
    tokenizer_b: Any
    prompts_a: list[str]
    prompts_b: list[str]
    scripts_a: dict[str, list[list[int]]]
    scripts_b: dict[str, list[list[int]]]


_RAMP_FILLER = " ".join(f"filler{index}" for index in range(30))

SETTINGS = (
    EndToEndSetting(
        name="two_chunks_microbatch_two",
        max_microbatch_size=2,
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
        max_microbatch_size=3,
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
        max_microbatch_size=2,
        top_k_a=2,
        top_k_b=2,
        max_num_batched_tokens=256,
        cases=(
            StringCase("Explicit A0:", (" north wind", " returns"), "Explicit B0:", (" cielo gris", " vuelve")),
            StringCase("Explicit A1:", (" south star", " rises"), "Explicit B1:", (" bosque seco", " duerme")),
            StringCase("Explicit A2:", (" east road", " bends"), "Explicit B2:", (" rio largo", " canta")),
        ),
    ),
    # Long prompts against a budget of one max_model_len force the coordinator
    # to admit the window over several decision rounds instead of all at once.
    EndToEndSetting(
        name="budget_limited_admission_ramp",
        max_microbatch_size=4,
        top_k_a=2,
        top_k_b=2,
        max_num_batched_tokens=128,
        cases=(
            StringCase(
                f"A ramp 0 {_RAMP_FILLER}:",
                (" red apple", " on table"),
                f"B ramp 0 {_RAMP_FILLER}:",
                (" gato azul", " salta alto"),
            ),
            StringCase(
                f"A ramp 1 {_RAMP_FILLER}:",
                (" blue stone", " near river"),
                f"B ramp 1 {_RAMP_FILLER}:",
                (" perro verde", " corre lejos"),
            ),
            StringCase(
                f"A ramp 2 {_RAMP_FILLER}:",
                (" gold coin", " under sand"),
                f"B ramp 2 {_RAMP_FILLER}:",
                (" luna blanca", " brilla hoy"),
            ),
            StringCase(
                f"A ramp 3 {_RAMP_FILLER}:",
                (" black bird", " above trees"),
                f"B ramp 3 {_RAMP_FILLER}:",
                (" sol rojo", " cae tarde"),
            ),
            StringCase(
                f"A ramp 4 {_RAMP_FILLER}:",
                (" green leaf", " after rain"),
                f"B ramp 4 {_RAMP_FILLER}:",
                (" mar frio", " sube lento"),
            ),
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
        scripts_a = _scripts_by_rid(tokenizer_a, [case.chunks_a for case in setting.cases])
        scripts_b = _scripts_by_rid(tokenizer_b, [case.chunks_b for case in setting.cases])
        _append_eos(scripts_a, tokenizer_a)
        _append_eos(scripts_b, tokenizer_b)
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
                max_tokens_a=max(len(_flatten_steps(chunks)) for chunks in scripts_a.values()),
                max_tokens_b=max(len(_flatten_steps(chunks)) for chunks in scripts_b.values()),
                top_k_a=setting.top_k_a,
                top_k_b=setting.top_k_b,
                barrier_timeout_s=60.0,
                seed=0,
                stop=(),
                max_microbatch_size=setting.max_microbatch_size,
                max_num_batched_tokens=setting.max_num_batched_tokens,
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


@pytest.mark.gpu
def test_large_window_initialization_smoke() -> None:
    """A 1024-request cap with the default token budget must initialize.

    The retired full-window budget derivation (max_microbatch_size *
    max_model_len = 2,097,152 batched tokens) made vLLM fail during startup
    profiling before the handshake.
    """
    model_a = _required_env_or_skip("JOINT_DECODE_GPU_TEST_MODEL_A")
    model_b = _required_env_or_skip("JOINT_DECODE_GPU_TEST_MODEL_B")

    config = JointDecodeConfig(
        model_a=JointDecodeModelConfig(
            model_path=model_a,
            gpu_index=0,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=False,
            enforce_eager=True,
        ),
        model_b=JointDecodeModelConfig(
            model_path=model_b,
            gpu_index=1,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=False,
            enforce_eager=True,
        ),
        sampling=JointDecodeSamplingConfig(
            max_tokens_a=8,
            max_tokens_b=8,
            top_k_a=2,
            top_k_b=2,
            barrier_timeout_s=60.0,
            seed=0,
            stop=(),
        ),
    )
    try:
        with JointDecoder(config, select_token=lambda *_args, **_kwargs: 0) as decoder:
            assert decoder._max_live_requests_a >= 1
            assert decoder._max_live_requests_b >= 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_a_side_max_tokens_returns_expected_a_text(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _scripted_case()
    expected_a = {
        rid: _decode_token_prefix(case.tokenizer_a, steps, 3)
        for rid, steps in case.scripts_a.items()
    }

    outputs = _run_scripted_case(
        monkeypatch,
        case,
        max_tokens_a=3,
        max_tokens_b=16,
        stop=(),
    )

    assert [output.text for output in outputs] == _ordered_expected(expected_a, outputs)


@pytest.mark.gpu
def test_b_side_max_tokens_force_stops_a(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _scripted_case()
    expected_a = {
        rid: _decode_token_prefix(case.tokenizer_a, case.scripts_a[rid], 3)
        for rid in case.scripts_a
    }

    outputs = _run_scripted_case(
        monkeypatch,
        case,
        max_tokens_a=16,
        max_tokens_b=3,
        stop=(),
    )

    assert [output.text for output in outputs] == _ordered_expected(expected_a, outputs)


@pytest.mark.gpu
def test_a_side_stop_string_returns_expected_a_text(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _scripted_case()
    stop = "STOP"
    case.scripts_a = {
        rid: _token_steps(case.tokenizer_a, f" red blue {stop} green yellow {index}")
        for index, rid in enumerate(case.scripts_a)
    }
    expected_a = {
        rid: _decode_until_stop(case.tokenizer_a, steps, stop)
        for rid, steps in case.scripts_a.items()
    }

    outputs = _run_scripted_case(
        monkeypatch,
        case,
        max_tokens_a=16,
        max_tokens_b=16,
        stop=(stop,),
    )

    assert [output.text for output in outputs] == _ordered_expected(expected_a, outputs)


@pytest.mark.gpu
def test_b_side_stop_string_force_stops_a(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _scripted_case()
    stop = "STOP"
    case.scripts_b = {
        rid: _token_steps(case.tokenizer_b, f" uno dos {stop} cuatro cinco {index}")
        for index, rid in enumerate(case.scripts_b)
    }
    expected_a = {
        rid: _decode_token_prefix(
            case.tokenizer_a,
            case.scripts_a[rid],
            _first_stop_step(case.tokenizer_b, case.scripts_b[rid], stop),
        )
        for rid in case.scripts_a
    }

    outputs = _run_scripted_case(
        monkeypatch,
        case,
        max_tokens_a=16,
        max_tokens_b=16,
        stop=(stop,),
    )

    assert [output.text for output in outputs] == _ordered_expected(expected_a, outputs)


def _scripted_case() -> ScriptedCase:
    model_a = _required_env_or_skip("JOINT_DECODE_GPU_TEST_MODEL_A")
    model_b = _required_env_or_skip("JOINT_DECODE_GPU_TEST_MODEL_B")

    from transformers import AutoTokenizer

    tokenizer_a = AutoTokenizer.from_pretrained(model_a, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b, trust_remote_code=True)
    if tokenizer_a.get_vocab() == tokenizer_b.get_vocab():
        pytest.skip("test requires two models with different tokenizers")

    prompts_a = [f"A stop case {index}:" for index in range(5)]
    prompts_b = [f"B stop case {index}:" for index in range(5)]
    scripts_a = {
        f"jd-r{index:06d}": _token_steps(tokenizer_a, f" red blue green yellow purple orange {index}")
        for index in range(len(prompts_a))
    }
    scripts_b = {
        f"jd-r{index:06d}": _token_steps(tokenizer_b, f" uno dos tres cuatro cinco seis {index}")
        for index in range(len(prompts_b))
    }
    return ScriptedCase(
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        prompts_a=prompts_a,
        prompts_b=prompts_b,
        scripts_a=scripts_a,
        scripts_b=scripts_b,
    )


def _run_scripted_case(
    monkeypatch: pytest.MonkeyPatch,
    case: ScriptedCase,
    *,
    max_tokens_a: int,
    max_tokens_b: int,
    stop: tuple[str, ...],
) -> list[Any]:
    positions = {rid: 0 for rid in case.scripts_a}

    def scripted_select_for_rid(
        self: Coordinator,
        rid: str,
        entry_a: object,
        entry_b: object,
    ) -> tuple[list[int], list[int]]:
        del self, entry_a, entry_b
        position = positions[rid]
        positions[rid] = position + 1
        return case.scripts_a[rid][position], case.scripts_b[rid][position]

    monkeypatch.setattr(Coordinator, "_select_for_rid", scripted_select_for_rid)

    config = JointDecodeConfig(
        model_a=JointDecodeModelConfig(
            model_path=case.model_a,
            gpu_index=0,
            max_model_len=128,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=False,
            enforce_eager=True,
        ),
        model_b=JointDecodeModelConfig(
            model_path=case.model_b,
            gpu_index=1,
            max_model_len=128,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=False,
            enforce_eager=True,
        ),
        sampling=JointDecodeSamplingConfig(
            max_tokens_a=max_tokens_a,
            max_tokens_b=max_tokens_b,
            top_k_a=3,
            top_k_b=3,
            barrier_timeout_s=60.0,
            seed=0,
            stop=stop,
            max_microbatch_size=2,
            max_num_batched_tokens=None,
        ),
    )
    try:
        return run_joint_decode(config, case.prompts_a, case.prompts_b, select_token=lambda *_args, **_kwargs: 0)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _ordered_expected(expected_by_rid: dict[str, str], outputs: list[Any]) -> list[str]:
    return [
        expected_by_rid[f"jd-r{index:06d}"]
        for index in range(len(outputs))
    ]


def _required_env_or_skip(name: str) -> str:
    import os

    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"set {name}")
    return value


def _scripts_by_rid(
    tokenizer: Any,
    all_chunks: list[tuple[str, ...]],
) -> dict[str, list[list[int]]]:
    scripts: dict[str, list[list[int]]] = {}
    for case_index, chunks in enumerate(all_chunks):
        rid = f"jd-r{case_index:06d}"
        scripts[rid] = [_encode_nonempty(tokenizer, chunk) for chunk in chunks]
    return scripts


def _append_eos(scripts: dict[str, list[list[int]]], tokenizer: Any) -> None:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("test tokenizer must define eos_token_id")
    for script in scripts.values():
        script.append([int(eos_id)])


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


def _token_steps(tokenizer: Any, text: str) -> list[list[int]]:
    token_ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    if not token_ids:
        raise ValueError(f"text tokenized to an empty list: {text!r}")
    return [[token_id] for token_id in token_ids]


def _decode_token_prefix(tokenizer: Any, steps: list[list[int]], count: int) -> str:
    return tokenizer.decode(_flatten_steps(steps)[:count], skip_special_tokens=True)


def _decode_until_stop(tokenizer: Any, steps: list[list[int]], stop: str) -> str:
    decoded = tokenizer.decode(_flatten_steps(steps), skip_special_tokens=True)
    if stop not in decoded:
        raise ValueError(f"stop string {stop!r} not present in decoded text {decoded!r}")
    return decoded.split(stop, maxsplit=1)[0]


def _first_stop_step(tokenizer: Any, steps: list[list[int]], stop: str) -> int:
    token_ids: list[int] = []
    for index, step in enumerate(steps, start=1):
        token_ids.extend(step)
        if stop in tokenizer.decode(token_ids, skip_special_tokens=True):
            return index
    raise ValueError(f"stop string {stop!r} not produced by scripted tokens")


def _flatten_steps(steps: list[list[int]]) -> list[int]:
    return [token_id for step in steps for token_id in step]

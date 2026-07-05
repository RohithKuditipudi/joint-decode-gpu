from __future__ import annotations

import pytest

from joint_decode_gpu.config import DEFAULT_MAX_MICROBATCH_SIZE, JointDecodeSamplingConfig


def _sampling_config(**overrides: object) -> JointDecodeSamplingConfig:
    values = {
        "max_tokens_a": 8,
        "max_tokens_b": 8,
        "top_k_a": 4,
        "top_k_b": 4,
        "max_num_batched_tokens": None,
        "barrier_timeout_s": 1.0,
        "seed": 0,
        "stop": (),
    }
    values.update(overrides)
    return JointDecodeSamplingConfig(**values)


def test_default_max_microbatch_size_is_1024() -> None:
    assert DEFAULT_MAX_MICROBATCH_SIZE == 1024
    assert _sampling_config().max_microbatch_size == 1024


def test_max_num_batched_tokens_defaults_to_none() -> None:
    config = JointDecodeSamplingConfig(
        max_tokens_a=8,
        max_tokens_b=8,
        top_k_a=4,
        top_k_b=4,
        barrier_timeout_s=1.0,
        seed=0,
        stop=(),
    )

    assert config.max_num_batched_tokens is None


def test_rejects_invalid_max_microbatch_size() -> None:
    with pytest.raises(ValueError, match="max_microbatch_size must be >= 1"):
        _sampling_config(max_microbatch_size=0)

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_MICROBATCH_SIZE = 1024
# When max_num_batched_tokens is omitted, the derived per-step token budget
# admits up to this many maximum-length prompts per decision round while the
# live window ramps up.
ADMISSION_RAMP_PROMPTS = 8


@dataclass(frozen=True)
class JointDecodeModelConfig:
    model_path: str
    gpu_index: int
    max_model_len: int
    gpu_memory_utilization: float | None
    enable_prefix_caching: bool
    enforce_eager: bool


@dataclass(frozen=True)
class JointDecodeSamplingConfig:
    max_tokens_a: int
    max_tokens_b: int
    top_k_a: int
    top_k_b: int
    barrier_timeout_s: float
    seed: int
    stop: tuple[str, ...]
    max_microbatch_size: int = DEFAULT_MAX_MICROBATCH_SIZE
    max_num_batched_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens_a < 1:
            raise ValueError("max_tokens_a must be >= 1")
        if self.max_tokens_b < 1:
            raise ValueError("max_tokens_b must be >= 1")
        if self.top_k_a < 1 or self.top_k_b < 1:
            raise ValueError("top_k_a and top_k_b must both be >= 1")
        if self.max_microbatch_size < 1:
            raise ValueError("max_microbatch_size must be >= 1")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if self.barrier_timeout_s <= 0:
            raise ValueError("barrier_timeout_s must be > 0")


@dataclass(frozen=True)
class JointDecodeConfig:
    model_a: JointDecodeModelConfig
    model_b: JointDecodeModelConfig
    sampling: JointDecodeSamplingConfig


@dataclass(frozen=True)
class GenerateOutput:
    text: str
    finish_reason: str


VLLM_GPU_ENV_VARS: dict[str, str] = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_LOGGING_STREAM": "ext://sys.stderr",
}

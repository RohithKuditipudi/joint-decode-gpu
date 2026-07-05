from __future__ import annotations

import argparse
import functools
import json
import logging

import fsspec

from joint_decode_gpu.aggregation import select_avg_logits
from joint_decode_gpu.config import (
    DEFAULT_MAX_MICROBATCH_SIZE,
    JointDecodeConfig,
    JointDecodeModelConfig,
    JointDecodeSamplingConfig,
)
from joint_decode_gpu.coordinator import run_joint_decode

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not 0.0 <= args.advisor_weight <= 1.0:
        raise ValueError("advisor_weight must be in [0, 1]")
    if args.temperature < 0.0:
        raise ValueError("temperature must be >= 0")

    rows = _read_prompt_rows(args.prompts)
    prompt_ids = [str(row["id"]) for row in rows]
    prompts_a = [str(row["prompt_a"]) for row in rows]
    prompts_b = [str(row["prompt_b"]) for row in rows]

    config = JointDecodeConfig(
        model_a=JointDecodeModelConfig(
            model_path=args.model_a,
            gpu_index=args.gpu_a,
            max_model_len=args.max_model_len_a,
            gpu_memory_utilization=args.gpu_memory_utilization_a,
            enable_prefix_caching=args.enable_prefix_caching_a,
            enforce_eager=not args.compile_a,
        ),
        model_b=JointDecodeModelConfig(
            model_path=args.model_b,
            gpu_index=args.gpu_b,
            max_model_len=args.max_model_len_b,
            gpu_memory_utilization=args.gpu_memory_utilization_b,
            enable_prefix_caching=args.enable_prefix_caching_b,
            enforce_eager=not args.compile_b,
        ),
        sampling=JointDecodeSamplingConfig(
            max_tokens_a=args.max_tokens_a if args.max_tokens_a is not None else args.max_tokens,
            max_tokens_b=args.max_tokens_b if args.max_tokens_b is not None else args.max_tokens,
            top_k_a=args.top_k_a,
            top_k_b=args.top_k_b,
            barrier_timeout_s=args.barrier_timeout_s,
            seed=args.seed,
            stop=tuple(args.stop or ()),
            max_microbatch_size=args.max_microbatch_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
        ),
    )
    select_token = functools.partial(
        select_avg_logits,
        advisor_weight=args.advisor_weight,
        temperature=args.temperature,
    )
    outputs = run_joint_decode(config, prompts_a, prompts_b, select_token=select_token)
    _write_outputs(args.output, prompt_ids, prompts_a, prompts_b, outputs)
    logger.info("wrote %d completions to %s", len(outputs), args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-a", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-b", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu-a", type=int, default=0)
    parser.add_argument("--gpu-b", type=int, default=1)
    parser.add_argument("--prompts", type=str, default="examples/prompts.jsonl")
    parser.add_argument("--top-k-a", type=int, default=32)
    parser.add_argument("--top-k-b", type=int, default=32)
    parser.add_argument("--advisor-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-tokens-a", type=int, default=None)
    parser.add_argument("--max-tokens-b", type=int, default=None)
    parser.add_argument("--max-model-len-a", type=int, default=2048)
    parser.add_argument("--max-model-len-b", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization-a", type=float, default=0.9)
    parser.add_argument("--gpu-memory-utilization-b", type=float, default=0.9)
    parser.add_argument("--enable-prefix-caching-a", action="store_true")
    parser.add_argument("--enable-prefix-caching-b", action="store_true")
    parser.add_argument("--compile-a", action="store_true")
    parser.add_argument("--compile-b", action="store_true")
    parser.add_argument("--max-microbatch-size", type=int, default=DEFAULT_MAX_MICROBATCH_SIZE)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--barrier-timeout-s", type=float, default=60.0)
    parser.add_argument("--stop", action="append", default=[])
    return parser.parse_args()


def _read_prompt_rows(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with fsspec.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "id" not in row or "prompt_a" not in row or "prompt_b" not in row:
                    raise ValueError(f"prompt row must contain id, prompt_a, prompt_b: {row!r}")
                rows.append(row)
    return rows


def _write_outputs(
    path: str,
    prompt_ids: list[str],
    prompts_a: list[str],
    prompts_b: list[str],
    outputs: list[object],
) -> None:
    with fsspec.open(path, "wt") as f:
        for prompt_id, prompt_a, prompt_b, output in zip(prompt_ids, prompts_a, prompts_b, outputs, strict=True):
            f.write(
                json.dumps(
                    {
                        "id": prompt_id,
                        "prompt_a": prompt_a,
                        "prompt_b": prompt_b,
                        "completion": output.text,
                        "finish_reason": output.finish_reason,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

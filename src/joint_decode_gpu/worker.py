from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import torch.distributed as dist

from joint_decode_gpu.config import VLLM_GPU_ENV_VARS
from joint_decode_gpu.ipc import emit_ipc

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()

    try:
        run_worker(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s [worker pid=%(process)d] %(message)s",
    )
    for key, value in VLLM_GPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    from vllm import LLM, SamplingParams

    from joint_decode_gpu.logits_processor import JointDecodeLogitsProcessor

    kwargs: dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "seed": args.seed,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "logits_processors": [JointDecodeLogitsProcessor],
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    emit_ipc({"kind": "handshake", "vocab_size": len(tokenizer), "eos_token_id": eos_id})

    engine = llm.llm_engine
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        message = json.loads(line)
        command = message.get("command")
        if command == "shutdown":
            break
        if command != "process_chunk":
            raise RuntimeError(f"unknown joint-decode worker command: {command!r}")

        request_ids: list[str] = message["request_ids"]
        prompts: list[str] = message["prompts"]
        for rid, prompt in zip(request_ids, prompts, strict=True):
            sampling_params = SamplingParams(
                max_tokens=args.max_tokens,
                ignore_eos=False,
                stop_token_ids=[eos_id] if eos_id is not None else None,
                stop=json.loads(args.stop) if args.stop else None,
                extra_args={"joint_decode_rid": rid},
            )
            engine.add_request(request_id=rid, prompt=prompt, params=sampling_params)

        live = set(request_ids)
        text_results: dict[str, str] = {}
        finish_reasons: dict[str, str] = {}
        while live:
            for output in engine.step():
                if not output.finished:
                    continue
                rid = output.request_id
                if rid not in live:
                    continue
                completion = output.outputs[0]
                text_results[rid] = completion.text
                finish_reasons[rid] = completion.finish_reason or "unknown"
                live.remove(rid)

        emit_ipc(
            {
                "kind": "result",
                "results": text_results,
                "finish_reasons": finish_reasons,
            }
        )


if __name__ == "__main__":
    main()

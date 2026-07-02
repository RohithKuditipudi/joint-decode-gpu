from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any

import torch.distributed as dist

from joint_decode_gpu.config import VLLM_GPU_ENV_VARS
from joint_decode_gpu.ipc import emit_ipc
from joint_decode_gpu.runtime_state import WorkerCommands, runtime_state

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
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
        os.environ[key] = value

    from vllm import LLM, SamplingParams

    from joint_decode_gpu.logits_processor import JointDecodeLogitsProcessor

    side = os.environ["RERANK_TOKEN_DECISION_SIDE"]
    decision_url = os.environ["RERANK_TOKEN_DECISION_URL"]
    kwargs: dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "seed": args.seed,
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": False,
        "logits_processors": [JointDecodeLogitsProcessor],
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    emit_ipc({"kind": "handshake", "vocab_size": len(tokenizer), "eos_token_id": eos_id})

    engine = llm.llm_engine
    _validate_engine(engine, args.max_num_seqs, args.max_num_batched_tokens)
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

        runtime_state.reset()
        request_ids: list[str] = message["request_ids"]
        prompts: list[str] = message["prompts"]
        initial_admit: list[str] = message.get("initial_admit") or request_ids
        prompts_by_rid = dict(zip(request_ids, prompts, strict=True))
        _validate_prompt_lengths(tokenizer, prompts, args.max_model_len)

        live: set[str] = set()
        pending_admits: list[str] = []
        text_results: dict[str, str] = {}
        finish_reasons: dict[str, str] = {}
        timeout = float(os.environ["RERANK_TOKEN_DECISION_TIMEOUT"])
        _admit_requests(engine, tokenizer, prompts_by_rid, initial_admit, live, eos_id, args.max_model_len)

        run_done = False
        while not run_done:
            _stash_admits(_drain_worker_commands(), pending_admits)
            if pending_admits and _local_decision_boundary(live):
                admits = pending_admits
                pending_admits = []
                _admit_requests(engine, tokenizer, prompts_by_rid, admits, live, eos_id, args.max_model_len)

            if not live:
                response = _post_decision(
                    decision_url,
                    {
                        "kind": "control",
                        "side": side,
                    },
                    timeout=timeout,
                )
                if response.get("abort"):
                    raise RuntimeError(str(response["abort"]))
                if response.get("done"):
                    run_done = True
                    continue
                pending_admits.extend(str(rid) for rid in response.get("admit") or [])
                continue

            _set_held_request_ids(engine, live)
            finished: list[dict[str, Any]] = []
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
                finished.append(
                    {
                        "rid": rid,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": getattr(completion, "stop_reason", None),
                    }
                )
            if finished:
                _post_decision(
                    decision_url,
                    {
                        "kind": "finish",
                        "side": side,
                        "finished": finished,
                    },
                    timeout=timeout,
                )
            _stash_admits(_drain_worker_commands(), pending_admits)

        emit_ipc(
            {
                "kind": "result",
                "results": text_results,
                "finish_reasons": finish_reasons,
            }
        )


def _drain_worker_commands() -> WorkerCommands:
    commands = runtime_state.drain_commands()
    if commands.abort is not None:
        raise RuntimeError(commands.abort)
    return commands


def _stash_admits(commands: WorkerCommands, pending_admits: list[str]) -> None:
    pending_admits.extend(commands.admit)


def _local_decision_boundary(live: set[str]) -> bool:
    return not any(runtime_state.pending_tokens.get(rid) for rid in live)


def _admit_requests(
    engine: Any,
    tokenizer: Any,
    prompts_by_rid: dict[str, str],
    request_ids: list[str],
    live: set[str],
    eos_id: int | None,
    max_model_len: int,
) -> None:
    for rid in request_ids:
        if rid in live:
            continue
        prompt = prompts_by_rid[rid]
        sampling_params = SamplingParams(
            max_tokens=_local_max_tokens(tokenizer, prompt, max_model_len),
            ignore_eos=False,
            stop_token_ids=[eos_id] if eos_id is not None else None,
            extra_args={"joint_decode_rid": rid},
        )
        engine.add_request(request_id=rid, prompt=prompt, params=sampling_params)
        live.add(rid)


def _set_held_request_ids(engine: Any, live: set[str]) -> None:
    scheduler = _scheduler(engine)
    pending_tokens = runtime_state.pending_tokens
    decode_live: dict[str, str] = {}
    for internal_id, request in scheduler.requests.items():
        assert request.sampling_params is not None
        extra_args = request.sampling_params.extra_args
        assert extra_args is not None
        rid = str(extra_args["joint_decode_rid"])

        if rid not in live:
            continue
        if request.num_computed_tokens >= request.num_prompt_tokens:
            decode_live[rid] = internal_id

    busy = any(pending_tokens.get(rid) for rid in decode_live)
    scheduler.held_request_ids = {
        internal_id
        for rid, internal_id in decode_live.items()
        if busy and not pending_tokens.get(rid)
    }


def _scheduler(engine: Any) -> Any:
    engine_core = getattr(engine, "engine_core", None)
    inner_core = getattr(engine_core, "engine_core", None)
    scheduler = getattr(inner_core, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("vLLM engine does not expose engine.engine_core.engine_core.scheduler")
    return scheduler


def _validate_engine(engine: Any, max_num_seqs: int, max_num_batched_tokens: int) -> None:
    scheduler = _scheduler(engine)
    if not hasattr(scheduler, "held_request_ids"):
        raise RuntimeError("vLLM scheduler does not expose held_request_ids")
    scheduler_config = scheduler.scheduler_config
    if scheduler_config.async_scheduling is not False:
        raise RuntimeError("joint decode requires async_scheduling=False")
    if scheduler_config.max_num_seqs != max_num_seqs:
        raise RuntimeError(
            f"vLLM max_num_seqs mismatch: scheduler={scheduler_config.max_num_seqs} expected={max_num_seqs}"
        )
    scheduled_tokens = getattr(scheduler, "max_num_scheduled_tokens", max_num_batched_tokens)
    if scheduled_tokens < max_num_batched_tokens:
        raise RuntimeError(
            f"vLLM max_num_scheduled_tokens={scheduled_tokens} is below max_num_batched_tokens={max_num_batched_tokens}"
        )


def _validate_prompt_lengths(tokenizer: Any, prompts: list[str], max_model_len: int) -> None:
    for prompt in prompts:
        prompt_len = _prompt_len(tokenizer, prompt)
        if prompt_len + 1 > max_model_len:
            raise ValueError(
                f"prompt length {prompt_len} leaves no room for a generated token under max_model_len={max_model_len}"
            )


def _local_max_tokens(tokenizer: Any, prompt: str, max_model_len: int) -> int:
    return max_model_len - _prompt_len(tokenizer, prompt)


def _prompt_len(tokenizer: Any, prompt: str) -> int:
    tokenized = tokenizer(prompt)
    prompt_tokens = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    return len(prompt_tokens)


def _post_decision(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"joint-decode worker request failed: {exc}") from exc


if __name__ == "__main__":
    main()

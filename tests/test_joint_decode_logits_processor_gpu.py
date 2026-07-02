from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import torch.distributed as dist

from joint_decode_gpu.config import VLLM_GPU_ENV_VARS


@pytest.mark.gpu
def test_forced_tokens_stay_attached_to_request_ids_under_real_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = os.environ.get("JOINT_DECODE_GPU_TEST_MODEL")
    if not model_path:
        pytest.skip("set JOINT_DECODE_GPU_TEST_MODEL to a local tiny vLLM-compatible model")

    for key, value in VLLM_GPU_ENV_VARS.items():
        monkeypatch.setenv(key, value)

    scripts: dict[str, list[int]] = {}
    calls: list[dict[str, Any]] = []
    httpd = _decision_server(scripts, calls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    monkeypatch.setenv("RERANK_TOKEN_DECISION_URL", f"http://127.0.0.1:{port}/tokens")
    monkeypatch.setenv("RERANK_TOKEN_DECISION_SIDE", "a")
    monkeypatch.setenv("RERANK_TOKEN_DECISION_TOP_K", "4")
    monkeypatch.setenv("RERANK_TOKEN_DECISION_TIMEOUT", "30")

    try:
        from vllm import LLM, SamplingParams

        from joint_decode_gpu.logits_processor import JointDecodeLogitsProcessor

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=128,
            max_num_seqs=4,
            logits_processors=[JointDecodeLogitsProcessor],
            enforce_eager=True,
        )
        tokenizer = llm.get_tokenizer()
        eos_id = int(tokenizer.eos_token_id)
        scripts.update(_scripts(vocab_size=len(tokenizer), eos_id=eos_id, num_requests=8))

        engine = llm.llm_engine
        for rid, script in scripts.items():
            engine.add_request(
                request_id=rid,
                prompt="Hello",
                params=SamplingParams(
                    max_tokens=len(script),
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=False,
                    stop_token_ids=[eos_id],
                    extra_args={"joint_decode_rid": rid},
                ),
            )

        live = set(scripts)
        results: dict[str, Any] = {}
        while live:
            for output in engine.step():
                if not output.finished or output.request_id not in live:
                    continue
                results[output.request_id] = output.outputs[0]
                live.remove(output.request_id)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

        if dist.is_initialized():
            dist.destroy_process_group()

    for rid, script in scripts.items():
        assert list(results[rid].token_ids) == script
        assert results[rid].finish_reason == "stop"

    request_sets = [set(call["request_ids"]) for call in calls]
    assert any(len(request_set) == 4 for request_set in request_sets)
    assert any(request_set != request_sets[0] for request_set in request_sets[1:])


def _scripts(vocab_size: int, eos_id: int, num_requests: int) -> dict[str, list[int]]:
    special = {eos_id}
    scripts: dict[str, list[int]] = {}
    next_token = 0
    for index in range(num_requests):
        rid = f"r{index}"
        length = 2 + index
        tokens: list[int] = []
        while len(tokens) < length:
            if next_token >= vocab_size:
                raise ValueError("test tokenizer does not have enough non-EOS token ids")
            token = next_token
            next_token += 1
            if token in special:
                continue
            tokens.append(token)
        scripts[rid] = tokens + [eos_id]
    return scripts


def _decision_server(
    scripts: dict[str, list[int]],
    calls: list[dict[str, Any]],
) -> ThreadingHTTPServer:
    positions: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length))
            calls.append(payload)
            tokens: dict[str, list[int]] = {}
            for rid in payload["request_ids"]:
                position = positions.get(rid, 0)
                tokens[rid] = [scripts[rid][position]]
                positions[rid] = position + 1
            response = json.dumps({"tokens": tokens, "force_stop": [], "admit": [], "abort": None}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)

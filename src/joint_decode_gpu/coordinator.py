from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from joint_decode_gpu.config import VLLM_GPU_ENV_VARS, GenerateOutput, JointDecodeConfig, JointDecodeModelConfig
from joint_decode_gpu.ipc import read_ipc

logger = logging.getLogger(__name__)

SelectToken = Callable[..., int]


class Coordinator:
    def __init__(self, timeout_s: float, select_token: SelectToken, rng: random.Random) -> None:
        self._timeout_s = timeout_s
        self._select_token = select_token
        self._rng = rng
        self._lock = threading.Lock()
        self._barriers: dict[bytes, dict[str, Any]] = {}

    def handle(self, side: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_ids = list(payload["request_ids"])
        step_indices = payload["step_indices"]
        topk = payload.get("topk") or {}
        key = json.dumps(sorted((rid, step_indices[rid]) for rid in request_ids)).encode()

        with self._lock:
            entry = self._barriers.get(key)
            if entry is None:
                entry = {
                    "a": None,
                    "b": None,
                    "ready": threading.Event(),
                    "result": None,
                    "request_ids": request_ids,
                }
                self._barriers[key] = entry
            entry[side] = topk

            if entry["a"] is not None and entry["b"] is not None:
                tokens: dict[str, int] = {}
                for rid in entry["request_ids"]:
                    tokens[rid] = self._select_token(
                        entry["a"].get(rid, []),
                        entry["b"].get(rid, []),
                        rng=self._rng,
                    )
                entry["result"] = {"tokens": tokens}
                entry["ready"].set()

        if not entry["ready"].wait(timeout=self._timeout_s):
            raise TimeoutError(f"joint-decode barrier timed out for request_ids={request_ids}")

        with self._lock:
            self._barriers.pop(key, None)

        assert entry["result"] is not None
        return entry["result"]


class DecisionHandler(BaseHTTPRequestHandler):
    coordinator: Coordinator | None = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length))
            side = self.path.lstrip("/")
            if side not in ("a", "b"):
                self.send_error(404, f"unknown path {self.path!r}")
                return
            assert self.coordinator is not None
            response = self.coordinator.handle(side, payload)
            response_bytes = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as exc:
            logger.exception("error handling token-decision POST")
            self.send_error(500, str(exc))


class JointDecoder:
    def __init__(self, config: JointDecodeConfig, *, select_token: SelectToken) -> None:
        self.config = config
        self._select_token = select_token
        self._rng = random.Random(config.sampling.seed)
        self._coordinator: Coordinator | None = None
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._proc_a: subprocess.Popen | None = None
        self._proc_b: subprocess.Popen | None = None
        self._chunk_seq = 0

    def __enter__(self) -> JointDecoder:
        self._coordinator = Coordinator(
            self.config.sampling.barrier_timeout_s,
            self._select_token,
            self._rng,
        )
        DecisionHandler.coordinator = self._coordinator
        self._http_server = ThreadingHTTPServer(("127.0.0.1", 0), DecisionHandler)
        actual_port = self._http_server.server_address[1]
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()
        try:
            self._proc_a = self._spawn_worker(
                side="a",
                model_config=self.config.model_a,
                top_k=self.config.sampling.top_k_a,
                decision_url=f"http://127.0.0.1:{actual_port}/a",
            )
            self._proc_b = self._spawn_worker(
                side="b",
                model_config=self.config.model_b,
                top_k=self.config.sampling.top_k_b,
                decision_url=f"http://127.0.0.1:{actual_port}/b",
            )
            handshake_a = read_ipc(self._proc_a, expect_kind="handshake")
            handshake_b = read_ipc(self._proc_b, expect_kind="handshake")
            self._validate_handshake(handshake_a, handshake_b)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _spawn_worker(
        self,
        *,
        side: str,
        model_config: JointDecodeModelConfig,
        top_k: int,
        decision_url: str,
    ) -> subprocess.Popen:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(model_config.gpu_index)
        env["RERANK_TOKEN_DECISION_URL"] = decision_url
        env["RERANK_TOKEN_DECISION_TOP_K"] = str(top_k)
        env["RERANK_TOKEN_DECISION_TIMEOUT"] = str(self.config.sampling.barrier_timeout_s + 10.0)
        for key, value in VLLM_GPU_ENV_VARS.items():
            env[key] = value

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "joint_decode_gpu.worker",
            "--model-path",
            model_config.model_path,
            "--max-tokens",
            str(self.config.sampling.max_tokens),
            "--max-model-len",
            str(model_config.max_model_len),
            "--seed",
            str(self.config.sampling.seed),
        ]
        if model_config.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(model_config.gpu_memory_utilization)]
        if model_config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if model_config.enforce_eager:
            cmd.append("--enforce-eager")
        if self.config.sampling.stop:
            cmd += ["--stop", json.dumps(list(self.config.sampling.stop))]

        logger.info("spawning joint-decode worker %s on CUDA_VISIBLE_DEVICES=%s", side, model_config.gpu_index)
        return subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    @staticmethod
    def _validate_handshake(handshake_a: dict[str, Any], handshake_b: dict[str, Any]) -> None:
        if handshake_a["vocab_size"] != handshake_b["vocab_size"]:
            raise RuntimeError(
                f"tokenizer vocab size mismatch: A={handshake_a['vocab_size']} B={handshake_b['vocab_size']}"
            )
        if handshake_a["eos_token_id"] != handshake_b["eos_token_id"]:
            raise RuntimeError(
                f"EOS token id mismatch: A={handshake_a['eos_token_id']} B={handshake_b['eos_token_id']}"
            )

    def generate(self, prompts_a: list[str], prompts_b: list[str]) -> list[GenerateOutput]:
        outputs: list[GenerateOutput] = []
        microbatch_size = self.config.sampling.microbatch_size
        for start in range(0, len(prompts_a), microbatch_size):
            outputs.extend(
                self._generate_microbatch(
                    prompts_a[start : start + microbatch_size],
                    prompts_b[start : start + microbatch_size],
                )
            )
        return outputs

    def _generate_microbatch(self, prompts_a: list[str], prompts_b: list[str]) -> list[GenerateOutput]:
        request_ids = [f"jd-c{self._chunk_seq}-r{i:06d}" for i in range(len(prompts_a))]
        self._chunk_seq += 1
        for proc, prompts in ((self._proc_a, prompts_a), (self._proc_b, prompts_b)):
            assert proc is not None and proc.stdin is not None
            command = {
                "command": "process_chunk",
                "request_ids": request_ids,
                "prompts": prompts,
            }
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()

        results: dict[str, Any] = {}

        def reader(name: str, proc: subprocess.Popen) -> None:
            try:
                results[name] = read_ipc(proc, expect_kind="result")
            except Exception as exc:
                results[name] = exc

        assert self._proc_a is not None and self._proc_b is not None
        threads = [
            threading.Thread(target=reader, args=("a", self._proc_a), daemon=True),
            threading.Thread(target=reader, args=("b", self._proc_b), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for name in ("a", "b"):
            if isinstance(results.get(name), Exception):
                raise results[name]

        result_a = results["a"]
        text_results: dict[str, str] = result_a["results"]
        finish_reasons: dict[str, str] = result_a["finish_reasons"]
        return [
            GenerateOutput(
                text=text_results[rid],
                finish_reason=finish_reasons[rid],
            )
            for rid in request_ids
        ]

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for name, proc in (("a", self._proc_a), ("b", self._proc_b)):
            if proc is None:
                continue
            try:
                if proc.poll() is None and proc.stdin is not None and not proc.stdin.closed:
                    try:
                        proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                        proc.stdin.flush()
                    except (BrokenPipeError, ValueError):
                        pass
                    try:
                        proc.stdin.close()
                    except Exception:
                        logger.exception("error closing stdin for worker %s", name)
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception:
                logger.exception("error shutting down worker %s", name)
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
            self._http_thread = None


def run_joint_decode(
    config: JointDecodeConfig,
    prompts_a: list[str],
    prompts_b: list[str],
    *,
    select_token: SelectToken,
) -> list[GenerateOutput]:
    with JointDecoder(config, select_token=select_token) as decoder:
        return decoder.generate(prompts_a, prompts_b)

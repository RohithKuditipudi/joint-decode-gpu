from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from joint_decode_gpu.config import (
    ADMISSION_RAMP_PROMPTS,
    VLLM_GPU_ENV_VARS,
    GenerateOutput,
    JointDecodeConfig,
    JointDecodeModelConfig,
)
from joint_decode_gpu.ipc import read_ipc

logger = logging.getLogger(__name__)

SelectTokens = Callable[..., int | tuple[list[int], list[int]]]


class Side(StrEnum):
    A = "a"
    B = "b"

    def peer(self) -> "Side":
        return Side.B if self is Side.A else Side.A


@dataclass
class RequestState:
    rid: str
    index: int
    done: set[Side] = field(default_factory=set)


@dataclass(frozen=True)
class SidePlan:
    """Per-side admission inputs for one run: the worker's per-step token budget
    and each request's prompt length in that side's tokens."""

    budget: int
    prompt_tokens: dict[str, int]


@dataclass
class PendingEntry:
    side: Side
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: Exception | None = None

    def resolve(self, response: dict[str, Any]) -> None:
        self.response = response
        self.event.set()

    def fail(self, error: Exception) -> None:
        self.error = error
        self.event.set()


@dataclass
class DecodeEntry(PendingEntry):
    request_ids: list[str] = field(default_factory=list)
    topk: dict[str, list[dict[str, int | float]]] = field(default_factory=dict)


@dataclass
class ControlEntry(PendingEntry):
    pass


def is_retired(state: RequestState) -> bool:
    return state.done == {Side.A, Side.B}


def needs_force_stop(state: RequestState, side: Side) -> bool:
    return side.peer() in state.done and side not in state.done


class Coordinator:
    def __init__(
        self,
        timeout_s: float,
        select_tokens: SelectTokens,
        rng: random.Random,
    ) -> None:
        self._timeout_s = timeout_s
        self._select_tokens = select_tokens
        self._rng = rng
        self._lock = threading.Lock()
        self._pending_decode: dict[Side, DecodeEntry] = {}
        self._pending_control: dict[Side, ControlEntry] = {}
        self._requests: dict[str, RequestState] = {}
        self._queue: deque[str] = deque()
        self._live: set[str] = set()
        self._max_concurrent = 0
        self._plan_a = SidePlan(budget=0, prompt_tokens={})
        self._plan_b = SidePlan(budget=0, prompt_tokens={})
        self._abort: str | None = None

    def begin_run(
        self,
        request_ids: list[str],
        max_concurrent: int,
        plan_a: SidePlan,
        plan_b: SidePlan,
    ) -> list[str]:
        with self._lock:
            self._fail_pending(RuntimeError("starting new run while requests are pending"))
            self._pending_decode.clear()
            self._pending_control.clear()
            self._requests = {
                rid: RequestState(rid=rid, index=index)
                for index, rid in enumerate(request_ids)
            }
            self._queue = deque(request_ids)
            self._live.clear()
            self._max_concurrent = max_concurrent
            self._plan_a = plan_a
            self._plan_b = plan_b
            self._abort = None
            return self._admit_available()

    def handle(self, side: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_side = Side(side)
        payload_side = payload.get("side")
        if payload_side is not None and payload_side != request_side.value:
            raise ValueError(f"payload side {payload_side!r} does not match path side {request_side.value!r}")
        kind = payload.get("kind", "decode")
        if kind == "decode":
            return self.handle_decode(request_side, payload)
        if kind == "finish":
            return self.handle_finish(request_side, payload)
        if kind == "control":
            return self.handle_control(request_side, payload)
        raise ValueError(f"unknown joint-decode request kind: {kind!r}")

    def handle_decode(self, side: Side, payload: dict[str, Any]) -> dict[str, Any]:
        entry = DecodeEntry(
            side=side,
            request_ids=list(payload["request_ids"]),
            topk=payload.get("topk") or {},
        )
        with self._lock:
            self._ensure_live(entry.request_ids)
            self._store_pending(self._pending_decode, entry)
            self._try_resolve_or_abort()
        return self._wait(entry)

    def handle_control(self, side: Side, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        entry = ControlEntry(side=side)
        with self._lock:
            self._store_pending(self._pending_control, entry)
            self._try_resolve_or_abort()
        return self._wait(entry)

    def handle_finish(self, side: Side, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for item in payload.get("finished", []):
                rid = str(item["rid"])
                state = self._requests[rid]
                state.done.add(side)
                if is_retired(state):
                    self._live.discard(rid)
            self._try_resolve_or_abort()
        return {"ok": True}

    def _store_pending(self, pending: dict[Side, PendingEntry], entry: PendingEntry) -> None:
        if entry.side in pending:
            raise RuntimeError(f"side {entry.side.value} already has a pending request")
        pending[entry.side] = entry

    def _wait(self, entry: PendingEntry) -> dict[str, Any]:
        if not entry.event.wait(timeout=self._timeout_s):
            with self._lock:
                self._pending_decode.pop(entry.side, None)
                self._pending_control.pop(entry.side, None)
            raise TimeoutError(f"joint-decode coordinator timed out waiting for side={entry.side.value}")
        if entry.error is not None:
            raise entry.error
        assert entry.response is not None
        return entry.response

    def _try_resolve(self) -> None:
        if self._abort is not None:
            self._fail_pending(RuntimeError(self._abort))
            return
        decode_a = self._pending_decode.get(Side.A)
        decode_b = self._pending_decode.get(Side.B)
        control_a = self._pending_control.get(Side.A)
        control_b = self._pending_control.get(Side.B)
        if decode_a is not None and decode_b is not None:
            self._resolve_decode_pair(decode_a, decode_b)
            return
        if control_a is not None and control_b is not None:
            self._resolve_control_pair(control_a, control_b)
            return
        if control_a is not None and decode_b is not None:
            self._resolve_control_decode(control_a, decode_b)
            return
        if decode_a is not None and control_b is not None:
            self._resolve_control_decode(control_b, decode_a)
            return

        for side, entry in tuple(self._pending_decode.items()):
            if self._can_force_stop_all(entry.request_ids, side):
                self._pending_decode.pop(side, None)
                entry.resolve(self._decode_response(force_stop=entry.request_ids))
                return

        if self._is_done():
            for side, entry in tuple(self._pending_control.items()):
                self._pending_control.pop(side, None)
                entry.resolve(self._control_response(done=True))

    def _try_resolve_or_abort(self) -> None:
        try:
            self._try_resolve()
        except Exception as exc:
            self._abort_all(exc)
            raise

    def _resolve_decode_pair(self, entry_a: DecodeEntry, entry_b: DecodeEntry) -> None:
        sa = set(entry_a.request_ids)
        sb = set(entry_b.request_ids)
        shared = sa & sb
        force_a = [
            rid
            for rid in entry_a.request_ids
            if rid not in sb and needs_force_stop(self._requests[rid], Side.A)
        ]
        force_b = [
            rid
            for rid in entry_b.request_ids
            if rid not in sa and needs_force_stop(self._requests[rid], Side.B)
        ]
        invalid_a = sa - sb - set(force_a)
        invalid_b = sb - sa - set(force_b)
        if invalid_a or invalid_b:
            self._abort_all(RuntimeError(f"joint-decode desync: only_a={sorted(invalid_a)} only_b={sorted(invalid_b)}"))
            return

        tokens_a: dict[str, list[int]] = {}
        tokens_b: dict[str, list[int]] = {}
        for rid in entry_a.request_ids:
            if rid not in shared:
                continue
            state = self._requests[rid]
            if state.done:
                self._abort_all(RuntimeError(f"joint-decode desync: decoded finished request {rid}"))
                return
            selected_a, selected_b = self._select_for_rid(rid, entry_a, entry_b)
            tokens_a[rid] = selected_a
            tokens_b[rid] = selected_b

        self._pending_decode.pop(Side.A, None)
        self._pending_decode.pop(Side.B, None)
        admit = []
        if self._can_admit_from_decode_response(tokens_a, tokens_b, force_a, force_b):
            admit = self._admit_available()
        entry_a.resolve(self._decode_response(tokens=tokens_a, force_stop=force_a, admit=admit))
        entry_b.resolve(self._decode_response(tokens=tokens_b, force_stop=force_b, admit=admit))

    def _resolve_control_pair(self, entry_a: ControlEntry, entry_b: ControlEntry) -> None:
        if self._is_done():
            self._pending_control.pop(Side.A, None)
            self._pending_control.pop(Side.B, None)
            entry_a.resolve(self._control_response(done=True))
            entry_b.resolve(self._control_response(done=True))
            return

        admit = self._admit_available()
        if admit:
            self._pending_control.pop(Side.A, None)
            self._pending_control.pop(Side.B, None)
            entry_a.resolve(self._control_response(admit=admit))
            entry_b.resolve(self._control_response(admit=admit))
            return

        self._abort_all(RuntimeError("joint-decode desync: both workers idle before run completion"))

    def _resolve_control_decode(self, control_entry: ControlEntry, decode_entry: DecodeEntry) -> None:
        if not self._can_force_stop_all(decode_entry.request_ids, decode_entry.side):
            self._abort_all(
                RuntimeError(
                    "joint-decode desync: one worker is idle while peer still needs logits "
                    f"for rids={decode_entry.request_ids}"
                )
            )
            return

        self._pending_decode.pop(decode_entry.side, None)
        decode_entry.resolve(self._decode_response(force_stop=decode_entry.request_ids))
        if self._is_done():
            self._pending_control.pop(control_entry.side, None)
            control_entry.resolve(self._control_response(done=True))

    def _select_for_rid(
        self,
        rid: str,
        entry_a: DecodeEntry,
        entry_b: DecodeEntry,
    ) -> tuple[list[int], list[int]]:
        selected = self._select_tokens(
            entry_a.topk.get(rid, []),
            entry_b.topk.get(rid, []),
            rng=self._rng,
        )
        if isinstance(selected, int):
            tokens_a = [selected]
            tokens_b = [selected]
        else:
            tokens_a, tokens_b = selected
        if not tokens_a or not tokens_b:
            raise ValueError(f"selector returned an empty token list for rid={rid}")
        return list(tokens_a), list(tokens_b)

    def _can_force_stop_all(self, request_ids: list[str], side: Side) -> bool:
        return bool(request_ids) and all(needs_force_stop(self._requests[rid], side) for rid in request_ids)

    def _decode_response(
        self,
        *,
        tokens: dict[str, list[int]] | None = None,
        force_stop: list[str] | None = None,
        admit: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "tokens": tokens or {},
            "force_stop": force_stop or [],
            "admit": admit or [],
            "abort": None,
        }

    def _control_response(self, *, admit: list[str] | None = None, done: bool = False) -> dict[str, Any]:
        return {
            "admit": admit or [],
            "abort": None,
            "done": done,
        }

    def _abort_all(self, error: Exception) -> None:
        self._abort = str(error)
        self._fail_pending(error)

    def _fail_pending(self, error: Exception) -> None:
        for entry in self._pending_decode.values():
            entry.fail(error)
        for entry in self._pending_control.values():
            entry.fail(error)

    def _is_done(self) -> bool:
        return not self._queue and not self._live

    def _can_admit_from_decode_response(
        self,
        tokens_a: dict[str, list[int]],
        tokens_b: dict[str, list[int]],
        force_a: list[str],
        force_b: list[str],
    ) -> bool:
        if not self._queue or len(self._live) >= self._max_concurrent:
            return False
        if any(len(tokens) != 1 for tokens in tokens_a.values()):
            return False
        if any(len(tokens) != 1 for tokens in tokens_b.values()):
            return False
        return bool(tokens_a or tokens_b or force_a or force_b)

    def _admit_available(self) -> list[str]:
        admits: list[str] = []
        tokens_a = tokens_b = 0
        while self._queue and len(self._live) + len(admits) < self._max_concurrent:
            rid = self._queue[0]
            need_a = tokens_a + self._plan_a.prompt_tokens[rid]
            need_b = tokens_b + self._plan_b.prompt_tokens[rid]
            # The next scheduler step must fit one decode token per live request
            # plus the full prompts of every request admitted this round, on
            # both sides, so the admitted prompts prefill in the same step.
            if len(self._live) + need_a > self._plan_a.budget:
                break
            if len(self._live) + need_b > self._plan_b.budget:
                break
            tokens_a, tokens_b = need_a, need_b
            admits.append(self._queue.popleft())
        self._live.update(admits)
        if admits:
            logger.info(
                "admitted %d requests (live=%d, queued=%d)",
                len(admits),
                len(self._live),
                len(self._queue),
            )
        return admits

    def _ensure_live(self, request_ids: list[str]) -> None:
        missing = [rid for rid in request_ids if rid not in self._live]
        if missing:
            raise RuntimeError(f"decode request contains non-live rids: {missing}")


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


def _validated_side_plan(
    payload: dict[str, Any],
    *,
    request_ids: list[str],
    max_model_len: int,
    budget: int,
    side: str,
) -> SidePlan:
    prompt_tokens = {str(rid): int(count) for rid, count in payload["prompt_tokens"].items()}
    if set(prompt_tokens) != set(request_ids):
        raise RuntimeError(f"worker {side} plan request ids do not match the run request ids")
    for rid, count in prompt_tokens.items():
        if count < 1:
            raise RuntimeError(f"worker {side} reported an empty prompt for rid={rid}")
        if count > max_model_len:
            raise RuntimeError(
                f"worker {side} prompt for rid={rid} is {count} tokens, over max_model_len={max_model_len}"
            )
        if count > budget:
            raise RuntimeError(
                f"worker {side} prompt for rid={rid} is {count} tokens, "
                f"over max_num_batched_tokens={budget}"
            )
    return SidePlan(budget=budget, prompt_tokens=prompt_tokens)


class JointDecoder:
    def __init__(self, config: JointDecodeConfig, *, select_token: SelectTokens) -> None:
        self.config = config
        self._select_token = select_token
        self._rng = random.Random(config.sampling.seed)
        self._coordinator: Coordinator | None = None
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._proc_a: subprocess.Popen | None = None
        self._proc_b: subprocess.Popen | None = None
        self._max_live_requests_a = 0
        self._max_live_requests_b = 0
        self._budget_a = 0
        self._budget_b = 0

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
            max_num_seqs_a, self._budget_a = self._worker_scheduler_args(self.config.model_a)
            max_num_seqs_b, self._budget_b = self._worker_scheduler_args(self.config.model_b)
            self._proc_a = self._spawn_worker(
                side="a",
                model_config=self.config.model_a,
                top_k=self.config.sampling.top_k_a,
                max_tokens=self.config.sampling.max_tokens_a,
                decision_url=f"http://127.0.0.1:{actual_port}/a",
                max_num_seqs=max_num_seqs_a,
                max_num_batched_tokens=self._budget_a,
            )
            self._proc_b = self._spawn_worker(
                side="b",
                model_config=self.config.model_b,
                top_k=self.config.sampling.top_k_b,
                max_tokens=self.config.sampling.max_tokens_b,
                decision_url=f"http://127.0.0.1:{actual_port}/b",
                max_num_seqs=max_num_seqs_b,
                max_num_batched_tokens=self._budget_b,
            )
            handshake_a = read_ipc(self._proc_a, expect_kind="handshake")
            handshake_b = read_ipc(self._proc_b, expect_kind="handshake")
            self._max_live_requests_a = int(handshake_a["max_live_requests"])
            self._max_live_requests_b = int(handshake_b["max_live_requests"])
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
        max_tokens: int,
        decision_url: str,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> subprocess.Popen:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(model_config.gpu_index)
        env["RERANK_TOKEN_DECISION_URL"] = decision_url
        env["RERANK_TOKEN_DECISION_SIDE"] = side
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
            str(max_tokens),
            "--max-model-len",
            str(model_config.max_model_len),
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
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

    def _worker_scheduler_args(self, model_config: JointDecodeModelConfig) -> tuple[int, int]:
        max_num_seqs = self.config.sampling.max_microbatch_size
        configured = self.config.sampling.max_num_batched_tokens
        if configured is None:
            return max_num_seqs, max_num_seqs + ADMISSION_RAMP_PROMPTS * model_config.max_model_len
        if configured < model_config.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens={configured} is smaller than "
                f"max_model_len={model_config.max_model_len}"
            )
        if configured < max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens={configured} is smaller than "
                f"max_num_seqs={max_num_seqs}"
            )
        return max_num_seqs, configured

    def generate(self, prompts_a: list[str], prompts_b: list[str]) -> list[GenerateOutput]:
        if len(prompts_a) != len(prompts_b):
            raise ValueError(f"prompt count mismatch: A={len(prompts_a)} B={len(prompts_b)}")
        return self._generate_run(prompts_a, prompts_b)

    def _generate_run(self, prompts_a: list[str], prompts_b: list[str]) -> list[GenerateOutput]:
        request_ids = [f"jd-r{i:06d}" for i in range(len(prompts_a))]
        assert self._coordinator is not None
        assert self._proc_a is not None and self._proc_b is not None
        for proc, prompts in ((self._proc_a, prompts_a), (self._proc_b, prompts_b)):
            assert proc.stdin is not None
            command = {
                "command": "process_chunk",
                "request_ids": request_ids,
                "prompts": prompts,
            }
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()

        plan_a = _validated_side_plan(
            read_ipc(self._proc_a, expect_kind="plan"),
            request_ids=request_ids,
            max_model_len=self.config.model_a.max_model_len,
            budget=self._budget_a,
            side="a",
        )
        plan_b = _validated_side_plan(
            read_ipc(self._proc_b, expect_kind="plan"),
            request_ids=request_ids,
            max_model_len=self.config.model_b.max_model_len,
            budget=self._budget_b,
            side="b",
        )
        window = min(
            len(prompts_a),
            self.config.sampling.max_microbatch_size,
            self._max_live_requests_a,
            self._max_live_requests_b,
        )
        initial_admit = self._coordinator.begin_run(request_ids, window, plan_a, plan_b)
        for proc in (self._proc_a, self._proc_b):
            assert proc.stdin is not None
            proc.stdin.write(json.dumps({"command": "start", "initial_admit": initial_admit}) + "\n")
            proc.stdin.flush()

        results: dict[str, Any] = {}

        def reader(name: str, proc: subprocess.Popen) -> None:
            try:
                results[name] = read_ipc(proc, expect_kind="result")
            except Exception as exc:
                results[name] = exc

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
    select_token: SelectTokens,
) -> list[GenerateOutput]:
    with JointDecoder(config, select_token=select_token) as decoder:
        return decoder.generate(prompts_a, prompts_b)

from __future__ import annotations

from dataclasses import dataclass

import pytest

from joint_decode_gpu.runtime_state import runtime_state
from joint_decode_gpu.worker import (
    _drain_worker_commands,
    _post_decision,
    _set_held_request_ids,
    _validate_prompt_lengths,
)


def test_worker_drains_runtime_abort_command() -> None:
    runtime_state.reset()
    runtime_state.publish_commands(abort="boom")

    with pytest.raises(RuntimeError, match="boom"):
        _drain_worker_commands()
    assert runtime_state.latest_commands is None


def test_compute_holds_skips_prefill_and_holds_empty_decode_rows() -> None:
    runtime_state.reset()
    runtime_state.pending_tokens["busy"] = [7]
    scheduler = _FakeScheduler(
        requests={
            "busy": _FakeRequest(num_computed_tokens=5, num_prompt_tokens=5),
            "empty": _FakeRequest(num_computed_tokens=5, num_prompt_tokens=5),
            "prefill": _FakeRequest(num_computed_tokens=2, num_prompt_tokens=5),
        }
    )
    engine = _FakeEngine(scheduler)

    _set_held_request_ids(engine, {"busy", "empty", "prefill"})

    assert scheduler.held_request_ids == {"empty"}


def test_prompt_length_validation_requires_generation_room() -> None:
    tokenizer = _FakeTokenizer({"ok": [1, 2], "too-long": [1, 2, 3]})

    _validate_prompt_lengths(tokenizer, ["ok"], max_model_len=3)
    with pytest.raises(ValueError, match="leaves no room"):
        _validate_prompt_lengths(tokenizer, ["too-long"], max_model_len=3)


def test_worker_post_uses_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        seen["request"] = request
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert _post_decision("http://127.0.0.1:1/a", {"kind": "finish"}, timeout=3.5) == {"ok": True}
    assert seen["timeout"] == 3.5


@dataclass
class _FakeRequest:
    num_computed_tokens: int
    num_prompt_tokens: int


class _FakeScheduler:
    def __init__(self, requests: dict[str, _FakeRequest]) -> None:
        self.requests = requests
        self.held_request_ids: set[str] = set()


class _FakeEngineCore:
    def __init__(self, scheduler: _FakeScheduler) -> None:
        self.scheduler = scheduler


class _FakeEngineCoreClient:
    def __init__(self, scheduler: _FakeScheduler) -> None:
        self.engine_core = _FakeEngineCore(scheduler)


class _FakeEngine:
    def __init__(self, scheduler: _FakeScheduler) -> None:
        self.engine_core = _FakeEngineCoreClient(scheduler)


class _FakeTokenizer:
    def __init__(self, tokenized: dict[str, list[int]]) -> None:
        self._tokenized = tokenized

    def __call__(self, prompt: str) -> dict[str, list[int]]:
        return {"input_ids": self._tokenized[prompt]}

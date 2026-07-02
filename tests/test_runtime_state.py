from __future__ import annotations

from dataclasses import dataclass

import pytest

from joint_decode_gpu.runtime_state import runtime_state
from joint_decode_gpu.worker import (
    _drain_worker_commands,
    _local_decision_boundary,
    _post_decision,
    _set_held_request_ids,
)


def test_worker_drains_runtime_abort_command() -> None:
    runtime_state.reset()
    runtime_state.publish_commands(abort="boom")

    with pytest.raises(RuntimeError, match="boom"):
        _drain_worker_commands()
    assert runtime_state.latest_commands is None


def test_local_decision_boundary_requires_no_live_pending_tokens() -> None:
    runtime_state.reset()
    runtime_state.pending_tokens["r0"] = [7]

    assert not _local_decision_boundary({"r0", "r1"})
    assert _local_decision_boundary({"r1"})


def test_compute_holds_skips_prefill_and_holds_empty_decode_rows() -> None:
    runtime_state.reset()
    runtime_state.pending_tokens["busy"] = [7]
    scheduler = _FakeScheduler(
        requests={
            "busy-internal": _FakeRequest("busy", num_computed_tokens=5, num_prompt_tokens=5),
            "empty-internal": _FakeRequest("empty", num_computed_tokens=5, num_prompt_tokens=5),
            "prefill-internal": _FakeRequest("prefill", num_computed_tokens=2, num_prompt_tokens=5),
        }
    )
    engine = _FakeEngine(scheduler)

    _set_held_request_ids(engine, {"busy", "empty", "prefill"})

    assert scheduler.held_request_ids == {"empty-internal"}


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
    rid: str
    num_computed_tokens: int
    num_prompt_tokens: int

    @property
    def sampling_params(self) -> _FakeSamplingParams:
        return _FakeSamplingParams(self.rid)


@dataclass(frozen=True)
class _FakeSamplingParams:
    rid: str

    @property
    def extra_args(self) -> dict[str, str]:
        return {"joint_decode_rid": self.rid}


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

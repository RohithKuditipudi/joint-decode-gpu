from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from joint_decode_gpu.worker import _commands, _max_live_requests, _read_start


def _fake_engine(max_num_seqs: int) -> SimpleNamespace:
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    return SimpleNamespace(
        vllm_config=object(),
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler)),
    )


def test_max_live_requests_floors_kv_concurrency() -> None:
    assert _max_live_requests(_fake_engine(max_num_seqs=64), lambda *_args: 7.9) == 7


def test_max_live_requests_caps_at_max_num_seqs() -> None:
    assert _max_live_requests(_fake_engine(max_num_seqs=4), lambda *_args: 7.9) == 4


def test_commands_skips_blank_lines_and_parses_json() -> None:
    lines = ["\n", "  \n", json.dumps({"command": "start", "initial_admit": ["r0"]}) + "\n"]

    assert list(_commands(lines)) == [{"command": "start", "initial_admit": ["r0"]}]


def test_read_start_returns_initial_admit() -> None:
    commands = iter([{"command": "start", "initial_admit": ["r0", "r1"]}])

    assert _read_start(commands) == ["r0", "r1"]


def test_read_start_rejects_other_commands() -> None:
    with pytest.raises(RuntimeError, match="expected start command"):
        _read_start(iter([{"command": "shutdown"}]))


def test_read_start_rejects_closed_stream() -> None:
    with pytest.raises(RuntimeError, match="expected start command"):
        _read_start(iter([]))

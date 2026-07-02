from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from joint_decode_gpu.coordinator import Coordinator, Side


def test_decode_pair_returns_side_local_token_lists() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    coordinator.begin_run(["r0"], 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))

    assert future_a.result()["tokens"] == {"r0": [11]}
    assert future_b.result()["tokens"] == {"r0": [22]}


def test_selector_empty_token_list_aborts_both_waiters() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([], [22]))
    coordinator.begin_run(["r0"], 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))

    with pytest.raises(ValueError, match="empty token list"):
        future_a.result()
    with pytest.raises(ValueError, match="empty token list"):
        future_b.result()


def test_finish_wakes_peer_decode_with_force_stop() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    coordinator.begin_run(["r0"], 1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        _wait_for_pending_decode(coordinator, Side.A)
        coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})

    assert future.result()["force_stop"] == ["r0"]


def test_joint_limit_returns_side_local_eos_for_shared_decode() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]), max_joint_decisions=1)
    coordinator.begin_run(["r0"], 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))

    assert future_a.result()["tokens"] == {"r0": [11]}
    assert future_b.result()["tokens"] == {"r0": [22]}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))

    assert future_a.result()["tokens"] == {"r0": [101]}
    assert future_b.result()["tokens"] == {"r0": [202]}


def test_begin_run_resets_phase1_chunk_state() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    assert coordinator.begin_run(["r0", "r1"], 2) == ["r0", "r1"]
    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}, {"rid": "r1"}]})
    coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}, {"rid": "r1"}]})

    assert coordinator.begin_run(["r2"], 1) == ["r2"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r2"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r2"]))

    assert future_a.result()["tokens"] == {"r2": [11]}
    assert future_b.result()["tokens"] == {"r2": [22]}


def _coordinator(selector, *, max_joint_decisions: int = 10) -> Coordinator:
    coordinator = Coordinator(
        timeout_s=1.0,
        select_tokens=selector,
        rng=random.Random(0),
        max_joint_decisions=max_joint_decisions,
    )
    coordinator.set_eos_token_ids(a=101, b=202)
    return coordinator


def _decode(side: str, request_ids: list[str]) -> dict[str, object]:
    return {
        "kind": "decode",
        "side": side,
        "request_ids": request_ids,
        "topk": {rid: [{"token_id": 1, "logit": 1.0}] for rid in request_ids},
    }


def _wait_for_pending_decode(coordinator: Coordinator, side: Side) -> None:
    for _ in range(1000):
        with coordinator._lock:
            if side in coordinator._pending_decode:
                return
        time.sleep(0.001)
    raise AssertionError(f"side {side.value} did not post a pending decode")

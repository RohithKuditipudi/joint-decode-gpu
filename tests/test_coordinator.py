from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from joint_decode_gpu.coordinator import Coordinator, Side, SidePlan, _validated_side_plan


def test_decode_pair_returns_side_local_token_lists() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    coordinator.begin_run(["r0"], 1, *_plans(["r0"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))

    assert future_a.result()["tokens"] == {"r0": [11]}
    assert future_b.result()["tokens"] == {"r0": [22]}


def test_selector_empty_token_list_aborts_both_waiters() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([], [22]))
    coordinator.begin_run(["r0"], 1, *_plans(["r0"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))

    with pytest.raises(ValueError, match="empty token list"):
        future_a.result()
    with pytest.raises(ValueError, match="empty token list"):
        future_b.result()


def test_finish_wakes_peer_decode_with_force_stop() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    coordinator.begin_run(["r0"], 1, *_plans(["r0"]))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        _wait_for_pending_decode(coordinator, Side.A)
        coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})

    assert future.result()["force_stop"] == ["r0"]


def test_begin_run_resets_run_state() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    assert coordinator.begin_run(["r0", "r1"], 2, *_plans(["r0", "r1"])) == ["r0", "r1"]
    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}, {"rid": "r1"}]})
    coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}, {"rid": "r1"}]})

    assert coordinator.begin_run(["r2"], 1, *_plans(["r2"])) == ["r2"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r2"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r2"]))

    assert future_a.result()["tokens"] == {"r2": [11]}
    assert future_b.result()["tokens"] == {"r2": [22]}


def test_decode_pair_admits_from_derived_capacity_after_retirement() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    assert coordinator.begin_run(["r0", "r1", "r2"], 2, *_plans(["r0", "r1", "r2"])) == ["r0", "r1"]
    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}]})
    coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r1"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r1"]))

    assert future_a.result()["admit"] == ["r2"]
    assert future_b.result()["admit"] == ["r2"]


def test_decode_pair_does_not_admit_with_multi_token_tail() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11, 12], [22]))
    assert coordinator.begin_run(["r0", "r1", "r2"], 2, *_plans(["r0", "r1", "r2"])) == ["r0", "r1"]
    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}]})
    coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r1"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r1"]))

    assert future_a.result()["tokens"] == {"r1": [11, 12]}
    assert future_a.result()["admit"] == []
    assert future_b.result()["admit"] == []


def test_control_pair_admits_when_window_has_capacity() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    assert coordinator.begin_run(["r0", "r1"], 1, *_plans(["r0", "r1"])) == ["r0"]
    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}]})
    coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _control("a"))
        future_b = pool.submit(coordinator.handle, "b", _control("b"))

    assert future_a.result() == {"admit": ["r1"], "abort": None, "done": False}
    assert future_b.result() == {"admit": ["r1"], "abort": None, "done": False}


def test_control_pair_returns_done_at_end() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    assert coordinator.begin_run(["r0"], 1, *_plans(["r0"])) == ["r0"]
    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}]})
    coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _control("a"))
        future_b = pool.submit(coordinator.handle, "b", _control("b"))

    assert future_a.result()["done"] is True
    assert future_b.result()["done"] is True


def test_control_decode_force_stops_terminal_decode_side() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    assert coordinator.begin_run(["r0"], 1, *_plans(["r0"])) == ["r0"]
    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_control = pool.submit(coordinator.handle, "a", _control("a"))
        _wait_for_pending_control(coordinator, Side.A)
        future_decode = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))
        assert future_decode.result()["force_stop"] == ["r0"]
        coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})

    assert future_control.result()["done"] is True


def test_control_decode_aborts_when_decode_side_still_needs_peer_logits() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    assert coordinator.begin_run(["r0"], 1, *_plans(["r0"])) == ["r0"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_control = pool.submit(coordinator.handle, "a", _control("a"))
        _wait_for_pending_control(coordinator, Side.A)
        future_decode = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))

    with pytest.raises(RuntimeError, match="worker is idle"):
        future_control.result()
    with pytest.raises(RuntimeError, match="worker is idle"):
        future_decode.result()


def test_begin_run_initial_admits_limited_by_token_budget() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    plan = SidePlan(budget=100, prompt_tokens={"r0": 60, "r1": 40, "r2": 10})

    assert coordinator.begin_run(["r0", "r1", "r2"], 3, plan, plan) == ["r0", "r1"]


def test_admission_respects_tighter_side_budget() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    plan_a = SidePlan(budget=1000, prompt_tokens={"r0": 10, "r1": 10})
    plan_b = SidePlan(budget=25, prompt_tokens={"r0": 20, "r1": 20})

    assert coordinator.begin_run(["r0", "r1"], 2, plan_a, plan_b) == ["r0"]


def test_blocked_head_admits_after_retirement_frees_budget() -> None:
    coordinator = _coordinator(lambda *_args, **_kwargs: ([11], [22]))
    plan = SidePlan(budget=100, prompt_tokens={"r0": 10, "r1": 100, "r2": 10})
    assert coordinator.begin_run(["r0", "r1", "r2"], 3, plan, plan) == ["r0"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _decode("a", ["r0"]))
        future_b = pool.submit(coordinator.handle, "b", _decode("b", ["r0"]))
    # The live decode row keeps the head prompt over budget, and FIFO order
    # must not skip ahead to r2 even though r2 alone would fit.
    assert future_a.result()["admit"] == []
    assert future_b.result()["admit"] == []

    coordinator.handle("a", {"kind": "finish", "side": "a", "finished": [{"rid": "r0"}]})
    coordinator.handle("b", {"kind": "finish", "side": "b", "finished": [{"rid": "r0"}]})
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(coordinator.handle, "a", _control("a"))
        future_b = pool.submit(coordinator.handle, "b", _control("b"))

    assert future_a.result()["admit"] == ["r1"]
    assert future_b.result()["admit"] == ["r1"]


def test_validated_side_plan_accepts_matching_payload() -> None:
    plan = _validated_side_plan(
        {"prompt_tokens": {"r0": 3, "r1": 128}},
        request_ids=["r0", "r1"],
        max_model_len=128,
        budget=256,
        side="a",
    )

    assert plan == SidePlan(budget=256, prompt_tokens={"r0": 3, "r1": 128})


@pytest.mark.parametrize(
    ("payload_tokens", "match"),
    [
        ({"r0": 3}, "do not match"),
        ({"r0": 3, "r1": 4, "r2": 5}, "do not match"),
        ({"r0": 0, "r1": 4}, "empty prompt"),
        ({"r0": 129, "r1": 4}, "over max_model_len"),
        ({"r0": 100, "r1": 4}, "over max_num_batched_tokens"),
    ],
)
def test_validated_side_plan_rejects_invalid_payloads(
    payload_tokens: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _validated_side_plan(
            {"prompt_tokens": payload_tokens},
            request_ids=["r0", "r1"],
            max_model_len=128,
            budget=64,
            side="a",
        )


def _coordinator(selector) -> Coordinator:
    return Coordinator(
        timeout_s=1.0,
        select_tokens=selector,
        rng=random.Random(0),
    )


def _plans(request_ids: list[str]) -> tuple[SidePlan, SidePlan]:
    plan = SidePlan(budget=1_000_000, prompt_tokens={rid: 1 for rid in request_ids})
    return plan, plan


def _decode(side: str, request_ids: list[str]) -> dict[str, object]:
    return {
        "kind": "decode",
        "side": side,
        "request_ids": request_ids,
        "topk": {rid: [{"token_id": 1, "logit": 1.0}] for rid in request_ids},
    }


def _control(side: str) -> dict[str, object]:
    return {
        "kind": "control",
        "side": side,
    }


def _wait_for_pending_decode(coordinator: Coordinator, side: Side) -> None:
    for _ in range(1000):
        with coordinator._lock:
            if side in coordinator._pending_decode:
                return
        time.sleep(0.001)
    raise AssertionError(f"side {side.value} did not post a pending decode")


def _wait_for_pending_control(coordinator: Coordinator, side: Side) -> None:
    for _ in range(1000):
        with coordinator._lock:
            if side in coordinator._pending_control:
                return
        time.sleep(0.001)
    raise AssertionError(f"side {side.value} did not post a pending control")

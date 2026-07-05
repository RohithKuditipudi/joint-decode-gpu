from __future__ import annotations

from types import SimpleNamespace

from joint_decode_gpu.worker import _max_live_requests


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

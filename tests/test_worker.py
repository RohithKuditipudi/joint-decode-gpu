from __future__ import annotations

from types import SimpleNamespace

from joint_decode_gpu.worker import _max_live_requests


def test_max_live_requests_uses_minimum_scheduler_capacity() -> None:
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        scheduler_config=SimpleNamespace(
            max_num_seqs=64,
            max_num_batched_tokens=4096,
        ),
    )
    engine = SimpleNamespace(
        vllm_config=object(),
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler)),
    )

    capacity = _max_live_requests(
        engine,
        lambda _vllm_config, _kv_cache_config: 512.8,
        max_model_len=128,
    )

    assert capacity == 32

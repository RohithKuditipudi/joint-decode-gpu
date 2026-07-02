from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from joint_decode_gpu.runtime_state import runtime_state


@dataclass
class RequestState:
    rid: str
    output_token_ids: list[int]


class JointDecodeLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool) -> None:
        del is_pin_memory
        self.device = device
        self.decision_url = _required_env("RERANK_TOKEN_DECISION_URL")
        self.side = _required_env("RERANK_TOKEN_DECISION_SIDE")
        self.top_k = int(_required_env("RERANK_TOKEN_DECISION_TOP_K"))
        self.timeout = float(_required_env("RERANK_TOKEN_DECISION_TIMEOUT"))
        if self.top_k < 1:
            raise ValueError("RERANK_TOKEN_DECISION_TOP_K must be >= 1")
        if self.timeout <= 0:
            raise ValueError("RERANK_TOKEN_DECISION_TIMEOUT must be > 0")
        self.eos_token_id = _eos_token_id(vllm_config)
        self._rows: dict[int, RequestState] = {}
        self._batch_size = 0

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args
        if not extra_args or "joint_decode_rid" not in extra_args:
            raise ValueError("SamplingParams.extra_args must include joint_decode_rid")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for index in batch_update.removed:
            self._rows.pop(index, None)

        for index, params, _, output_token_ids in batch_update.added:
            extra_args = params.extra_args
            if not extra_args or "joint_decode_rid" not in extra_args:
                raise ValueError("SamplingParams.extra_args must include joint_decode_rid")
            self._rows[index] = RequestState(
                rid=str(extra_args["joint_decode_rid"]),
                output_token_ids=output_token_ids,
            )

        for source, dest, direction in batch_update.moved:
            source_state = self._rows.get(source)
            dest_state = self._rows.get(dest)
            if source_state is None:
                raise RuntimeError(f"vLLM moved missing source row {source} -> {dest}")
            self._rows[dest] = source_state
            if direction == MoveDirectionality.SWAP:
                if dest_state is None:
                    raise RuntimeError(f"vLLM swapped source row {source} with empty row {dest}")
                self._rows[source] = dest_state
            elif direction == MoveDirectionality.UNIDIRECTIONAL:
                self._rows.pop(source, None)
            else:
                raise RuntimeError(f"unknown vLLM move direction {direction!r}")

        self._batch_size = batch_update.batch_size
        for index in tuple(self._rows):
            if index >= self._batch_size:
                del self._rows[index]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"expected 2D logits tensor, got shape={tuple(logits.shape)}")
        forced_by_rid: dict[str, int] = {}
        decode_rows: list[tuple[int, RequestState]] = []
        topk_payload: dict[str, list[dict[str, int | float]]] = {}
        k = min(self.top_k, logits.shape[-1])

        for row in range(logits.shape[0]):
            state = self._rows.get(row)
            if state is None:
                raise RuntimeError(f"missing request state for logits row {row}")
            pending = runtime_state.pending_tokens.get(state.rid)
            if pending:
                forced_by_rid[state.rid] = pending.pop(0)
                if not pending:
                    runtime_state.pending_tokens.pop(state.rid, None)
            else:
                decode_rows.append((row, state))

        if decode_rows:
            top_values, top_indices = torch.topk(logits, k=k, dim=-1)
            top_values_cpu = top_values.detach().cpu()
            top_indices_cpu = top_indices.detach().cpu()
            request_ids = [state.rid for _, state in decode_rows]
            for row, state in decode_rows:
                topk_payload[state.rid] = [
                    {
                        "token_id": int(token_id),
                        "logit": float(logit),
                    }
                    for token_id, logit in zip(
                        top_indices_cpu[row].tolist(),
                        top_values_cpu[row].tolist(),
                        strict=True,
                    )
                ]

            response = self._post_decision(
                {
                    "kind": "decode",
                    "side": self.side,
                    "request_ids": request_ids,
                    "topk": topk_payload,
                }
            )
            self._apply_response(response, forced_by_rid, {state.rid for _, state in decode_rows})

        forced = torch.tensor(
            [forced_by_rid[self._rows[row].rid] for row in range(logits.shape[0])],
            dtype=torch.long,
            device=logits.device,
        ).unsqueeze(-1)
        out = torch.full_like(logits, -float("inf"))
        out.scatter_(-1, forced, 0.0)
        return out

    def _apply_response(
        self,
        response: dict[str, Any],
        forced_by_rid: dict[str, int],
        decoded_rids: set[str],
    ) -> None:
        abort = response.get("abort")
        if abort:
            runtime_state.publish_commands(abort=str(abort))
            raise RuntimeError(str(abort))
        runtime_state.publish_commands(admit=response.get("admit") or [])

        for rid, token_list in (response.get("tokens") or {}).items():
            if isinstance(token_list, int):
                tokens = [token_list]
            else:
                tokens = list(token_list)
            if not tokens:
                raise RuntimeError(f"coordinator returned an empty token list for rid={rid}")
            forced_by_rid[rid] = int(tokens.pop(0))
            if tokens:
                runtime_state.pending_tokens[rid] = [int(token) for token in tokens]
            else:
                runtime_state.pending_tokens.pop(rid, None)

        for rid in response.get("force_stop") or []:
            if rid in decoded_rids:
                forced_by_rid[rid] = self.eos_token_id
                runtime_state.pending_tokens.pop(rid, None)

        missing = decoded_rids - set(forced_by_rid)
        if missing:
            raise RuntimeError(f"coordinator did not return tokens or force_stop for rids={sorted(missing)}")

    def _post_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.decision_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"joint-decode decision request failed: {exc}") from exc


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} must be set")
    return value


def _eos_token_id(vllm_config: Any) -> int:
    eos_token_id = vllm_config.model_config.hf_config.eos_token_id
    if isinstance(eos_token_id, list):
        if not eos_token_id:
            raise RuntimeError("model EOS token list must not be empty")
        eos_token_id = eos_token_id[0]
    if eos_token_id is None:
        raise RuntimeError("model config must define eos_token_id")
    return int(eos_token_id)

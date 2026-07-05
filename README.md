# joint-decode-gpu

Standalone GPU joint decoding for two vLLM models, including models with
different tokenizers.

At each decode step, worker A and worker B each send top-k logits for the same
active request ids to a parent HTTP coordinator. The coordinator chooses one
side-local token list per request. Each worker then masks its logits so vLLM emits
the chosen local token. Output text is taken from worker A.

## CLI

We provide a CLI as a simple end-to-end demonstration.

```bash
uv run joint-decode-gpu --output completions.jsonl
```

Important flags:

- `--model-a`, `--model-b`: model paths or HF ids. Tokenizers may differ.
- `--gpu-a`, `--gpu-b`: physical GPU ids. The parent sets
  `CUDA_VISIBLE_DEVICES` separately for each worker.
- `--top-k-a`, `--top-k-b`: top-k payload size each worker sends to the
  coordinator.
- `--advisor-weight`: weight on model B in the default average-logits rule.
  Model A weight is `1 - advisor_weight`.
- `--temperature`: joint sampling temperature used by the coordinator.
- `--max-microbatch-size`: cap on live requests in the synchronized sliding
  window. The effective window is the minimum of this cap and each worker's KV
  cache capacity, and it ramps up over the first decision rounds within the
  per-step token budget (`--max-num-batched-tokens`).
- `--max-tokens`: shared side-local vLLM generation cap.
- `--max-tokens-a`, `--max-tokens-b`: optional side-specific generation caps.
- `--prompts`: path to JSONL file.
  Each row must contain `id`, `prompt_a`, and `prompt_b` —
  `prompt_a` is fed to model A, `prompt_b` to model B. They can differ (e.g.
  different chat templates or system prompts):

  ```json
  {"id": "p0", "prompt_a": "Write a short proof that...", "prompt_b": "Write a short proof that..."}
  {"id": "p1", "prompt_a": "Explain why...", "prompt_b": "Explain why..."}
  ```

The CLI writes JSONL, one row per input row in input order:

```json
{"id": "p0", "prompt_a": "...", "prompt_b": "...", "completion": "...", "finish_reason": "..."}
```

`completion` is the model-A continuation.

## Custom Aggregation

The CLI uses average-logits aggregation. Experiments can bypass the CLI and call
`run_joint_decode` directly with their own token selector. A selector can be
an arbitrary function that takes two lists of tokens and associated logits
(i.e., the top-k tokens from each model) as well as an RNG instance and returns
side-local token lists. Returning a single `int` is supported only as a shorthand
for forcing the same token id on both sides. For example:

```python
import random
from typing import Any

from joint_decode_gpu.coordinator import run_joint_decode

def select_dumb(
    a_topk: list[dict[str, Any]], # [{"token_id": int, "logit": float},...]
    b_topk: list[dict[str, Any]],
    *,
    rng: random.Random,
):
  return [a_topk[0]["token_id"]], [b_topk[0]["token_id"]]

outputs = run_joint_decode(
    config,
    prompts_a,
    prompts_b,
    select_token=select_dumb
)
```

We have the option of running the two models with different prompts. This is
useful for when we want to provide hints (in the prompt) to the second model.

## Tests

The core correctness test is a single-worker vLLM workload with a deterministic
HTTP decision coordinator. It forces distinct token streams per request id and
asserts vLLM outputs by request id. This tests the GPU-specific row-to-request-id
mapping in `JointDecodeLogitsProcessor`.

Install the project and test dependencies:

```bash
uv sync
```

Run it with a local tiny vLLM-compatible model:

```bash
JOINT_DECODE_GPU_TEST_MODEL="Qwen/Qwen3-0.6B" \
uv run pytest -q tests/test_joint_decode_logits_processor_gpu.py -m gpu
```

Without `JOINT_DECODE_GPU_TEST_MODEL`, the test skips.

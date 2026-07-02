from __future__ import annotations

import math
import random
from typing import Any


def select_avg_logits(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    advisor_weight: float,
    temperature: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    if not 0.0 <= advisor_weight <= 1.0:
        raise ValueError("advisor_weight must be in [0, 1]")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")
    a_logits = {int(t["token_id"]): float(t["logit"]) for t in a_topk}
    b_logits = {int(t["token_id"]): float(t["logit"]) for t in b_topk}
    if not a_logits or not b_logits:
        raise ValueError("both sides must provide at least one top-k logit")

    a_floor = min(a_logits.values())
    b_floor = min(b_logits.values())
    w_a, w_b = 1.0 - advisor_weight, advisor_weight
    union = list(set(a_logits) | set(b_logits))
    scores = [w_a * a_logits.get(token_id, a_floor) + w_b * b_logits.get(token_id, b_floor) for token_id in union]
    if temperature == 0.0:
        token = union[scores.index(max(scores))]
        return [token], [token]
    max_score = max(scores)
    weights = [math.exp((score - max_score) / temperature) for score in scores]
    token = rng.choices(union, weights=weights, k=1)[0]
    return [token], [token]


def select_product_of_experts(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    temperature: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    return select_avg_logits(
        a_topk,
        b_topk,
        advisor_weight=0.5,
        temperature=temperature,
        rng=rng,
    )

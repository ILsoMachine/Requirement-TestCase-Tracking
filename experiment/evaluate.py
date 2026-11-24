"""Shared ranking metrics, CSV output, and API usage accounting."""

import csv
import json


def score_ranking(ranked_ids, related, candidate_ids, display_k, rerank_k):
    if len(ranked_ids) != len(candidate_ids) or set(ranked_ids) != candidate_ids:
        raise ValueError("ranking is not a permutation of the complete candidate library")
    if not related or not related <= candidate_ids:
        raise ValueError("invalid reference IDs")
    hits = 0
    precision_sum = 0.0
    for position, test_id in enumerate(ranked_ids, 1):
        if test_id in related:
            hits += 1
            precision_sum += hits / position
    return {
        f"recall_at_{display_k}": len(related.intersection(ranked_ids[:display_k])) / len(related),
        f"recall_at_{rerank_k}": len(related.intersection(ranked_ids[:rerank_k])) / len(related),
        "ap": precision_sum / len(related),
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                for key, value in row.items()
            })


def add_usage(totals, response):
    usage = response.get("usage") or {}
    if not usage:
        return
    totals["responses_with_usage"] += 1
    totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
    hits = usage.get("prompt_cache_hit_tokens", 0)
    totals["prompt_cache_hit_tokens"] += hits
    totals["prompt_cache_miss_tokens"] += usage.get("prompt_cache_miss_tokens", usage["prompt_tokens"] - hits)
    totals["completion_tokens"] += usage.get("completion_tokens", 0)

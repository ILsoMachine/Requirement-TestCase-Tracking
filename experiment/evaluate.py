"""Offline scoring against the original TRIAD reference matrix."""

from collections import Counter, defaultdict
import csv
import json
from statistics import mean, median

from .common import DATA, file_hash, read_artifacts, read_json, write_json

METHODS = ("bm25", "semantic", "deepseek_rerank")


def load_references():
    references = defaultdict(set)
    with (DATA / "reference_links.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            references[row["requirement_id"]].add(row["test_case_id"])
    return references


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


def evaluate(run_dir):
    config = read_json(run_dir / "config.json")
    baselines = read_json(run_dir / "baselines.json")
    references = load_references()
    query_ids = sorted(references)
    candidate_ids = set(read_artifacts("test_cases.csv"))
    display_k, rerank_k = config["display_k"], config["rerank_k"]
    display_metric, candidate_metric = f"recall_at_{display_k}", f"recall_at_{rerank_k}"
    if any(set(baselines[method]) != set(query_ids) for method in METHODS[:2]):
        raise ValueError("baseline queries do not match the scored reference queries")

    records = []
    deltas = []
    ceilings = []
    attempts = 0
    retry_count = 0
    statuses = Counter()
    evidence = Counter()
    usage = Counter()
    model_ids = set()
    request_ids = set()
    for query_id in query_ids:
        reranked = read_json(run_dir / "rerank" / f"{query_id}.json")
        if reranked["query_id"] != query_id:
            raise ValueError(f"reranker output does not match query {query_id}")
        semantic = baselines["semantic"][query_id]
        original = semantic["ranked_ids"]
        ranked = reranked["ranked_ids"]
        if (
            reranked["baseline_top_k"] != original[:rerank_k]
            or set(original[:rerank_k]) != set(ranked[:rerank_k])
            or original[rerank_k:] != ranked[rerank_k:]
        ):
            raise ValueError(f"reranker changed the candidate pool for query {query_id}")
        statuses[reranked["status"]] += 1
        evidence.update(row["evidence_status"] for row in (reranked["assessments"] or {}).values())
        attempts += len(reranked["attempts"])
        retry_count += len(reranked["attempts"]) > 1
        for attempt in reranked["attempts"]:
            response = attempt.get("response") or {}
            if response.get("model"):
                model_ids.add(response["model"])
            if response.get("id"):
                response_id = response["id"]
                if response_id in request_ids:
                    raise ValueError(f"duplicate API response ID for query {query_id}")
                request_ids.add(response_id)
            add_usage(usage, response)

        related = references[query_id]
        query_scores = {}
        for method, row in zip(METHODS, (baselines["bm25"][query_id], semantic, reranked)):
            scores = score_ranking(row["ranked_ids"], related, candidate_ids, display_k, rerank_k)
            records.append({
                "query_id": query_id, "method": method, "ranked_ids": row["ranked_ids"],
                "reference_ids": sorted(related), **scores,
                "elapsed_ms": row["elapsed_ms"], "rerank_status": row.get("status", ""),
            })
            query_scores[method] = scores
        before = query_scores["semantic"]
        after = query_scores["deepseek_rerank"]
        candidate_hits = len(related.intersection(original[:rerank_k]))
        ceiling = min(display_k, candidate_hits) / len(related)
        ceilings.append(ceiling)
        deltas.append({
            "query_id": query_id,
            "reference_ids": sorted(related),
            "semantic_top_5": original[:display_k],
            "rerank_top_5": ranked[:display_k],
            "semantic_top_10": original[:rerank_k],
            "semantic_recall_at_5": before[display_metric],
            "rerank_recall_at_5": after[display_metric],
            "delta_recall_at_5": after[display_metric] - before[display_metric],
            "delta_ap": after["ap"] - before["ap"],
            "rerank_recall_at_5_ceiling": ceiling,
        })

    method_summaries = []
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        method_summaries.append({
            "method": method,
            "queries": len(rows),
            display_metric: mean(row[display_metric] for row in rows),
            candidate_metric: mean(row[candidate_metric] for row in rows),
            "map": mean(row["ap"] for row in rows),
            "median_elapsed_ms": median(row["elapsed_ms"] for row in rows),
        })
    delta_values = [row["delta_recall_at_5"] for row in deltas]
    summary = {
        "reference_links_sha256": file_hash(DATA / "reference_links.csv"),
        "queries": len(query_ids),
        "candidates": len(candidate_ids),
        "reference_links": sum(map(len, references.values())),
        "methods": method_summaries,
        "paired_rerank_vs_semantic": {
            "mean_recall_at_5_delta": mean(delta_values),
            "mean_ap_delta": mean(row["delta_ap"] for row in deltas),
            "improved": sum(value > 0 for value in delta_values),
            "unchanged": sum(value == 0 for value in delta_values),
            "worsened": sum(value < 0 for value in delta_values),
            "mean_recall_at_5_ceiling_given_candidates": mean(ceilings),
        },
        "deepseek": {
            "attempts": attempts, "retried_queries": retry_count,
            "reranked": statuses["reranked"], "fallback": statuses["fallback"],
            "response_models": sorted(model_ids), "evidence_status_counts": dict(evidence),
            "usage": dict(usage),
        },
    }
    write_csv(run_dir / "per-query.csv", records)
    write_csv(run_dir / "paired.csv", deltas)
    write_csv(run_dir / "summary.csv", method_summaries)
    write_json(run_dir / "summary.json", summary)
    return summary

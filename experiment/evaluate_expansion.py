"""Score each project and estimate paired changes by resampling requirements."""

from collections import Counter, defaultdict
import csv
import math
import random
from statistics import mean, median

from .common import file_hash, read_json, write_json
from .evaluate import add_usage, score_ranking, write_csv
from .expansion import check_snapshot, dataset_dir, load_baselines, load_inputs, record_path

CONTRASTS = (
    ("semantic", "bm25"),
    ("bm25_deepseek_k10", "bm25"),
    ("semantic_deepseek_k10", "semantic"),
    ("semantic_deepseek_k30", "semantic"),
    ("semantic_deepseek_k30", "semantic_deepseek_k10"),
    ("semantic_deepseek_k10", "bm25_deepseek_k10"),
)


def load_references(path):
    references = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            references[row["requirement_id"]].add(row["test_case_id"])
    return references


def percentile(samples, probability):
    position = (len(samples) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    return samples[left] + (position - left) * (samples[right] - samples[left])


def paired_interval(values, samples, seed):
    generator = random.Random(seed)
    simulated = sorted(sum(generator.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return [percentile(simulated, 0.025), percentile(simulated, 0.975)]


def baseline_ceiling(baseline, related, display_k, candidate_k):
    return min(display_k, len(related.intersection(baseline[:candidate_k]))) / len(related)


def check_reranking(record, query_id, original, cutoff):
    ranked = record["ranked_ids"]
    if (
        record["query_id"] != query_id
        or record["baseline_top_k"] != original[:cutoff]
        or set(ranked[:cutoff]) != set(original[:cutoff])
        or ranked[cutoff:] != original[cutoff:]
    ):
        raise ValueError(f"{query_id}: inconsistent reranking candidate set")
    if record["status"] == "fallback":
        if ranked != original or record["assessments"] is not None:
            raise ValueError(f"{query_id}: inconsistent fallback")
    elif record["status"] == "reranked":
        assessments = record["assessments"]
        if set(assessments) != set(original[:cutoff]):
            raise ValueError(f"{query_id}: incomplete assessments")
        expected = sorted(original[:cutoff], key=lambda candidate: -assessments[candidate]["effective_grade"])
        if expected != ranked[:cutoff]:
            raise ValueError(f"{query_id}: unstable grade-based ordering")
    else:
        raise ValueError(f"{query_id}: unsupported reranking status")


def score_project(run_dir, dataset, config, manifest):
    specification = config["datasets"][dataset]
    baselines = load_baselines(run_dir, dataset)
    queries, cases = load_inputs(dataset, specification)
    candidate_ids = set(cases)
    reference_path = dataset_dir(dataset) / "reference_links.csv"
    references = load_references(reference_path)
    if set(references) != set(queries) or any(set(baselines[m]) != set(queries) for m in ("bm25", "semantic")):
        raise ValueError(f"{dataset}: evaluation queries and labels do not agree")

    treatments = config["treatments"]
    display_k = config["base"]["display_k"]
    query_records = []
    usage = {}
    counts = {}
    evidence_status = {}
    response_models = {}
    request_ids = set()
    for method in ("bm25", "semantic", *treatments):
        usage[method] = Counter()
        counts[method] = Counter()
        evidence_status[method] = Counter()
        response_models[method] = set()
        for query_id, related in sorted(references.items()):
            parameters = treatments.get(method)
            ceiling, cutoff = "", ""
            if parameters:
                path = record_path(run_dir, dataset, method, query_id)
                row = read_json(path)
                original = baselines[parameters["baseline"]][query_id]["ranked_ids"]
                cutoff = min(parameters["candidate_k"], len(cases))
                check_reranking(row, query_id, original, cutoff)
                ceiling = baseline_ceiling(original, related, display_k, cutoff)
                counts[method]["attempts"] += len(row["attempts"])
                counts[method]["retried_queries"] += len(row["attempts"]) > 1
                counts[method][row["status"]] += 1
                evidence_status[method].update(
                    grade["evidence_status"] for grade in (row["assessments"] or {}).values()
                )
                for attempt in row["attempts"]:
                    response = attempt.get("response") or {}
                    if response.get("model"):
                        response_models[method].add(response["model"])
                    if response.get("id"):
                        if response["id"] in request_ids:
                            raise ValueError(f"{dataset}: duplicate response ID for {method}")
                        request_ids.add(response["id"])
                    add_usage(usage[method], response)
            else:
                row = baselines[method][query_id]
            ranked = row["ranked_ids"]
            metrics = score_ranking(ranked, related, candidate_ids, display_k, 10)
            query_records.append({
                "dataset": dataset,
                "query_id": query_id, "method": method, "ranked_ids": ranked,
                "reference_ids": sorted(related),
                **metrics,
                "recall_at_30": len(related.intersection(ranked[:30])) / len(related),
                "top5_hits": len(related.intersection(ranked[:display_k])),
                "elapsed_ms": row["elapsed_ms"], "rerank_status": row.get("status", ""),
                "rerank_candidate_k": cutoff, "rerank_recall_at_5_ceiling": ceiling,
            })

    sources = manifest.get("method_sources", {}).get(dataset, {})
    default_source = manifest["default_result_source"]
    summary, indexed = [], {}
    for method in ("bm25", "semantic", *treatments):
        rows = [row for row in query_records if row["method"] == method]
        indexed[method] = {row["query_id"]: row for row in rows}
        summary.append({
            "dataset": dataset,
            "method": method,
            "queries": len(rows), "candidates": len(cases), "reference_links": sum(map(len, references.values())),
            "recall_at_5": mean(row["recall_at_5"] for row in rows),
            "recall_at_10": mean(row["recall_at_10"] for row in rows),
            "recall_at_30": mean(row["recall_at_30"] for row in rows),
            "map": mean(row["ap"] for row in rows),
            "top5_hits": sum(row["top5_hits"] for row in rows),
            "median_elapsed_ms": median(row["elapsed_ms"] for row in rows),
            "effective_rerank_k": rows[0]["rerank_candidate_k"],
            "mean_recall_at_5_ceiling": (
                mean(row["rerank_recall_at_5_ceiling"] for row in rows) if method in treatments else ""
            ),
            "retries": counts[method]["retried_queries"],
            "fallbacks": counts[method]["fallback"],
            "api_attempts": counts[method]["attempts"],
            "source": sources.get(method, default_source),
        })

    contrasts, paired_rows = [], []
    for after, before in CONTRASTS:
        per_metric = defaultdict(list)
        for query_id in sorted(references):
            after_row, before_row = indexed[after][query_id], indexed[before][query_id]
            deltas = {metric: after_row[metric] - before_row[metric] for metric in ("recall_at_5", "ap")}
            paired_rows.append({
                "dataset": dataset, "query_id": query_id, "after": after, "before": before,
                "reference_ids": sorted(references[query_id]),
                "before_top5": before_row["ranked_ids"][:display_k],
                "after_top5": after_row["ranked_ids"][:display_k],
                "before_top10": before_row["ranked_ids"][:10],
                "before_recall_at_5": before_row["recall_at_5"],
                "after_recall_at_5": after_row["recall_at_5"],
                "delta_recall_at_5": deltas["recall_at_5"], "delta_ap": deltas["ap"],
                "after_rerank_recall_at_5_ceiling": after_row["rerank_recall_at_5_ceiling"],
            })
            for metric, delta in deltas.items():
                per_metric[metric].append(delta)
        for metric, values in per_metric.items():
            seed = f"{config['base']['seed']}:{dataset}:{after}:{before}:{metric}"
            low, high = paired_interval(values, config["bootstrap_resamples"], seed)
            contrasts.append({
                "dataset": dataset, "after": after, "before": before, "metric": metric,
                "mean_delta": mean(values), "ci95_low": low, "ci95_high": high,
                "improved": sum(delta > 0 for delta in values),
                "unchanged": sum(delta == 0 for delta in values),
                "worsened": sum(delta < 0 for delta in values),
            })

    write_csv(run_dir / dataset / "per-query.csv", query_records)
    write_csv(run_dir / dataset / "paired.csv", paired_rows)
    detail = {"reference_links_sha256": file_hash(reference_path), "api": {}}
    for method in treatments:
        counts[method]["invalid_evidence"] = evidence_status[method]["invalid"]
        detail["api"][method] = {
            "counts": dict(counts[method]),
            "usage": dict(usage[method]),
            "response_models": sorted(response_models[method]),
            "evidence_status_counts": dict(sorted(evidence_status[method].items())),
        }
    return summary, contrasts, detail


def aggregate_api(details):
    counts, usage, evidence = Counter(), Counter(), Counter()
    models = set()
    for project in details.values():
        for result in project["api"].values():
            counts.update(result["counts"])
            usage.update(result["usage"])
            evidence.update(result["evidence_status_counts"])
            models.update(result["response_models"])
    return {
        "counts": dict(counts), "usage": dict(usage), "response_models": sorted(models),
        "evidence_status_counts": dict(sorted(evidence.items())),
    }


def evaluate(run_dir):
    config = read_json(run_dir / "config.json")
    manifest = check_snapshot(run_dir)
    summaries, contrasts, details = [], [], {}
    for dataset in config["datasets"]:
        results, changes, detail = score_project(run_dir, dataset, config, manifest)
        summaries.extend(results)
        contrasts.extend(changes)
        details[dataset] = detail
        print(f"{dataset}: scored {len(results)} methods", flush=True)
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "contrasts.csv", contrasts)
    write_json(run_dir / "summary.json", {
        "methods": summaries,
        "contrasts": contrasts,
        "details": details,
        "api_totals": aggregate_api(details),
    })

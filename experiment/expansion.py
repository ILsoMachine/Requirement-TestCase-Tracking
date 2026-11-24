"""Prepare, run, or score the cross-project EBT, Mozilla, and EDC experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import platform
import random
from threading import Event
from importlib.metadata import version

from .common import CONFIG, MODEL_DIR, ROOT, file_hash, read_json, utc_now, write_json
from .rerank import ApiConfigurationError, DeepSeek, make_request, rerank
from .retrieval import MiniLM, bm25
from .run import load_prompts

DESIGN = ROOT / "config" / "expansion.json"
DEFAULT_RUN = ROOT / "results" / "expansion-20261124"
TABLES = ("requirements.csv", "test_cases.csv")
REUSED_TREATMENT = "semantic_deepseek_k10"


def dataset_dir(dataset):
    return ROOT / "data" / "processed" / dataset


def artifact_texts(dataset, filename):
    with (dataset_dir(dataset) / filename).open(encoding="utf-8", newline="") as stream:
        return {row["id"]: row["text"] for row in csv.DictReader(stream)}


def load_inputs(dataset, specification):
    queries = artifact_texts(dataset, TABLES[0])
    for query_id in specification["excluded_query_ids"]:
        del queries[query_id]
    return queries, artifact_texts(dataset, TABLES[1])


def load_baselines(run_dir, dataset, specification):
    if "prior_run" in specification:
        return read_json(ROOT / specification["prior_run"] / "baselines.json")
    return read_json(run_dir / dataset / "baselines.json")


def record_path(run_dir, dataset, specification, treatment, query_id):
    if treatment == REUSED_TREATMENT and "prior_run" in specification:
        return ROOT / specification["prior_run"] / "rerank" / f"{query_id}.json"
    return run_dir / dataset / treatment / f"{query_id}.json"


def treatment_config(base, treatment):
    return {
        **base, "rerank_k": treatment["candidate_k"],
        "reranker": {**base["reranker"], "max_tokens": treatment["max_tokens"]},
    }


def hashed_artifacts(dataset):
    return {name: file_hash(dataset_dir(dataset) / name) for name in TABLES}


def check_legacy(specification, hashes, base, prompts):
    path = ROOT / specification["prior_run"]
    legacy = read_json(path / "manifest.json")
    if legacy["ranking_input_sha256"] != hashes:
        raise ValueError("legacy EBT text changed")
    if read_json(path / "config.json") != base or read_json(path / "prompts.json") != prompts:
        raise ValueError("legacy EBT configuration or prompts changed")


def prepare(run_dir):
    if (run_dir / "manifest.json").exists():
        raise ValueError("run exists; choose a new --run-dir")
    base = read_json(CONFIG)
    design = read_json(DESIGN)
    prompts = load_prompts()
    embedding = base["embedding"]
    hashes = {
        "model.onnx": file_hash(MODEL_DIR / "model.onnx"),
        "tokenizer.json": file_hash(MODEL_DIR / "tokenizer.json"),
    }
    if (hashes["model.onnx"], hashes["tokenizer.json"]) != (
        embedding["onnx_sha256"], embedding["tokenizer_sha256"]
    ):
        raise ValueError("the encoder does not match the pinned files")
    encoder = MiniLM(MODEL_DIR, embedding)
    input_hashes = {}
    baseline_hashes = {}
    for dataset, specification in design["datasets"].items():
        input_hashes[dataset] = hashed_artifacts(dataset)
        if "prior_run" in specification:
            check_legacy(specification, input_hashes[dataset], base, prompts)
            baseline_path = ROOT / specification["prior_run"] / "baselines.json"
        else:
            queries, cases = load_inputs(dataset, specification)
            lexical = bm25(queries, cases, **base["bm25"])
            semantic, metadata = encoder.rank(queries, cases)
            baseline_path = run_dir / dataset / "baselines.json"
            write_json(baseline_path, {"bm25": lexical, "semantic": semantic, "embedding_metadata": metadata})
            print(f"{dataset}: prepared {len(queries)} queries and {len(cases)} candidates", flush=True)
        baseline_hashes[dataset] = file_hash(baseline_path)
    write_json(run_dir / "config.json", {"base": base, **design})
    write_json(run_dir / "prompts.json", prompts)
    write_json(run_dir / "manifest.json", {
        "prepared_at_utc": utc_now(),
        "ranking_input_sha256": input_hashes,
        "baselines_sha256": baseline_hashes,
        "embedding_sha256": hashes,
        "script_sha256": {name: file_hash(Path(__file__).parent / name)
                          for name in ("expansion.py", "retrieval.py", "rerank.py", "import_expansion.py")},
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     **{package: version(package) for package in ("numpy", "onnxruntime", "requests", "tokenizers",
                                                                  "openpyxl")}},
    })


def check_snapshot(run_dir, config, audit_code=False):
    manifest = read_json(run_dir / "manifest.json")
    for dataset, expected in manifest["ranking_input_sha256"].items():
        if hashed_artifacts(dataset) != expected:
            raise ValueError(f"ranking input text changed: {dataset}")
        prior = config["datasets"][dataset].get("prior_run")
        folder = ROOT / prior if prior else run_dir / dataset
        if file_hash(folder / "baselines.json") != manifest["baselines_sha256"][dataset]:
            raise ValueError(f"baseline rankings changed: {dataset}")
    if audit_code:
        for name, expected in manifest["script_sha256"].items():
            if file_hash(Path(__file__).parent / name) != expected:
                raise ValueError(f"ranking script changed: {name}")


def pending_jobs(run_dir, config, prompts):
    jobs = []
    for dataset, specification in config["datasets"].items():
        queries, cases = load_inputs(dataset, specification)
        baselines = load_baselines(run_dir, dataset, specification)
        for treatment, parameters in config["treatments"].items():
            current = treatment_config(config["base"], parameters)
            baseline = baselines[parameters["baseline"]]
            for query_id, text in queries.items():
                path = record_path(run_dir, dataset, specification, treatment, query_id)
                if path.exists():
                    record = read_json(path)
                    top = baseline[query_id]["ranked_ids"][:current["rerank_k"]]
                    request, aliases = make_request(query_id, text, top, cases, prompts, current)
                    if record["request"] != request or record["aliases"] != aliases:
                        raise ValueError(f"stored request differs from frozen protocol: {path}")
                    continue
                if treatment == REUSED_TREATMENT and "prior_run" in specification:
                    raise ValueError(f"missing legacy EBT response: {path}")
                jobs.append((path, query_id, text, baseline[query_id], cases, current))
    random.Random(config["base"]["seed"]).shuffle(jobs)
    return jobs


def run_batch(jobs, settings, prompts, stop):
    with DeepSeek(settings) as client:
        try:
            for path, query_id, text, baseline, cases, config in jobs:
                if stop.is_set():
                    return
                record = rerank(query_id, text, baseline, cases, prompts, config, client)
                write_json(path, record)
        except BaseException:
            stop.set()
            raise


def run_api(run_dir, workers):
    config = read_json(run_dir / "config.json")
    check_snapshot(run_dir, config, audit_code=True)
    prompts = read_json(run_dir / "prompts.json")
    jobs = pending_jobs(run_dir, config, prompts)
    print(f"{len(jobs)} unsaved requests, {workers} parallel workers", flush=True)
    if not jobs:
        return
    execution = run_dir / "execution.json"
    sessions = read_json(execution) if execution.exists() else []
    sessions.append({"started_at_utc": utc_now(), "workers": workers, "pending_jobs": len(jobs)})
    write_json(execution, sessions)
    stop = Event()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_batch, jobs[i::workers], config["base"]["reranker"], prompts, stop)
            for i in range(min(workers, len(jobs)))
        ]
        for future in futures:
            future.result()
    print(f"Saved {len(jobs)} reranking records", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "rerank", "score"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        if args.stage == "prepare":
            prepare(args.run_dir)
        elif args.stage == "rerank":
            run_api(args.run_dir, args.workers)
        else:
            from .evaluate_expansion import evaluate

            evaluate(args.run_dir)
    except (OSError, ValueError, ApiConfigurationError) as error:
        parser.exit(1, f"{args.stage}: {error}\n")


if __name__ == "__main__":
    main()

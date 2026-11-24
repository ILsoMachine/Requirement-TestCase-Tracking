"""Prepare, run, or score the cross-project EBT, Mozilla, and EDC experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import platform
import random
from threading import Event
from importlib.metadata import version

from .common import CONFIG, MODEL_DIR, PROMPTS, ROOT, file_hash, read_json, utc_now, write_json
from .rerank import ApiConfigurationError, DeepSeek, make_request, rerank
from .retrieval import MiniLM, bm25
DEFAULT_RUN = ROOT / "results" / "experiment-20261124"
TABLES = ("requirements.csv", "test_cases.csv")
RANKING_SCRIPTS = ("common.py", "expansion.py", "retrieval.py", "rerank.py", "import_expansion.py")
SNAPSHOTS = ("config.json", "prompts.json")


def load_prompts():
    return {
        kind: (PROMPTS / f"rerank-{kind}.txt").read_text(encoding="utf-8").strip()
        for kind in ("system", "task")
    }


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


def load_baselines(run_dir, dataset):
    return read_json(run_dir / dataset / "baselines.json")


def record_path(run_dir, dataset, treatment, query_id):
    return run_dir / dataset / treatment / f"{query_id}.json"


def treatment_config(base, treatment):
    return {
        **base, "rerank_k": treatment["candidate_k"],
        "reranker": {**base["reranker"], "max_tokens": treatment["max_tokens"]},
    }


def hashed_artifacts(dataset):
    return {name: file_hash(dataset_dir(dataset) / name) for name in TABLES}


def smoke(run_dir, design, prompts):
    requirement = "Reject an expired invitation."
    cases = {
        "T1": "Attempt to redeem an expired invitation. The page reports that it was rejected.",
        "T2": "Redeem a valid invitation. The page opens the registration form.",
        "T3": "Change the display name in the profile. The updated name appears.",
    }
    baseline = {"ranked_ids": ["T2", "T3", "T1"], "elapsed_ms": 0}
    config = treatment_config(design["base"], design["treatments"]["semantic_deepseek_k10"])
    with DeepSeek(config["reranker"]) as client:
        record = rerank("smoke", requirement, baseline, cases, prompts, config, client)
    write_json(run_dir / "smoke.json", record)
    if record["status"] != "reranked" or record["ranked_ids"][0] != "T1":
        raise ValueError("the hand-written live smoke check did not pass; see smoke.json")
    print("Live smoke passed")


def prepare(run_dir):
    if (run_dir / "manifest.json").exists():
        raise ValueError("run exists; choose a new --run-dir")
    design = read_json(CONFIG)
    base = design["base"]
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
        queries, cases = load_inputs(dataset, specification)
        lexical = bm25(queries, cases, **base["bm25"])
        semantic, metadata = encoder.rank(queries, cases)
        baseline_path = run_dir / dataset / "baselines.json"
        write_json(baseline_path, {"bm25": lexical, "semantic": semantic, "embedding_metadata": metadata})
        print(f"{dataset}: prepared {len(queries)} queries and {len(cases)} candidates", flush=True)
        baseline_hashes[dataset] = file_hash(baseline_path)
    write_json(run_dir / "config.json", design)
    write_json(run_dir / "prompts.json", prompts)
    write_json(run_dir / "manifest.json", {
        "prepared_at_utc": utc_now(),
        "ranking_input_sha256": input_hashes,
        "baselines_sha256": baseline_hashes,
        "embedding_sha256": hashes,
        "snapshot_sha256": {name: file_hash(run_dir / name) for name in SNAPSHOTS},
        "script_sha256": {name: file_hash(Path(__file__).parent / name) for name in RANKING_SCRIPTS},
        "default_result_source": run_dir.resolve().name,
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     **{package: version(package) for package in ("numpy", "onnxruntime", "requests", "tokenizers",
                                                                  "openpyxl")}},
    })


def check_snapshot(run_dir, audit_code=False):
    manifest = read_json(run_dir / "manifest.json")
    for name, expected in manifest["snapshot_sha256"].items():
        if file_hash(run_dir / name) != expected:
            raise ValueError(f"run snapshot changed: {name}")
    for dataset, expected in manifest["ranking_input_sha256"].items():
        if hashed_artifacts(dataset) != expected:
            raise ValueError(f"ranking input text changed: {dataset}")
        if file_hash(run_dir / dataset / "baselines.json") != manifest["baselines_sha256"][dataset]:
            raise ValueError(f"baseline rankings changed: {dataset}")
    if audit_code:
        hashes = manifest.get("resume_script_sha256", manifest["script_sha256"])
        for name, expected in hashes.items():
            if file_hash(Path(__file__).parent / name) != expected:
                raise ValueError(f"ranking script changed: {name}")
    return manifest


def pending_jobs(run_dir, config, prompts):
    jobs = []
    for dataset, specification in config["datasets"].items():
        queries, cases = load_inputs(dataset, specification)
        baselines = load_baselines(run_dir, dataset)
        for treatment, parameters in config["treatments"].items():
            current = treatment_config(config["base"], parameters)
            baseline = baselines[parameters["baseline"]]
            for query_id, text in queries.items():
                path = record_path(run_dir, dataset, treatment, query_id)
                if path.exists():
                    record = read_json(path)
                    top = baseline[query_id]["ranked_ids"][:current["rerank_k"]]
                    request, aliases = make_request(query_id, text, top, cases, prompts, current)
                    if record["request"] != request or record["aliases"] != aliases:
                        raise ValueError(f"stored request differs from frozen protocol: {path}")
                    continue
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
    check_snapshot(run_dir, audit_code=True)
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
    parser.add_argument("stage", choices=("prepare", "baselines", "smoke", "rerank", "score"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        if args.stage in ("prepare", "baselines"):
            prepare(args.run_dir)
        elif args.stage == "smoke":
            smoke(args.run_dir, read_json(CONFIG), load_prompts())
        elif args.stage == "rerank":
            run_api(args.run_dir, args.workers)
        else:
            from .evaluate_expansion import evaluate

            evaluate(args.run_dir)
    except (OSError, ValueError, ApiConfigurationError) as error:
        parser.exit(1, f"{args.stage}: {error}\n")


if __name__ == "__main__":
    main()

"""Run or rescore the reproducible EBT retrieval experiment."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys

from .common import CONFIG, DATA, MODEL_DIR, PROMPTS, ROOT
from .common import file_hash, read_artifacts, read_json, utc_now, write_json
from .rerank import ApiConfigurationError, DeepSeek, rerank
from .retrieval import MiniLM, bm25

DEFAULT_RUN = ROOT / "results" / "ebt-20261124"


def load_prompts():
    return {
        kind: (PROMPTS / f"rerank-{kind}.txt").read_text(encoding="utf-8").strip()
        for kind in ("system", "task")
    }


def load_inputs(config):
    queries = read_artifacts("requirements.csv")
    for query_id in config["excluded_query_ids"]:
        del queries[query_id]
    return queries, read_artifacts("test_cases.csv")


def smoke(run_dir, config, prompts):
    requirement = "Reject an expired invitation."
    cases = {
        "T1": "Attempt to redeem an expired invitation. The page reports that it was rejected.",
        "T2": "Redeem a valid invitation. The page opens the registration form.",
        "T3": "Change the display name in the profile. The updated name appears.",
    }
    baseline = {"ranked_ids": ["T2", "T3", "T1"], "elapsed_ms": 0}
    with DeepSeek(config["reranker"]) as client:
        record = rerank("smoke", requirement, baseline, cases, prompts, config, client)
    write_json(run_dir / "smoke.json", record)
    if record["status"] != "reranked" or record["ranked_ids"][0] != "T1":
        raise ValueError("the hand-written live smoke check did not pass; see smoke.json")
    print("Live smoke passed")


def baselines(run_dir, config, prompts):
    path = run_dir / "baselines.json"
    if path.exists():
        raise ValueError(f"{path.name} already exists; use a fresh --run-dir")

    model_hash = file_hash(MODEL_DIR / "model.onnx")
    if model_hash != config["embedding"]["onnx_sha256"]:
        raise ValueError("MiniLM ONNX file does not match the published SHA-256")
    tokenizer_hash = file_hash(MODEL_DIR / "tokenizer.json")
    if tokenizer_hash != config["embedding"]["tokenizer_sha256"]:
        raise ValueError("MiniLM tokenizer does not match the pinned SHA-256")
    manifest = {
        "started_at_utc": utc_now(),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{package: version(package) for package in ("numpy", "onnxruntime", "tokenizers", "requests")},
        },
        "ranking_input_sha256": {
            name: file_hash(DATA / name) for name in ("requirements.csv", "test_cases.csv")
        },
        "script_sha256": {
            path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "embedding_files_sha256": {
            "model.onnx": model_hash,
            "tokenizer.json": tokenizer_hash,
        },
    }
    queries, cases = load_inputs(config)
    lexical = bm25(queries, cases, **config["bm25"])
    semantic, metadata = MiniLM(MODEL_DIR, config["embedding"]).rank(queries, cases)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "prompts.json", prompts)
    write_json(run_dir / "manifest.json", manifest)
    write_json(path, {"bm25": lexical, "semantic": semantic, "embedding_metadata": metadata})
    print(f"Saved both baselines for {len(queries)} queries against {len(cases)} candidates")


def rerank_all(run_dir):
    config = read_json(run_dir / "config.json")
    prompts = read_json(run_dir / "prompts.json")
    manifest = read_json(run_dir / "manifest.json")
    for name, expected in manifest["ranking_input_sha256"].items():
        if file_hash(DATA / name) != expected:
            raise ValueError(f"{name} changed since baseline generation")
    for name, expected in manifest["script_sha256"].items():
        if file_hash(Path(__file__).parent / name) != expected:
            raise ValueError(f"{name} changed since baseline generation")
    queries, cases = load_inputs(config)
    semantic = read_json(run_dir / "baselines.json")["semantic"]
    with DeepSeek(config["reranker"]) as client:
        for index, (query_id, text) in enumerate(queries.items(), 1):
            path = run_dir / "rerank" / f"{query_id}.json"
            if path.exists():
                continue
            record = rerank(query_id, text, semantic[query_id], cases, prompts, config, client)
            write_json(path, record)
            print(f"[{index}/{len(queries)}] {query_id}: {record['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "baselines", "rerank", "score"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    try:
        if args.stage == "smoke":
            smoke(args.run_dir, read_json(CONFIG), load_prompts())
        elif args.stage == "baselines":
            baselines(args.run_dir, read_json(CONFIG), load_prompts())
        elif args.stage == "rerank":
            rerank_all(args.run_dir)
        else:
            from .evaluate import evaluate

            summary = evaluate(args.run_dir)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ApiConfigurationError, OSError, ValueError) as error:
        print(f"{args.stage}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

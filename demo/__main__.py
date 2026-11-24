"""Serve the traceability prototype on the local computer."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from time import perf_counter

from experiment.common import MODEL_DIR, file_hash, read_json
from experiment.expansion import (
    DEFAULT_RUN, TABLES, artifact_texts, check_snapshot, load_baselines,
    record_path, treatment_config,
)
from experiment.rerank import ApiConfigurationError, DeepSeek, rerank
from experiment.retrieval import MiniLM, bm25


ASSETS = Path(__file__).parent
STATIC = {
    "/": ("index.html", "text/html"),
    "/demo.css": ("demo.css", "text/css"),
    "/demo.js": ("demo.js", "text/javascript"),
}


class Demo:
    def __init__(self):
        check_snapshot(DEFAULT_RUN)
        self.config = read_json(DEFAULT_RUN / "config.json")
        self.prompts = read_json(DEFAULT_RUN / "prompts.json")
        self.artifacts = {
            project: tuple(artifact_texts(project, filename) for filename in TABLES)
            for project in self.config["datasets"]
        }
        self.baselines = {
            project: load_baselines(DEFAULT_RUN, project) for project in self.artifacts
        }
        self.encoder = None
        self.encoder_lock = Lock()

    def catalog(self):
        return {
            "projects": {
                project: {
                    "queries": queries,
                    "count": len(cases),
                    "archived": list(self.baselines[project]["semantic"]),
                }
                for project, (queries, cases) in self.artifacts.items()
            },
            "treatments": self.config["treatments"],
            "display_k": self.config["base"]["display_k"],
            "archive": DEFAULT_RUN.name,
            "model": self.config["base"]["reranker"]["model"],
        }

    def retrieve(self, method, query, cases):
        start = perf_counter()
        base = self.config["base"]
        if method == "bm25":
            rows = bm25(query, cases, **base["bm25"])
        else:
            with self.encoder_lock:
                if self.encoder is None:
                    embedding = base["embedding"]
                    hashes = (file_hash(MODEL_DIR / "model.onnx"),
                              file_hash(MODEL_DIR / "tokenizer.json"))
                    if hashes != (embedding["onnx_sha256"], embedding["tokenizer_sha256"]):
                        raise ValueError("MiniLM 文件与实验配置不一致，请运行 python -m experiment.download。")
                    self.encoder = MiniLM(MODEL_DIR, embedding)
                rows, _ = self.encoder.rank(query, cases)
        elapsed_ms = (perf_counter() - start) * 1000
        return {**next(iter(rows.values())), "elapsed_ms": elapsed_ms}

    def compare(self, project, query_id, treatment, mode):
        if project not in self.artifacts or treatment not in self.config["treatments"]:
            raise ValueError("请选择有效的项目与对比方法。")
        queries, cases = self.artifacts[project]
        if query_id not in queries or mode not in ("archive", "live", "local"):
            raise ValueError("请选择有效的需求与运行方式。")
        settings = self.config["treatments"][treatment]
        method = settings["baseline"]
        cutoff = min(settings["candidate_k"], len(cases))
        baseline, record, notice = None, None, ""
        if mode == "archive":
            baseline = self.baselines[project][method].get(query_id)
            if baseline is None:
                raise ValueError("此需求没有归档记录，请切换到实时运行。")
            record = read_json(record_path(DEFAULT_RUN, project, treatment, query_id))
        else:
            baseline = self.retrieve(method, {query_id: queries[query_id]}, cases)
            if mode == "live":
                config = treatment_config(self.config["base"], settings)
                try:
                    with DeepSeek(config["reranker"]) as client:
                        record = rerank(query_id, queries[query_id], baseline, cases,
                                        self.prompts, config, client)
                except ApiConfigurationError as error:
                    notice = f"DeepSeek 未完成调用，已保留实时检索顺序。{error}"
                    record = {"ranked_ids": baseline["ranked_ids"], "status": "fallback",
                              "elapsed_ms": None, "assessments": None}

        display_k = self.config["base"]["display_k"]
        before = baseline["ranked_ids"]
        after = record["ranked_ids"] if record else []
        before_ranks = {case: rank for rank, case in enumerate(before, 1)}
        after_ranks = {case: rank for rank, case in enumerate(after, 1)}
        visible = dict.fromkeys(before[:display_k] + after[:display_k])
        return {
            "mode": mode,
            "status": record["status"] if record else "local",
            "notice": notice,
            "cutoff": cutoff,
            "requirement": queries[query_id],
            "baseline_ms": baseline["elapsed_ms"],
            "total_ms": record["elapsed_ms"] if record else None,
            "before": before[:display_k],
            "after": after[:display_k],
            "cases": {
                case: {"text": cases[case], "before": before_ranks[case],
                       "after": after_ranks.get(case),
                       "assessment": ((record or {}).get("assessments") or {}).get(case)}
                for case in visible
            },
        }


class Handler(BaseHTTPRequestHandler):
    def __init__(self, app, *args, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def send_bytes(self, payload, media_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", f"{media_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                        "application/json", status)

    def do_GET(self):
        if self.path in STATIC:
            file, media_type = STATIC[self.path]
            self.send_bytes((ASSETS / file).read_bytes(), media_type)
        elif self.path == "/api/catalog":
            self.send_json(self.app.catalog())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/compare":
            self.send_error(404)
            return
        host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != host or self.headers.get("Origin") not in (None, f"http://{host}"):
            self.send_error(403)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_error(415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("无效的请求长度。")
            request = json.loads(self.rfile.read(length))
            fields = ("project", "query", "treatment", "mode")
            if not isinstance(request, dict) or any(not isinstance(request.get(key), str) for key in fields):
                raise ValueError("无效的请求参数。")
            project, query, treatment, mode = (request[key] for key in fields)
        except (UnicodeError, ValueError) as error:
            self.send_json({"error": str(error)}, 400)
            return
        try:
            result = self.app.compare(project, query, treatment, mode)
        except ValueError as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.log_error("Demo run failed: %s", type(error).__name__)
            self.send_json({"error": "运行失败。请检查本地依赖和 MiniLM 模型文件，或切换到归档回放。"}, 503)
        else:
            self.send_json(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, Demo())) as server:
            print(f"Traceability demo: http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    except (OSError, ValueError, OverflowError) as error:
        parser.exit(1, f"demo: {error}\n")


if __name__ == "__main__":
    main()

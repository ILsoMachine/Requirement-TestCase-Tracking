"""DeepSeek candidate grading, exact-quote validation, and stable reranking."""

import json
import os
import random
from time import perf_counter, sleep
from urllib.parse import urlsplit

from .common import ROOT, read_json, utc_now


class ApiConfigurationError(RuntimeError):
    pass


def make_request(query_id, requirement, candidate_ids, test_cases, prompts, config):
    displayed = list(candidate_ids)
    random.Random(f"{config['seed']}:{query_id}").shuffle(displayed)
    aliases = {f"C{index:02}": test_id for index, test_id in enumerate(displayed, 1)}
    data = {
        "requirement": requirement,
        "candidates": [{"alias": alias, "text": test_cases[test_id]}
                       for alias, test_id in aliases.items()],
    }
    settings = config["reranker"]
    request = {
        "model": settings["model"],
        "thinking": {"type": settings["thinking"]},
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["task"] + "\n\n" + json.dumps(data, ensure_ascii=False)},
        ],
    }
    return request, aliases


def validate_response(response, aliases, requirement, test_cases):
    choice, = response["choices"]
    if choice.get("finish_reason") != "stop":
        raise ValueError("response did not finish normally")
    assessments = json.loads(choice["message"]["content"])["assessments"]
    if not isinstance(assessments, list) or len(assessments) != len(aliases):
        raise ValueError("wrong number of assessments")
    seen = set()
    validated = {}
    for item in assessments:
        alias = item["alias"]
        grade = item["grade"]
        if alias not in aliases or alias in seen:
            raise ValueError("missing, unknown, or duplicate alias")
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError("invalid grade")
        seen.add(alias)
        test_id = aliases[alias]
        requirement_quote, test_quote, reason = (
            item["requirement_quote"], item["test_quote"], item["reason"]
        )
        if not all(isinstance(value, str) for value in (requirement_quote, test_quote, reason)):
            raise ValueError("quotes and reason must be strings")
        invalid = grade > 0 and (
            not requirement_quote.strip() or requirement_quote not in requirement
            or not test_quote.strip() or test_quote not in test_cases[test_id]
        )
        validated[test_id] = {
            "alias": alias,
            "grade": grade,
            "effective_grade": 0 if invalid else grade,
            "evidence_status": "invalid" if invalid else ("valid" if grade else "not_required"),
            "requirement_quote": requirement_quote,
            "test_quote": test_quote,
            "reason": reason,
        }
    return validated


class DeepSeek:
    def __init__(self, settings):
        import requests

        config_path = ROOT / "model.local.json"
        local = read_json(config_path) if config_path.is_file() else {}
        url = urlsplit(local.get("request_url", "https://api.deepseek.com"))
        if (url.scheme, url.netloc) != ("https", "api.deepseek.com") or url.query or url.fragment:
            raise ApiConfigurationError("configure the official DeepSeek HTTPS endpoint")
        path = url.path.rstrip("/")
        if path not in ("", "/v1", "/chat/completions", "/v1/chat/completions"):
            raise ApiConfigurationError("configure an OpenAI-compatible Chat Completions endpoint")
        self.url = f"https://api.deepseek.com{path}"
        if not path.endswith("/chat/completions"):
            self.url += "/chat/completions"
        api_key = os.environ.get("DEEPSEEK_API_KEY") or local.get("api_key")
        if not api_key or not api_key.strip():
            raise ApiConfigurationError("set DEEPSEEK_API_KEY or provide model.local.json")
        self.settings = settings
        self.requests = requests
        self.session = requests.Session()
        self.session.headers["Authorization"] = "Bearer " + api_key.strip()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    def assess(self, body, aliases, requirement, test_cases):
        attempts = []
        for attempt_no in range(1, self.settings["max_attempts"] + 1):
            start = perf_counter()
            attempt = {"attempt": attempt_no, "started_at_utc": utc_now()}
            validated = None
            try:
                response = self.session.post(
                    self.url, json=body, timeout=(10, self.settings["timeout_seconds"]),
                    allow_redirects=False,
                )
                attempt["http_status"] = response.status_code
                if response.status_code == 200:
                    attempt["response"] = response.json()
                    validated = validate_response(attempt["response"], aliases, requirement, test_cases)
                elif response.status_code in (408, 429) or response.status_code >= 500:
                    attempt["error"] = f"http_{response.status_code}"
                else:
                    raise ApiConfigurationError(f"DeepSeek returned HTTP {response.status_code}")
            except self.requests.RequestException as error:
                attempt["error"] = type(error).__name__
            except (ValueError, TypeError, KeyError) as error:
                attempt["error"] = "invalid_response:" + type(error).__name__
            attempt["elapsed_ms"] = (perf_counter() - start) * 1000
            attempts.append(attempt)
            if validated is not None:
                return validated, attempts
            if attempt_no < self.settings["max_attempts"]:
                sleep(self.settings["retry_delay_seconds"])
        return None, attempts


def rerank(query_id, requirement, baseline, test_cases, prompts, config, client):
    start = perf_counter()
    cutoff = config["rerank_k"]
    candidate_ids = baseline["ranked_ids"][:cutoff]
    request, aliases = make_request(query_id, requirement, candidate_ids, test_cases, prompts, config)
    validated, attempts = client.assess(request, aliases, requirement, test_cases)
    ranked_ids = baseline["ranked_ids"]
    if validated is not None:
        reordered = sorted(candidate_ids, key=lambda test_id: -validated[test_id]["effective_grade"])
        ranked_ids = reordered + ranked_ids[cutoff:]
    rerank_ms = (perf_counter() - start) * 1000
    return {
        "query_id": query_id,
        "baseline_top_k": candidate_ids,
        "ranked_ids": ranked_ids,
        "status": "fallback" if validated is None else "reranked",
        "baseline_ms": baseline["elapsed_ms"],
        "rerank_ms": rerank_ms,
        "elapsed_ms": baseline["elapsed_ms"] + rerank_ms,
        "endpoint": client.url,
        "request": request,
        "aliases": aliases,
        "attempts": attempts,
        "assessments": validated,
    }

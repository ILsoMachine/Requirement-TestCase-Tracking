"""Shared paths and serialization for EBT experiments."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ebt"
CONFIG = ROOT / "config" / "ebt.json"
PROMPTS = ROOT / "prompts"
MODEL_DIR = ROOT / ".cache" / "minilm"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_artifacts(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return {row["id"]: row["text"] for row in csv.DictReader(stream)}

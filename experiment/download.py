"""Fetch and verify the pinned CPU encoder and tokenizer from Hugging Face."""

from tempfile import NamedTemporaryFile
from pathlib import Path

import requests

from .common import CONFIG, MODEL_DIR, file_hash, read_json


def fetch(url, destination, digest):
    if destination.exists() and file_hash(destination) == digest:
        print(f"{destination.name}: verified")
        return
    with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    output.write(block)
        if file_hash(temporary) != digest:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        temporary.replace(destination)
        print(f"{destination.name}: downloaded and verified")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    embedding = read_json(CONFIG)["embedding"]
    base = f"https://huggingface.co/{embedding['repository']}/resolve/{embedding['revision']}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fetch(f"{base}/{embedding['onnx_file']}?download=true", MODEL_DIR / "model.onnx", embedding["onnx_sha256"])
    fetch(f"{base}/tokenizer.json?download=true", MODEL_DIR / "tokenizer.json", embedding["tokenizer_sha256"])

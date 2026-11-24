"""CPU baselines. The ranking code does not load reference links."""

from collections import Counter
from math import log1p
import re
from time import perf_counter

TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(text):
    return TOKEN.findall(text.casefold())


def ranking_row(scores, start):
    ranked_ids = sorted(scores, key=lambda test_id: (-scores[test_id], test_id))
    return {
        "ranked_ids": ranked_ids,
        "scores": [scores[test_id] for test_id in ranked_ids],
        "elapsed_ms": (perf_counter() - start) * 1000,
    }


def bm25(queries, candidates, k1, b):
    documents = {test_id: Counter(tokenize(text)) for test_id, text in candidates.items()}
    lengths = {test_id: sum(counts.values()) for test_id, counts in documents.items()}
    mean_length = sum(lengths.values()) / len(lengths)
    frequency = Counter(token for doc in documents.values() for token in doc)
    idf = {token: log1p((len(documents) - df + 0.5) / (df + 0.5))
           for token, df in frequency.items()}
    normalization = {test_id: k1 * (1 - b + b * length / mean_length)
                     for test_id, length in lengths.items()}
    result = {}
    for query_id, text in queries.items():
        start = perf_counter()
        terms = list(dict.fromkeys(tokenize(text)))
        scores = {}
        for test_id, document in documents.items():
            scores[test_id] = sum(
                idf[token] * document[token] * (k1 + 1)
                / (document[token] + normalization[test_id])
                for token in terms if token in document
            )
        result[query_id] = ranking_row(scores, start)
    return result


class MiniLM:
    def __init__(self, directory, config):
        import numpy as np
        import onnxruntime
        from tokenizers import Tokenizer

        self.np = np
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.cls = self.tokenizer.token_to_id("[CLS]")
        self.sep = self.tokenizer.token_to_id("[SEP]")
        self.pad = self.tokenizer.token_to_id("[PAD]")
        self.content_length = config["max_tokens"] - 2
        self.batch_size = config["batch_size"]
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = config["cpu_threads"]
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.session = onnxruntime.InferenceSession(
            str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {value.name for value in self.session.get_inputs()}

    def encode(self, texts):
        np = self.np
        segments = []
        counts = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            chunks = [ids[i:i + self.content_length] for i in range(0, len(ids), self.content_length)]
            counts.append(len(chunks))
            segments.extend([self.cls, *chunk, self.sep] for chunk in chunks)

        batches = []
        for offset in range(0, len(segments), self.batch_size):
            batch = segments[offset:offset + self.batch_size]
            shape = (len(batch), max(map(len, batch)))
            input_ids = np.full(shape, self.pad, dtype=np.int64)
            attention = np.zeros(shape, dtype=np.int64)
            for index, ids in enumerate(batch):
                input_ids[index, :len(ids)] = ids
                attention[index, :len(ids)] = 1
            inputs = {"input_ids": input_ids, "attention_mask": attention}
            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = np.zeros(shape, dtype=np.int64)
            hidden, = self.session.run(["last_hidden_state"], inputs)
            mask = attention[..., None].astype(hidden.dtype)
            mean = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            batches.append(mean / np.linalg.norm(mean, axis=1, keepdims=True))

        vectors = np.concatenate(batches)
        boundaries = np.cumsum([0, *counts])
        return [vectors[a:b] for a, b in zip(boundaries, boundaries[1:])], counts

    def rank(self, queries, candidates):
        candidate_ids = list(candidates)
        start = perf_counter()
        vectors, counts = self.encode(candidates.values())
        index_ms = (perf_counter() - start) * 1000
        rankings = {}
        query_segments = {}
        for query_id, text in queries.items():
            start = perf_counter()
            (query_vector,), (segment_count,) = self.encode([text])
            scores = {
                test_id: float((query_vector @ case_vector.T).max())
                for test_id, case_vector in zip(candidate_ids, vectors)
            }
            rankings[query_id] = ranking_row(scores, start)
            query_segments[query_id] = segment_count
        metadata = {
            "candidate_index_ms": index_ms,
            "candidate_segments": dict(zip(candidate_ids, counts)),
            "query_segments": query_segments,
        }
        return rankings, metadata

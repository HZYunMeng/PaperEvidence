"""Optional local semantic retrieval. No network calls after model preparation."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .retrieval import BM25, Hit


E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def default_model_path():
    return Path(os.environ.get("PAPER_EVIDENCE_EMBEDDING_PATH", Path(__file__).resolve().parents[1] / "models" / "multilingual-e5-small"))


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Embedding must be a finite 2D matrix")
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Embedding contains a zero vector")
    return values / lengths


class E5Embedder:
    def __init__(self, path=None, device="cpu"):
        path = Path(path or default_model_path()).expanduser().resolve()
        if not (path / "model.safetensors").is_file() or not (path / "provenance.json").is_file():
            raise RuntimeError("语义模型未准备好。先运行 python scripts/prepare_embedding.py，或设置 PAPER_EVIDENCE_EMBEDDING_PATH。")
        provenance = json.loads((path / "provenance.json").read_text())
        if provenance.get("model_id") != E5_MODEL:
            raise ValueError("This adapter expects multilingual-e5-small")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements-semantic.txt。") from exc
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True, trust_remote_code=False)
        self.model.max_seq_length = 512
        self.metadata = {"model_id": E5_MODEL, "revision": provenance["revision"],
                         "max_tokens": 512, "window_overlap": 64, "device": device,
                         "adapter": "e5-token-windows-v1"}

    def windows(self, text):
        tokenizer = self.model.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        prefix_size = len(tokenizer.encode("passage: ", add_special_tokens=False))
        budget = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(False) - prefix_size
        if len(tokenizer.encode("passage: " + text, verbose=False)) <= 512:
            return [text]
        windows, start = [], 0
        while start < len(ids):
            end = min(start + budget, len(ids))
            window = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            # Decoding and re-tokenizing can change boundaries. Check the actual
            # prefixed model input instead of silently allowing encoder truncation.
            while len(tokenizer.encode("passage: " + window, verbose=False)) > 512:
                end -= 1
                if end <= start:
                    raise ValueError("Cannot fit embedding window within token budget")
                window = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            windows.append(window)
            if end == len(ids):
                break
            start = max(start + 1, end - 64)
        return windows

    def passages(self, texts):
        owners, windows = [], []
        for owner, text in enumerate(texts):
            for window in self.windows(text):
                owners.append(owner)
                windows.append("passage: " + window)
        vectors = self.model.encode(windows, normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=16, show_progress_bar=False)
        return vectors, np.asarray(owners, dtype=np.int64)

    def query(self, text):
        if len(self.model.tokenizer.encode("query: " + text, verbose=False)) > 512:
            raise ValueError("问题超过语义模型 512 token 限制，请缩短问题。")
        return self.model.encode(["query: " + text], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


class DenseRetriever:
    def __init__(self, chunks, embedder, cache_dir=None):
        self.chunks, self.embedder = tuple(chunks), embedder
        self.metadata = {"type": "dense", "embedding": embedder.metadata,
                         "aggregation": "max cosine across overlapping token windows"}
        manifest = {"chunks": [(c.id, c.text) for c in self.chunks], "embedding": embedder.metadata}
        self.fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.metadata["index_sha256"] = self.fingerprint
        if not self.chunks:
            self.vectors, self.owners = np.empty((0, 0)), np.empty((0,), dtype=np.int64)
            return
        path = Path(cache_dir) / (self.fingerprint + ".npz") if cache_dir else None
        if path and path.exists():
            with np.load(path, allow_pickle=False) as cached:
                vectors, owners = cached["vectors"], cached["owners"]
        else:
            vectors, owners = embedder.passages([c.text for c in self.chunks])
        self.vectors = unit_vectors(vectors)
        self.owners = np.asarray(owners)
        if (self.owners.ndim != 1 or len(self.owners) != len(self.vectors)
                or self.owners.dtype.kind not in "iu"
                or set(self.owners.tolist()) != set(range(len(self.chunks)))):
            raise ValueError("Embedding windows do not cover the current source chunks")
        if path and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Each cache write is atomic; no partial arrays are loaded by another session.
            import tempfile
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
                temp_path = Path(temporary.name)
            try:
                # Store the raw float32 vectors so cached and fresh paths normalize once.
                np.savez_compressed(temp_path, vectors=np.asarray(vectors, dtype=np.float32), owners=self.owners)
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)

    def search(self, question, k=5):
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        if not question.strip() or not self.chunks:
            return ()
        query = unit_vectors(self.embedder.query(question))
        if query.shape != (1, self.vectors.shape[1]):
            raise ValueError("Query embedding dimension does not match the index")
        similarities = np.clip(self.vectors @ query[0], -1, 1)
        scores = np.full(len(self.chunks), -np.inf)
        np.maximum.at(scores, self.owners, similarities)
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], self.chunks[i].id))
        return tuple(Hit(self.chunks[i], float(scores[i]), "cosine") for i in order[:k])


class HybridRetriever:
    def __init__(self, chunks, dense, candidates=20, rank_constant=60):
        if not 1 <= candidates <= 20 or rank_constant <= 0:
            raise ValueError("Invalid fusion configuration")
        self.lexical, self.dense = BM25(chunks), dense
        if self.lexical.chunks != dense.chunks:
            raise ValueError("Fusion requires identical source chunks")
        self.chunks = self.lexical.chunks
        self.candidates, self.rank_constant = candidates, rank_constant
        self.metadata = {"type": "hybrid-rrf", "candidates": candidates,
                         "rank_constant": rank_constant, "dense": dense.metadata}

    def search(self, question, k=5):
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        pool_size = max(k, self.candidates)
        scores, chunks = {}, {}
        for ranking in (self.lexical.search(question, pool_size), self.dense.search(question, pool_size)):
            for rank, hit in enumerate(ranking, 1):
                chunks[hit.chunk.id] = hit.chunk
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0) + 1 / (self.rank_constant + rank)
        ids = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
        return tuple(Hit(chunks[cid], scores[cid], "rrf") for cid in ids)


def make_retriever(chunks, mode="bm25", embedder=None, model_path=None, cache_dir=None):
    if mode == "bm25":
        return BM25(chunks)
    if mode not in {"dense", "hybrid"}:
        raise ValueError("Unknown retriever mode")
    dense = DenseRetriever(chunks, embedder or E5Embedder(model_path), cache_dir)
    return dense if mode == "dense" else HybridRetriever(chunks, dense)

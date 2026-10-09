"""Paired-language retrieval smoke test. No generator calls or answer-quality claims."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_evidence.answering import normalized
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf
from paper_evidence.retrieval import BM25
from paper_evidence.semantic import DenseRetriever, E5Embedder, HybridRetriever


def evaluate(index, cases, k):
    rows = []
    started = time.perf_counter()
    for case in cases:
        hits = index.search(case["question"], k)
        expected = case["evidence"]
        if not expected:
            raise ValueError("Answerable cases need evidence annotations")
        recovered = sum(any(h.chunk.page == e["page"] and normalized(e["quote"]) in normalized(h.chunk.text)
                            for h in hits) for e in expected)
        rows.append({"id":case["id"], "pair_id":case["pair_id"], "language":case["language"],
                     "question":case["question"], "evidence_recall":recovered / len(expected),
                     "hits":[{"chunk_id":h.chunk.id, "page":h.chunk.page, "score":h.score,
                              "score_kind":h.score_kind} for h in hits]})
    return {"mean_evidence_recall_at_k":sum(r["evidence_recall"] for r in rows)/len(rows),
            "by_language":{lang:sum(r["evidence_recall"] for r in rows if r["language"] == lang) /
                           sum(r["language"] == lang for r in rows)
                           for lang in sorted({r["language"] for r in rows})},
            "queries":len(rows), "query_seconds":time.perf_counter()-started, "rows":rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=ROOT / "examples/demo-paper.pdf")
    parser.add_argument("--qa", type=Path, default=ROOT / "examples/qa-multilingual.json")
    parser.add_argument("--embedding-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/semantic-smoke-results.json")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.k <= 20:
        parser.error("k must be between 1 and 20")
    cases = json.loads(args.qa.read_text(encoding="utf-8"))
    answerable = [c for c in cases if c["answerable"]]
    negative = [c for c in cases if not c["answerable"]]
    if not answerable or any(not c["evidence"] for c in answerable):
        parser.error("Answerable QA annotations must not be empty")
    paper = parse_pdf(args.pdf)
    chunks = chunk_blocks(paper.blocks)
    started = time.perf_counter()
    embedder = E5Embedder(args.embedding_path)
    # Intentionally rebuild the corpus embeddings; timing includes model startup.
    dense = DenseRetriever(chunks, embedder)
    preparation = time.perf_counter()-started
    indices = {"bm25":BM25(chunks), "dense":dense, "hybrid":HybridRetriever(chunks,dense)}
    result = {
        "dataset":"fictional multilingual smoke-test fixture" if args.pdf.name == "demo-paper.pdf" else "user-supplied annotations",
        "document_sha256":hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "qa_sha256":hashlib.sha256(args.qa.read_bytes()).hexdigest(),
        "python":platform.python_version(), "model":embedder.metadata,
        "index_sha256":dense.fingerprint, "embedding_preparation_seconds":preparation,
        "k":args.k, "chunks":len(chunks), "chunk_chars":1400,
        "metric":"fraction of annotated page/quote pairs fully present in at least one top-k chunk",
        "retrievers":{name:evaluate(index,answerable,args.k) for name,index in indices.items()},
        "unanswerable_probe":{
            name:[{"id":c["id"], "question":c["question"],
                   "candidate_count":len(index.search(c["question"],args.k))} for c in negative]
            for name,index in indices.items()},
        "limitations":"Fictional data, seven facts paired in two languages, no held-out tuning split. "
            "Results do not establish real-paper performance, answer correctness or calibrated abstention. "
            "Dense retrieval can return candidates for unanswerable queries; candidate count is not hallucination rate.",
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({name:{"all":r["mean_evidence_recall_at_k"], **r["by_language"]}
                      for name,r in result["retrievers"].items()},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

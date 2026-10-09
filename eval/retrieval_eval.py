"""Compare chunking on identical parsed blocks; no model quality claims."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_evidence.answering import normalized
from paper_evidence.chunking import chunk_blocks
from paper_evidence.models import Chunk
from paper_evidence.parsing import parse_pdf
from paper_evidence.retrieval import BM25


def fixed_chunks(blocks, size):
    pages = defaultdict(list)
    for block in blocks:
        if block.kind != "heading":
            pages[block.page].append(block)
    result = []
    for page, items in pages.items():
        text, spans = "", []
        for item in items:
            start = len(text)
            text += item.text + "\n"
            spans.append((start, len(text), item))
        for start in range(0, len(text), size):
            fragment = text[start:start + size]
            related = [b for a, z, b in spans if z > start and a < start + size]
            result.append(Chunk(hashlib.sha256(f"fixed:{page}:{start}:{fragment}".encode()).hexdigest()[:16],
                items[0].document_id, page, "Fixed window", "mixed", fragment,
                tuple(b.id for b in related), tuple(b.bbox for b in related)))
    return tuple(result)


def evaluate_retrieval(chunks, qa, k):
    index = BM25(chunks)
    rows = []
    for case in qa:
        expected = case["evidence"]
        if not expected:
            raise ValueError("Retrieval evaluation requires non-empty evidence annotations")
        hits = index.search(case["question"], k=k)
        recovered = sum(any(h.chunk.page == e["page"] and normalized(e["quote"]) in normalized(h.chunk.text)
                            for h in hits) for e in expected)
        rows.append({"id":case["id"], "evidence_recall":recovered/len(expected),
                     "retrieved_ids":[h.chunk.id for h in hits]})
    return {"mean_evidence_recall_at_k":sum(r["evidence_recall"] for r in rows)/len(rows),
            "queries":len(rows), "chunks":len(chunks), "rows":rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=Path(__file__).resolve().parents[1]/"examples/demo-paper.pdf")
    parser.add_argument("--qa", type=Path, default=Path(__file__).resolve().parents[1]/"examples/qa.json")
    parser.add_argument("--out", type=Path, default=Path("eval/results.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    args = parser.parse_args()
    if args.chunk_chars < 100:
        parser.error("--chunk-chars must be at least 100")
    paper = parse_pdf(args.pdf)
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    if not qa:
        parser.error("QA annotations must not be empty")
    result = {
        "dataset":"user-supplied annotations" if args.pdf.name != "demo-paper.pdf" else "fictional smoke-test fixture",
        "document_sha256":hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "qa_sha256":hashlib.sha256(args.qa.read_bytes()).hexdigest(),
        "parsed_blocks_sha256":hashlib.sha256(json.dumps([asdict(b) for b in paper.blocks],sort_keys=True).encode()).hexdigest(),
        "k":args.k,"chunk_chars":args.chunk_chars,"retriever":"BM25",
        "metric":"fraction of annotated page/quote pairs fully present in at least one top-k chunk",
        "fixed_window":evaluate_retrieval(fixed_chunks(paper.blocks,args.chunk_chars),qa,args.k),
        "section_aware":evaluate_retrieval(chunk_blocks(paper.blocks,args.chunk_chars),qa,args.k),
        "limitations":"Small retrieval-only test; does not measure answer correctness, hallucination or real scientific performance.",
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({name:result[name]["mean_evidence_recall_at_k"] for name in ("fixed_window","section_aware")}, indent=2))


if __name__ == "__main__":
    main()

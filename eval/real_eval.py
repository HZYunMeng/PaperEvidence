"""Development-corpus evidence retrieval, separating cell parsing from retrieval."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_evidence.answering import normalized
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf
from paper_evidence.retrieval import BM25
from paper_evidence.semantic import DenseRetriever, E5Embedder, HybridRetriever
from paper_evidence.tables import cell_reference


def region_contains(box, target):
    x,y = (target[0]+target[2])/2,(target[1]+target[3])/2
    return box[0]-1 <= x <= box[2]+1 and box[1]-1 <= y <= box[3]+1


def has_evidence(chunk, expected):
    return (chunk.page == expected["page"]
            and normalized(expected["quote"]) in normalized(chunk.text)
            and any(region_contains(box,expected["bbox"]) for box in chunk.boxes))


def compact(text):
    return "".join(normalized(text).split())


def has_bound_cell(chunk, expected):
    if not chunk.table or chunk.page != expected["page"]:
        return False
    wanted = expected["table"]
    for row in range(1,len(chunk.table.rows)):
        for column in range(1,len(chunk.table.rows[row])):
            try:
                reference = cell_reference(chunk,row,column)
            except ValueError:
                continue
            if (compact(reference["value"]) == compact(expected["quote"])
                    and compact(reference["row_label"]) == compact(wanted["row_label"])
                    and compact(reference["column_label"]) == compact(wanted["column_label"])
                    and ("row_context" not in wanted or compact(wanted["row_context"]) in
                         compact(" ".join(c.text for c in chunk.table.rows[row])))
                    and region_contains(reference["boxes"][2],expected["bbox"])):
                return True
    return False


def mean(rows,key):
    return sum(row[key] for row in rows)/len(rows) if rows else None


def summarize(rows):
    positive = [r for r in rows if r["answerable"]]
    tables = [r for r in positive if r["kind"] == "table"]
    return {"answerable_queries":len(positive),
            "evidence_recall_at_k":mean(positive,"evidence_recall"),
            "bound_cell_recall_at_k":mean(tables,"bound_cell_recall"),
            "by_language":{language:mean([r for r in positive if r["language"]==language],"evidence_recall")
                           for language in ("en","zh")},
            "by_paper":{paper:mean([r for r in positive if r["paper_id"]==paper],"evidence_recall")
                        for paper in sorted({r["paper_id"] for r in positive})}}


def main():
    import pdfplumber
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",type=int,default=3)
    parser.add_argument("--out",type=Path,default=ROOT / "eval/real/results.json")
    parser.add_argument("--retrievers",nargs="+",choices=["bm25","dense","hybrid"],default=["bm25","dense","hybrid"])
    args = parser.parse_args()
    if not 1 <= args.k <= 20:
        parser.error("k must be between 1 and 20")
    manifest = ROOT / "eval/real/papers.json"
    qa_path = ROOT / "eval/real/qa.json"
    papers = json.loads(manifest.read_text())
    cases = json.loads(qa_path.read_text())
    embedder = E5Embedder() if set(args.retrievers)-{"bm25"} else None
    rows = {name:[] for name in args.retrievers}
    documents = []
    for item in papers:
        path = ROOT / "data/real-papers" / (item["id"]+".pdf")
        if not path.exists():
            parser.error("Run python scripts/fetch_real_papers.py first")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            parser.error(f"Document hash changed: {item['id']}")
        selected = [c for c in cases if c["paper_id"]==item["id"]]
        # Validate source annotations directly against the PDF, not retrieved chunks.
        with pdfplumber.open(path) as pdf:
            for case in selected:
                for evidence in case["evidence"]:
                    text = pdf.pages[evidence["page"]-1].crop(evidence["bbox"]).extract_text() or ""
                    if compact(evidence["quote"]) not in compact(text):
                        raise ValueError(f"Invalid source annotation: {case['id']}")
        started = time.perf_counter()
        paper = parse_pdf(path)
        chunks = chunk_blocks(paper.blocks)
        preparation = time.perf_counter()-started
        indices = {"bm25":BM25(chunks)}
        embedding_seconds = None
        if embedder:
            started = time.perf_counter()
            dense = DenseRetriever(chunks,embedder)
            embedding_seconds = time.perf_counter()-started
            indices.update(dense=dense,hybrid=HybridRetriever(chunks,dense))
        documents.append({"id":item["id"],"sha256":item["sha256"],"pages":paper.pages,"chunks":len(chunks),
                          "table_chunks":sum(c.table is not None for c in chunks),
                          "parse_seconds":preparation,"embedding_seconds":embedding_seconds,
                          "warnings":list(paper.warnings)})
        for case in selected:
            expected = case["evidence"]
            source_available = (all(any(has_evidence(c,e) for c in chunks) for e in expected)
                                if expected else None)
            bound_available = (all(any(has_bound_cell(c,e) for c in chunks) for e in expected)
                               if expected and "table" in expected[0] else None)
            for name in args.retrievers:
                started = time.perf_counter()
                hits = indices[name].search(case["question"],args.k)
                elapsed = time.perf_counter()-started
                recovered = sum(any(has_evidence(h.chunk,e) for h in hits) for e in expected)
                bound = sum(any(has_bound_cell(h.chunk,e) for h in hits) for e in expected) if bound_available is not None else None
                rows[name].append({"id":case["id"],"paper_id":item["id"],"language":case["language"],
                    "question":case["question"],"answerable":case["answerable"],
                    "kind":expected[0]["kind"] if expected else "unanswerable-probe",
                    "source_available":source_available,"bound_cell_available":bound_available,
                    "evidence_recall":recovered/len(expected) if expected else None,
                    "bound_cell_recall":bound/len(expected) if bound is not None else None,
                    "query_seconds":elapsed,"candidate_count":len(hits),
                    "hits":[{"chunk_id":h.chunk.id,"page":h.chunk.page,"kind":h.chunk.kind,
                             "score":h.score,"score_kind":h.score_kind} for h in hits]})
    result = {"dataset":"PMLR three-paper development corpus; agent-authored annotation draft",
              "run_date_utc":time.strftime("%Y-%m-%d",time.gmtime()),
              "protocol":"Same paper-specific corpus, parser, 1400-character chunks and k for all retrievers. No generation calls.",
              "metrics":{"evidence_recall":"annotated quote plus source-region center present in a top-k chunk",
                         "bound_cell_recall":"correct parsed row label, header, value and source cell region in a top-k chunk"},
              "k":args.k,"annotation_sha256":hashlib.sha256(qa_path.read_bytes()).hexdigest(),
              "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "code_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                             ("paper_evidence/parsing.py","paper_evidence/layout.py","paper_evidence/models.py","paper_evidence/chunking.py","paper_evidence/retrieval.py","paper_evidence/semantic.py","paper_evidence/tables.py","eval/real_eval.py")},
              "python":platform.python_version(),"architecture":platform.machine(),
              "model":embedder.metadata if embedder else None,"documents":documents,
              "retrievers":{name:{"summary":summarize(entries),"rows":entries} for name,entries in rows.items()},
              "limitations":"Development data used while improving the parser, not a held-out benchmark. "
                  "Only three related computer-vision papers from one publisher; 12 facts paired across languages. "
                  "Annotations need independent human review. Evidence recall is not answer accuracy or semantic support. "
                  "Unanswerable probe candidate counts do not measure hallucinations or abstention."}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({name:data["summary"] for name,data in result["retrievers"].items()},indent=2))


if __name__ == "__main__":
    main()

"""Paper-specific evidence retrieval, separating cell parsing from retrieval."""
import argparse
import hashlib
import json
import math
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
from paper_evidence.corpus import load_manifest, verified_pdf


def validate_cases(cases, papers):
    """Fail on silent omissions, duplicate queries and inconsistent labels."""
    if not isinstance(cases,list) or not cases:
        raise ValueError("QA must be a non-empty list")
    paper_ids = {p["id"] for p in papers}
    seen = set()
    for case in cases:
        if (not isinstance(case,dict) or not isinstance(case.get("id"),str)
                or not case["id"] or case["id"] in seen):
            raise ValueError("QA IDs must be unique non-empty strings")
        seen.add(case["id"])
        if case.get("paper_id") not in paper_ids:
            raise ValueError(f"Unknown paper: {case['id']}")
        if "pair_id" in case and (not isinstance(case["pair_id"],str) or not case["pair_id"]):
            raise ValueError(f"Invalid fact-pair ID: {case['id']}")
        if (case.get("language") not in {"en","zh"} or type(case.get("answerable")) is not bool
                or not isinstance(case.get("question"),str) or not case["question"].strip()):
            raise ValueError(f"Invalid query: {case['id']}")
        evidence = case.get("evidence")
        if not isinstance(evidence,list) or bool(evidence) != case["answerable"]:
            raise ValueError(f"Answerability/evidence mismatch: {case['id']}")
        for entry in evidence:
            if (not isinstance(entry,dict) or type(entry.get("page")) is not int or entry["page"] < 1
                    or not isinstance(entry.get("quote"),str) or not entry["quote"].strip()
                    or entry.get("kind") not in {"text","table"}):
                raise ValueError(f"Invalid evidence: {case['id']}")
            box = entry.get("bbox")
            if (not isinstance(box,list) or len(box)!=4
                    or any(type(v) not in {int,float} or not math.isfinite(v) for v in box)
                    or box[0]>=box[2] or box[1]>=box[3]):
                raise ValueError(f"Invalid source region: {case['id']}")
            table = entry.get("table")
            if entry["kind"]=="table" and (not isinstance(table,dict)
                    or any(not isinstance(table.get(k),str) or not table[k].strip()
                           for k in ("row_label","column_label"))):
                raise ValueError(f"Missing table labels: {case['id']}")
        if len({e["kind"] for e in evidence}) > 1:
            raise ValueError(f"Each case must use one evidence kind: {case['id']}")


def check_checkpoint(protocol, root=ROOT):
    for name, expected in protocol.get("code_sha256",{}).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or path.suffix != ".py":
            raise ValueError("Invalid checkpoint source path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen-code checkpoint mismatch: {name}. Keep the original run; use a new labeled protocol for changed code.")


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
                    and compact(reference.get("column_leaf_label",reference["column_label"])) == compact(wanted["column_label"])
                    and ("column_group" not in wanted or compact(wanted["column_group"]) in
                         [compact(g) for g in reference.get("column_groups",[])])
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
            "source_available_queries":sum(r["source_available"] for r in positive),
            "table_queries":len(tables),
            "bound_cell_available_queries":sum(r["bound_cell_available"] for r in tables),
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
    parser.add_argument("--dataset",type=Path,default=ROOT / "eval/real")
    parser.add_argument("--data-dir",type=Path,default=ROOT / "data/real-papers")
    parser.add_argument("--protocol",type=Path,help="Explicit alternate protocol; retain frozen results when tuning")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--retrievers",nargs="+",choices=["bm25","dense","hybrid"],default=["bm25","dense","hybrid"])
    args = parser.parse_args()
    if not 1 <= args.k <= 20:
        parser.error("k must be between 1 and 20")
    if args.protocol and (not args.out or args.out.resolve()==(args.dataset/'results.json').resolve()):
        parser.error("An alternate protocol requires a separate --out file; preserve the original results.json")
    manifest = args.dataset / "papers.json"
    qa_path = args.dataset / "qa.json"
    papers = load_manifest(manifest)
    cases = json.loads(qa_path.read_text())
    validate_cases(cases,papers)
    protocol_path = args.protocol or args.dataset / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    check_checkpoint(protocol)
    embedder = E5Embedder() if set(args.retrievers)-{"bm25"} else None
    rows = {name:[] for name in args.retrievers}
    documents = []
    for item in papers:
        path = args.data_dir / (item["id"]+".pdf")
        if not path.exists():
            parser.error("Fetch the selected corpus with scripts/fetch_real_papers.py --dataset first")
        verified_pdf(path,item["sha256"])
        selected = [c for c in cases if c["paper_id"]==item["id"]]
        # Validate source annotations directly against the PDF, not retrieved chunks.
        with pdfplumber.open(path) as pdf:
            for case in selected:
                for evidence in case["evidence"]:
                    if evidence["page"] > len(pdf.pages):
                        raise ValueError(f"Invalid source page: {case['id']}")
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
    result = {"dataset":protocol.get("dataset","PMLR three-paper development corpus; agent-authored annotation draft"),
              "run_date_utc":time.strftime("%Y-%m-%d",time.gmtime()),
              "protocol":protocol.get("protocol","Same paper-specific corpus, parser, 1400-character chunks and k for all retrievers. No generation calls."),
              "checkpoint_commit":protocol.get("checkpoint_commit"),
              "checkpoint_verified":bool(protocol.get("code_sha256")),
              "protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.exists() else None,
              "counts":{"papers":len(papers),"queries":len(cases),
                        "answerable_facts":len({c.get("pair_id",c["id"]) for c in cases if c["answerable"]})},
              "metrics":{"evidence_recall":"annotated quote plus source-region center present in a top-k chunk",
                         "bound_cell_recall":"correct parsed row label, leaf header, required parent group, value and source cell region in a top-k chunk"},
              "k":args.k,"annotation_sha256":hashlib.sha256(qa_path.read_bytes()).hexdigest(),
              "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "code_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                             ("paper_evidence/parsing.py","paper_evidence/layout.py","paper_evidence/header_tables.py","paper_evidence/models.py","paper_evidence/chunking.py","paper_evidence/retrieval.py","paper_evidence/semantic.py","paper_evidence/tables.py","paper_evidence/answering.py","paper_evidence/corpus.py","eval/real_eval.py")},
              "python":platform.python_version(),"architecture":platform.machine(),
              "model":embedder.metadata if embedder else None,"documents":documents,
              "retrievers":{name:{"summary":summarize(entries),"rows":entries} for name,entries in rows.items()},
              "limitations":protocol.get("limitations","Development data used while improving the parser, not a held-out benchmark. "
                  "Only three related computer-vision papers from one publisher; 12 facts paired across languages. "
                  "Annotations need independent human review. Evidence recall is not answer accuracy or semantic support. "
                  "Unanswerable probe candidate counts do not measure hallucinations or abstention.")}
    args.out = args.out or args.dataset / "results.json"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({name:data["summary"] for name,data in result["retrievers"].items()},indent=2))


if __name__ == "__main__":
    main()

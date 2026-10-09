import argparse
import json
from pathlib import Path

from .answering import OllamaGenerator, answer_question
from .chunking import chunk_blocks
from .parsing import parse_pdf
from .retrieval import BM25
from .storage import load_index, save_index
from .semantic import make_retriever
from .config import load_settings
from .cloud import CompatibleGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(description="PaperEvidence academic PDF evidence assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--out", type=Path, default=Path("data/index.json"))
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=Path("data/index.json"))
    ask.add_argument("--model", help="already downloaded local Ollama model; omit for evidence-only mode")
    ask.add_argument("--provider",choices=["evidence","ollama","compatible"],default="evidence")
    ask.add_argument("--retriever",choices=["bm25","dense","hybrid"],default="bm25")
    ask.add_argument("--embedding-path",type=Path)
    ask.add_argument("--k",type=int,default=5)
    ask.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            paper = parse_pdf(args.pdf)
            chunks = chunk_blocks(paper.blocks)
            if not chunks:
                raise ValueError("没有可检索的文本或表格，请先 OCR。")
            save_index(paper, chunks, args.out)
            print(f"{paper.name}: {paper.pages} pages, {len(chunks)} chunks → {args.out}")
            for warning in paper.warnings:
                print(f"Warning: {warning}")
        else:
            _, chunks = load_index(args.index)
            settings = load_settings()
            if args.provider == "compatible":
                generator = CompatibleGenerator(settings)
            elif args.provider == "ollama" or args.model:
                if not args.model:
                    raise ValueError("Ollama 模式需要 --model。")
                generator = OllamaGenerator(args.model)
            else:
                generator = None
            retriever = make_retriever(chunks,args.retriever,
                model_path=args.embedding_path or settings.embedding_path or None,
                cache_dir=Path(__file__).resolve().parents[1]/".cache"/"embeddings")
            result = answer_question(args.question, retriever, generator, args.k)
            text = json.dumps(result, ensure_ascii=False, indent=2)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text + "\n", encoding="utf-8")
            print(text)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

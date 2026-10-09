"""Explicitly fetch a selected corpus, checking pinned publisher PDF hashes."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_evidence.corpus import load_manifest, download_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,default=ROOT / "eval/real")
    parser.add_argument("--data-dir",type=Path,default=ROOT / "data/real-papers")
    args = parser.parse_args()
    papers = load_manifest(args.dataset / "papers.json")
    directory = args.data_dir
    directory.mkdir(parents=True,exist_ok=True)
    for paper in papers:
        download_paper(paper,directory)
        print(f"Downloaded or already verified: {paper['id']}")


if __name__ == "__main__":
    main()

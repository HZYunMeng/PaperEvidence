"""Explicitly fetch a selected corpus, checking pinned publisher PDF hashes."""
import argparse
import hashlib
import urllib.request
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_evidence.corpus import load_manifest, verified_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,default=ROOT / "eval/real")
    parser.add_argument("--data-dir",type=Path,default=ROOT / "data/real-papers")
    args = parser.parse_args()
    papers = load_manifest(args.dataset / "papers.json")
    directory = args.data_dir
    directory.mkdir(parents=True,exist_ok=True)
    for paper in papers:
        path = directory / (paper["id"] + ".pdf")
        if path.exists():
            try:
                verified_pdf(path,paper["sha256"])
            except ValueError:
                pass
            else:
                print(f"Already verified: {paper['id']}")
                continue
        with urllib.request.urlopen(paper["pdf_url"],timeout=60) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("Publisher redirected to a non-HTTPS URL")
            data = response.read(30*1024*1024+1)
        if len(data)>30*1024*1024 or not data.startswith(b"%PDF-"):
            raise ValueError(f"Unexpected PDF response: {paper['id']}")
        if hashlib.sha256(data).hexdigest() != paper["sha256"]:
            raise ValueError(f"Publisher file changed: {paper['id']}. Review annotations before updating the manifest.")
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(path)
        print(f"Downloaded and verified: {paper['id']}")


if __name__ == "__main__":
    main()

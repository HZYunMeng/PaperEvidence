"""Explicitly fetch the small development corpus from its official publisher."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    papers = json.loads((ROOT / "eval/real/papers.json").read_text())
    directory = ROOT / "data/real-papers"
    directory.mkdir(parents=True,exist_ok=True)
    for paper in papers:
        path = directory / (paper["id"] + ".pdf")
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == paper["sha256"]:
            print(f"Already verified: {paper['id']}")
            continue
        with urllib.request.urlopen(paper["pdf_url"],timeout=60) as response:
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

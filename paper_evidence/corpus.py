"""Explicit local evaluation sources; no downloads during application startup."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse


def load_manifest(path):
    papers = json.loads(Path(path).read_text())
    if not isinstance(papers, list) or not papers:
        raise ValueError("Manifest must be a non-empty paper list")
    seen = set()
    for item in papers:
        if not isinstance(item, dict):
            raise ValueError("Invalid paper entry")
        stem = item.get("id", "")
        if not isinstance(stem, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", stem) or stem in seen:
            raise ValueError("Paper IDs must be unique, safe filename stems")
        seen.add(stem)
        digest = item.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA256: {stem}")
        for key in ("title", "publication", "page_url", "pdf_url"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Missing {key}: {stem}")
        for key in ("page_url", "pdf_url"):
            url = urlparse(item[key])
            if url.scheme != "https" or not url.hostname or url.username or url.password:
                raise ValueError(f"Official source URL must use HTTPS without credentials: {stem}")
    return papers


def verified_pdf(path, expected_sha256):
    data = Path(path).read_bytes()
    if len(data) > 30 * 1024 * 1024 or not data.startswith(b"%PDF-"):
        raise ValueError(f"Invalid or oversized PDF: {Path(path).name}")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Document hash changed: {Path(path).name}")
    return data


def local_examples(root):
    """Return only downloaded, verified papers and visible integrity errors."""
    root = Path(root)
    examples, errors, seen = {}, [], set()
    for manifest in sorted((root / "eval").glob("*/papers.json")):
        try:
            papers = load_manifest(manifest)
        except (ValueError, OSError) as exc:
            errors.append(f"{manifest.parent.name}: {exc}")
            continue
        for item in papers:
            path = root / "data/real-papers" / (item["id"] + ".pdf")
            if not path.exists():
                continue
            if item["id"] in seen:
                errors.append(f"重复论文 ID：{item['id']}")
                continue
            seen.add(item["id"])
            try:
                verified_pdf(path, item["sha256"])
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
                continue
            examples[f"{item['id']} · 真实论文"] = (path, item)
    return examples, errors

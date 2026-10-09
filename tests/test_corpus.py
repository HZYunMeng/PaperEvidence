import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_evidence.corpus import load_manifest, local_examples, verified_pdf
from scripts.fetch_real_papers import main as fetch

ROOT = Path(__file__).resolve().parents[1]


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "eval/custom"
        self.dataset.mkdir(parents=True)
        self.data = (ROOT / "examples/demo-paper.pdf").read_bytes()
        self.item = {"id":"demo","title":"Demo","publication":"Original fixture",
                     "page_url":"https://example.org/paper","pdf_url":"https://example.org/paper.pdf",
                     "sha256":hashlib.sha256(self.data).hexdigest()}
        self.manifest = self.dataset / "papers.json"
        self.manifest.write_text(json.dumps([self.item]))
        self.pdf = self.root / "data/real-papers/demo.pdf"
        self.pdf.parent.mkdir(parents=True)

    def test_manifest_rejects_duplicate_unsafe_ids_and_credential_urls(self):
        for change in ({"id":"../config"},{"sha256":"wrong"},
                       {"pdf_url":"http://example.org/paper.pdf"},
                       {"pdf_url":"https://secret@example.org/paper.pdf"}):
            self.manifest.write_text(json.dumps([{**self.item,**change}]))
            with self.assertRaises(ValueError):
                load_manifest(self.manifest)
        self.manifest.write_text(json.dumps([self.item,self.item]))
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_catalog_only_exposes_local_verified_documents(self):
        self.assertEqual(local_examples(self.root),({},[]))
        self.pdf.write_bytes(self.data)
        examples,errors = local_examples(self.root)
        self.assertEqual(errors,[])
        self.assertEqual(examples["demo · 真实论文"][0],self.pdf)
        self.pdf.write_bytes(self.data+b"tampered")
        examples,errors = local_examples(self.root)
        self.assertEqual(examples,{})
        self.assertIn("hash changed",errors[0])

    def test_non_pdf_is_rejected_even_if_hash_matches(self):
        self.pdf.write_bytes(b"not a PDF")
        with self.assertRaisesRegex(ValueError,"Invalid"):
            verified_pdf(self.pdf,hashlib.sha256(b"not a PDF").hexdigest())

    def test_fetch_hash_failure_preserves_existing_file(self):
        self.pdf.write_bytes(b"existing data")
        response = MagicMock()
        response.__enter__.return_value = response
        response.geturl.return_value = self.item["pdf_url"]
        response.read.return_value = self.data+b"changed"
        argv = ["fetch", "--dataset",str(self.dataset),"--data-dir",str(self.pdf.parent)]
        with patch("sys.argv",argv),patch("urllib.request.urlopen",return_value=response):
            with self.assertRaisesRegex(ValueError,"Publisher file changed"):
                fetch()
        self.assertEqual(self.pdf.read_bytes(),b"existing data")
        self.assertFalse(self.pdf.with_suffix(".download").exists())

    def test_fetch_selected_dataset_and_cache_skip(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.geturl.return_value = self.item["pdf_url"]
        response.read.return_value = self.data
        argv = ["fetch", "--dataset",str(self.dataset),"--data-dir",str(self.pdf.parent)]
        with patch("sys.argv",argv),patch("urllib.request.urlopen",return_value=response) as request:
            fetch()
            fetch()
            request.assert_called_once()
        self.assertEqual(verified_pdf(self.pdf,self.item["sha256"]),self.data)

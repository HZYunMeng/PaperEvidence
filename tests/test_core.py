import json
import tempfile
import unittest
from pathlib import Path

from paper_evidence.answering import answer_question, validate_answer, OllamaGenerator
from paper_evidence.chunking import chunk_blocks
from paper_evidence.models import Block, Chunk
from paper_evidence.parsing import parse_pdf, highlight_page
from paper_evidence.retrieval import BM25, Hit
from paper_evidence.storage import save_index, load_index


ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paper = parse_pdf(ROOT / "examples/demo-paper.pdf")
        cls.chunks = chunk_blocks(cls.paper.blocks)
        cls.index = BM25(cls.chunks)

    def test_real_pdf_table_preserves_rows_and_provenance(self):
        tables = [c for c in self.chunks if c.kind == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("Proposed | 91.2% | 310 ms", tables[0].text)
        self.assertEqual(tables[0].page, 2)
        self.assertIn("3 Experiments", tables[0].section)
        self.assertTrue(tables[0].boxes)

    def test_chunks_never_mix_page_or_section(self):
        blocks = {b.id: b for b in self.paper.blocks}
        for chunk in self.chunks:
            self.assertEqual({blocks[b].page for b in chunk.block_ids}, {chunk.page})
            self.assertEqual({blocks[b].section for b in chunk.block_ids}, {chunk.section})

    def test_empty_query_and_no_hit_behavior(self):
        with self.assertRaises(ValueError):
            answer_question(" ", self.index)
        result = answer_question("xylophonic biodiversity", self.index)
        self.assertTrue(result["abstain"])
        self.assertFalse(result["claims"])

    def test_extract_mode_returns_original_evidence(self):
        result = answer_question("proposed accuracy", self.index)
        self.assertEqual(result["mode"], "extractive")
        self.assertFalse(result["abstain"])
        self.assertTrue(all(c["evidence"][0]["quote_verified"] for c in result["claims"]))

    def test_pdf_source_coordinates_render(self):
        image = highlight_page(ROOT / "examples/demo-paper.pdf", self.chunks[0])
        self.assertTrue(image.startswith(b"\x89PNG"))

    def test_index_round_trip_retains_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"index.json"
            save_index(self.paper, self.chunks, path)
            paper, chunks = load_index(path)
            self.assertEqual(chunks, self.chunks)
            self.assertEqual(paper["id"], self.paper.id)

    def test_scanned_page_does_not_silently_become_text(self):
        from reportlab.pdfgen import canvas
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"blank.pdf"
            pdf = canvas.Canvas(str(path))
            pdf.rect(10,10,40,40)
            pdf.showPage()
            pdf.save()
            paper = parse_pdf(path)
            self.assertFalse(paper.blocks)
            self.assertTrue(any("OCR" in message for message in paper.warnings))


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.chunk = Chunk("real", "paper", 2, "Results", "text",
                           "The proposed method achieves 91.2% accuracy.", ("b",), ((0,0,100,20),))
        self.hits = (Hit(self.chunk, 2.1),)

    def response(self, text="The method achieves 91.2% accuracy.", quote=None, cid="real"):
        return {"abstain":False,"claims":[{"text":text,"evidence":[{
            "chunk_id":cid,"quote":quote or self.chunk.text}]}]}

    def test_unknown_citations_rejected(self):
        with self.assertRaises(ValueError):
            validate_answer(self.response(cid="invented"), self.hits)

    def test_fabricated_quotation_rejected(self):
        with self.assertRaises(ValueError):
            validate_answer(self.response(quote="The model beats all baselines."), self.hits)

    def test_unsupported_number_rejected(self):
        with self.assertRaises(ValueError):
            validate_answer(self.response(text="The method achieves 99.9% accuracy."), self.hits)

    def test_valid_quote_is_only_provenance_verified(self):
        result = validate_answer(self.response(), self.hits)
        self.assertEqual(result["verification"], "quote-provenance-only")
        self.assertEqual(result["claims"][0]["evidence"][0]["page"],2)

    def test_schema_does_not_coerce_abstain_strings(self):
        with self.assertRaises(ValueError):
            validate_answer({"abstain":"false","claims":[]},self.hits)

    def test_invalid_model_output_is_not_shown_as_an_answer(self):
        result = answer_question("accuracy", BM25([self.chunk]),
                                 lambda question,hits:self.response(cid="invented"))
        self.assertTrue(result["abstain"])
        self.assertTrue(result["validation_failed"])
        self.assertFalse(result["claims"])

    def test_external_ollama_endpoints_rejected(self):
        with self.assertRaises(ValueError):
            OllamaGenerator("model", "https://example.com")


if __name__ == "__main__":
    unittest.main()

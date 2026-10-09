import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from eval.real_eval import has_evidence, has_bound_cell, validate_cases, check_checkpoint
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paper=parse_pdf(Path(__file__).resolve().parents[1]/"examples/demo-paper.pdf")
        cls.table=next(c for c in chunk_blocks(paper.blocks) if c.table)
        cls.expected={"page":2,"quote":"91.2%","bbox":list(cls.table.table.rows[2][1].bbox),
                      "table":{"row_label":"Proposed","column_label":"Accuracy"}}

    def test_matching_number_on_wrong_page_or_region_is_not_a_hit(self):
        self.assertTrue(has_evidence(self.table,self.expected))
        for alteration in ({"page":1},{"bbox":[0,0,1,1]},{"quote":"99.2%"}):
            self.assertFalse(has_evidence(self.table,{**self.expected,**alteration}))

    def test_cell_metric_requires_value_method_header_and_region(self):
        self.assertTrue(has_bound_cell(self.table,self.expected))
        for wanted in ({"row_label":"Baseline"},{"column_label":"Latency"},{"row_context":"wrong architecture"}):
            self.assertFalse(has_bound_cell(self.table,{**self.expected,"table":{**self.expected["table"],**wanted}}))
        self.assertFalse(has_bound_cell(self.table,{**self.expected,"bbox":[0,0,1,1]}))

    def test_flat_header_does_not_prove_hierarchical_column_group(self):
        grouped = {**self.expected,"table":{**self.expected["table"],"column_group":"BLEU"}}
        self.assertFalse(has_bound_cell(self.table,grouped))

    def test_annotations_cannot_silently_drop_queries_or_change_answerability(self):
        root = Path(__file__).resolve().parents[1]
        papers = json.loads((root / "eval/nlp/papers.json").read_text())
        cases = json.loads((root / "eval/nlp/qa.json").read_text())
        validate_cases(cases,papers)
        check_checkpoint(json.loads((root / "eval/nlp/protocol.json").read_text()))
        for change in ({"paper_id":"missing"},{"answerable":False},{"language":"other"},
                       {"evidence":[{**cases[0]["evidence"][0],"page":True}]},
                       {"evidence":[{**cases[0]["evidence"][0],"bbox":[0,0,float('nan'),20]}]}):
            changed = [{**cases[0],**change}]+cases[1:]
            with self.assertRaises(ValueError):
                validate_cases(changed,papers)
        with self.assertRaises(ValueError):
            validate_cases(cases+[cases[0]],papers)

    def test_frozen_checkpoint_detects_code_changes_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "parser.py"
            source.write_text("original")
            protocol = {"code_sha256":{"parser.py":hashlib.sha256(source.read_bytes()).hexdigest()}}
            check_checkpoint(protocol,root)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError,"checkpoint mismatch"):
                check_checkpoint(protocol,root)
            with self.assertRaisesRegex(ValueError,"source path"):
                check_checkpoint({"code_sha256":{"../outside.py":"hash"}},root)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from eval.real_eval import has_evidence, has_bound_cell
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


if __name__ == "__main__":
    unittest.main()

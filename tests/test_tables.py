import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from paper_evidence.answering import messages_for, validate_answer
from paper_evidence.chunking import chunk_blocks
from paper_evidence.models import TableCell, TableData
from paper_evidence.parsing import highlight_page, parse_pdf, structured_table
from paper_evidence.retrieval import Hit
from paper_evidence.storage import load_index, save_index
from paper_evidence.tables import cell_reference

ROOT = Path(__file__).resolve().parents[1]


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paper = parse_pdf(ROOT / "examples/demo-paper.pdf")
        cls.chunks = chunk_blocks(cls.paper.blocks)
        cls.table = next(c for c in cls.chunks if c.table)
        cls.hits = [Hit(cls.table,1)]

    def raw(self,row=2,column=1):
        return {"abstain":False,"claims":[{"table_value":{
            "chunk_id":self.table.id,"row":row,"column":column}}]}

    def test_value_is_read_from_selected_row_not_from_model(self):
        result = validate_answer(self.raw(),self.hits)
        self.assertEqual(result["claims"][0]["text"],"Proposed · Accuracy：91.2%")
        reference = result["claims"][0]["evidence"][0]
        self.assertEqual((reference["row_label"],reference["column_label"],reference["value"]),
                         ("Proposed","Accuracy","91.2%"))
        self.assertEqual(len(reference["boxes"]),3)
        self.assertEqual(validate_answer(self.raw(row=1),self.hits)["claims"][0]["text"],"Baseline · Accuracy：82.4%")

    def test_model_cannot_override_source_value_or_labels(self):
        for key,value in [("value","99.2%"),("row_label","Baseline")]:
            raw = self.raw()
            raw["claims"][0]["table_value"][key]=value
            with self.assertRaises(ValueError):
                validate_answer(raw,self.hits)

    def test_free_form_claim_cannot_disguise_selected_row(self):
        raw = self.raw()
        raw["claims"][0]["text"]="Baseline achieved 91.2%"
        with self.assertRaises(ValueError):
            validate_answer(raw,self.hits)

    def test_citing_whole_table_does_not_validate_numeric_claim(self):
        raw = {"abstain":False,"claims":[{"text":"Baseline achieved 91.2%", "evidence":[{
            "chunk_id":self.table.id,"quote":self.table.text}]}]}
        with self.assertRaisesRegex(ValueError,"table_value"):
            validate_answer(raw,self.hits)

    def test_invalid_indices_and_unknown_table_rejected(self):
        for row,column in [(-1,1),(0,1),(2,0),(3,1),(2,99),(True,1),(2,"1")]:
            with self.assertRaises(ValueError):
                validate_answer(self.raw(row,column),self.hits)
        raw = self.raw()
        raw["claims"][0]["table_value"]["chunk_id"]="invented"
        with self.assertRaises(ValueError):
            validate_answer(raw,self.hits)

    def test_multiline_or_merged_cell_is_not_given_a_single_value(self):
        for cell in [TableCell("91.2%\n82.4%",(0,0,1,1)),TableCell("91.2%",None)]:
            rows = list(self.table.table.rows)
            rows[2] = (rows[2][0],cell,rows[2][2])
            source = replace(self.table,table=TableData(tuple(rows)))
            with self.assertRaises(ValueError):
                cell_reference(source,2,1)

    def test_cell_boxes_are_inside_table_and_render(self):
        reference = cell_reference(self.table,2,1)
        outer = self.table.boxes[0]
        for box in reference["boxes"]:
            self.assertTrue(outer[0]<=box[0]<box[2]<=outer[2])
            self.assertTrue(outer[1]<=box[1]<box[3]<=outer[3])
        png = highlight_page(ROOT / "examples/demo-paper.pdf",self.table,selected_boxes=reference["boxes"])
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_prompt_contains_table_coordinates_without_pdf_paths(self):
        evidence = json.loads(messages_for("question",self.hits)[1]["content"])["evidence"][0]
        self.assertEqual(evidence["table_rows"][2][1],"91.2%")
        self.assertNotIn("bbox",evidence)

    def test_legacy_index_loads_without_invented_cell_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"index.json"
            save_index(self.paper,self.chunks,path)
            data=json.loads(path.read_text())
            self.assertEqual(data["schema_version"],2)
            data["schema_version"]=1
            for c in data["chunks"]:
                c.pop("table")
            path.write_text(json.dumps(data))
            _,chunks=load_index(path)
            self.assertTrue(all(c.table is None for c in chunks))
            with self.assertRaises(ValueError):
                cell_reference(next(c for c in chunks if c.kind=="table"),2,1)

    def test_sparse_chart_grid_leaves_text_out_of_table_candidate(self):
        class Sparse:
            def extract(self):
                return [["60","","",""],["","","",""],["80","","","graph"]]
        self.assertIsNone(structured_table(Sparse()))


if __name__ == "__main__":
    unittest.main()

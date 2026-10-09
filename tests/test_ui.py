import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class UITests(unittest.TestCase):
    def test_evidence_browse_and_no_match(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app), default_timeout=30).run()
        self.assertFalse(at.exception)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        result = at.session_state["result"]
        self.assertEqual(result["mode"], "extractive")
        self.assertFalse(result["abstain"])
        self.assertEqual(len(at.table), 1)
        next(field for field in at.text_input if field.label == "向论文提问").set_value("xylophonic biodiversity")
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.session_state["result"]["abstain"])

    def test_changing_retrieval_mode_clears_previous_answer(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app), default_timeout=30).run()
        at.button[0].click().run()
        self.assertIn("result",at.session_state)
        next(field for field in at.selectbox if field.label == "检索方式").set_value("多语言语义").run()
        self.assertFalse(at.exception)
        self.assertNotIn("result",at.session_state)

    def test_table_inspector_reads_selected_method_and_metric(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app), default_timeout=30).run()
        at.button[0].click().run()
        cid = next(h["chunk_id"] for h in at.session_state["result"]["hits"] if h["kind"]=="table")
        next(field for field in at.selectbox if field.label=="选择证据片段").set_value(cid).run()
        next(field for field in at.checkbox if field.label=="定位表格单元格").set_value(True).run()
        next(field for field in at.selectbox if field.label=="数据行").set_value(2).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.success[0].value,"Proposed · Accuracy：91.2%")

    def test_parser_revision_clears_stale_answer(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app), default_timeout=30).run()
        at.button[0].click().run()
        self.assertIn('result',at.session_state)
        options=at.session_state['request_options']
        at.session_state['request_options']=options[:-1]+('0.3.0',)
        at.run()
        self.assertNotIn('result',at.session_state)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest
from unittest.mock import patch
from paper_evidence.models import Paper


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

    def test_query_free_browsing_can_inspect_table_and_return_to_full_page(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app),default_timeout=30).run()
        next(f for f in at.checkbox if f.label=="打开逐页浏览").set_value(True).run()
        next(f for f in at.selectbox if f.label=="PDF 页码").set_value(2).run()
        next(f for f in at.radio if f.label=="内容筛选").set_value("表格片段").run()
        field = next(f for f in at.selectbox if f.label=="浏览片段")
        self.assertEqual(len(field.options),2)
        field.select_index(1).run()
        next(f for f in at.checkbox if f.label=="定位表格单元格").set_value(True).run()
        next(f for f in at.selectbox if f.label=="数据行").set_value(2).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.success[0].value,"Proposed · Accuracy：91.2%")
        self.assertNotIn("result",at.session_state)
        next(f for f in at.selectbox if f.label=="浏览片段").select_index(0).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.success)

    def test_browser_and_search_inspectors_can_coexist(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app),default_timeout=30).run()
        next(f for f in at.checkbox if f.label=="打开逐页浏览").set_value(True).run()
        next(f for f in at.selectbox if f.label=="PDF 页码").set_value(2).run()
        next(f for f in at.radio if f.label=="内容筛选").set_value("表格片段").run()
        next(f for f in at.selectbox if f.label=="浏览片段").select_index(1).run()
        at.button[0].click().run()
        cid = next(h["chunk_id"] for h in at.session_state["result"]["hits"] if h["kind"]=="table")
        next(f for f in at.selectbox if f.label=="选择证据片段").set_value(cid).run()
        inspectors = [f for f in at.checkbox if f.label=="定位表格单元格"]
        self.assertEqual(len(inspectors),2)
        inspectors[0].set_value(True).run()
        next(f for f in at.checkbox if f.label=="定位表格单元格" and not f.value).set_value(True).run()
        self.assertFalse(at.exception)

    def test_no_indexable_text_still_allows_pdf_browsing(self):
        import streamlit as st
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        app = Path(__file__).resolve().parents[1] / "app.py"
        empty = Paper("empty-fixture","demo-paper.pdf",2,(),("No indexable text",))
        with patch("paper_evidence.parsing.parse_pdf",return_value=empty):
            at = AppTest.from_file(str(app),default_timeout=30).run()
            next(f for f in at.checkbox if f.label=="打开逐页浏览").set_value(True).run()
            self.assertFalse(at.exception)
            self.assertEqual(next(f for f in at.selectbox if f.label=="浏览片段").options,["整页 PDF 原文"])
            self.assertFalse(any(f.label=="向论文提问" for f in at.text_input))
        st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()

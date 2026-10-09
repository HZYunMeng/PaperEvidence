import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.test_ui import fixture_app
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf
from tests.test_header_tables import grouped_pdf
from paper_evidence.catalog import table_catalog


def field(at,label):
    return next(f for f in at.selectbox if f.label==label)


class CatalogUITests(unittest.TestCase):
    def test_catalog_without_structures_leaves_page_browsing_available(self):
        at=fixture_app()
        with patch('paper_evidence.catalog_view.table_catalog',return_value=()):
            next(f for f in at.checkbox if f.label=='打开表格目录').set_value(True).run()
        self.assertFalse(at.exception)
        self.assertTrue(any('暂无结构化表格' in i.value for i in at.info))
        self.assertTrue(any(f.label=='打开逐页浏览' for f in at.checkbox))

    def test_unbindable_table_preview_failure_keeps_text_and_no_cell_selection(self):
        at=self.open_catalog()
        chunk=next(c for c in chunk_blocks(parse_pdf(grouped_pdf()).blocks) if c.table)
        rows=tuple(tuple(replace(cell,bbox=None) for cell in row) for row in chunk.table.rows)
        chunks=[replace(chunk,table=replace(chunk.table,rows=rows))]
        with patch('paper_evidence.catalog_view.table_catalog',return_value=table_catalog(chunks)), \
             patch('paper_evidence.catalog_view.show_page',side_effect=ValueError('preview unavailable')):
            at.run()
            field(at,'选择表格').select_index(1).run()
        self.assertFalse(at.exception)
        self.assertTrue(any('没有可可靠绑定' in w.value for w in at.warning))
        self.assertTrue(any('PDF 页面预览不可用' in w.value for w in at.warning))
        self.assertFalse(any(f.label=='选择原表数据行' for f in at.selectbox))
        self.assertTrue(any('Encoder' in t.value.to_string() for t in at.table))

    def open_catalog(self):
        at=fixture_app()
        next(f for f in at.checkbox if f.label=='打开表格目录').set_value(True).run()
        self.assertFalse(at.exception)
        return at

    def test_explicit_source_selection_requires_row_and_metric(self):
        at=self.open_catalog()
        self.assertEqual(field(at,'选择表格').value,None)
        field(at,'选择表格').select_index(1).run()
        self.assertFalse(at.metric)
        field(at,'选择原表数据行').select_index(2).run()
        self.assertTrue(next(b for b in at.button if b.label=='核对这个单元格').disabled)
        field(at,'选择原表指标').select_index(1).run()
        with patch('paper_evidence.cloud.CompatibleGenerator.__call__') as cloud:
            next(b for b in at.button if b.label=='核对这个单元格').click().run()
            cloud.assert_not_called()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value,'91.2%')
        self.assertNotIn('result',at.session_state)
        self.assertEqual(at.session_state['cell_result']['matches'][0]['row_label'],'Proposed')
        self.assertEqual(next(f for f in at.text_input if f.label=='方法或模型名').value,'Proposed')

    def test_catalog_selection_switches_source_in_the_same_table(self):
        at=self.open_catalog()
        field(at,'选择表格').select_index(1).run()
        for i,expected in [(2,'91.2%'),(1,'82.4%')]:
            field(at,'选择原表数据行').select_index(i).run()
            field(at,'选择原表指标').select_index(1).run()
            next(b for b in at.button if b.label=='核对这个单元格').click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.metric[0].value,expected)
            self.assertTrue(any(expected in s.value for s in at.success))
        self.assertTrue(any(f.label=='导出同行字段与坐标' for f in at.get('download_button')))

    def test_no_match_still_exposes_label_discovery(self):
        at=fixture_app()
        next(f for f in at.text_input if f.label=='方法或模型名').set_value('Unknown')
        next(f for f in at.text_input if f.label=='指标名或完整分组路径').set_value('Unknown')
        next(b for b in at.button if b.label=='查找数值').click().run()
        self.assertTrue(any('表格目录' in w.value for w in at.warning))
        next(f for f in at.checkbox if f.label=='打开表格目录').set_value(True).run()
        self.assertGreater(len(field(at,'选择表格').options),1)
        self.assertFalse(at.exception)

    def test_catalog_keeps_grouped_metric_choices_distinct(self):
        at=fixture_app()
        chunks=chunk_blocks(parse_pdf(grouped_pdf()).blocks,max_chars=240)
        with patch('paper_evidence.chunking.chunk_blocks',return_value=chunks):
            # Clear only test parsing cache so the synthetic structure is used.
            import streamlit as st
            st.cache_data.clear()
            self.addCleanup(st.cache_data.clear)
            next(f for f in at.checkbox if f.label=='打开表格目录').set_value(True).run()
            field(at,'选择表格').select_index(1).run()
            field(at,'选择原表数据行').select_index(1).run()
            options=field(at,'选择原表指标').options
            self.assertEqual(options[1:],[f'第 {i+2} 列 · {name}' for i,name in enumerate(
                ['Quality / Dev','Quality / Test','Expense / Dev','Expense / Test'])])
            field(at,'选择原表指标').select_index(3).run()
            next(b for b in at.button if b.label=='核对这个单元格').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].label,'Expense / Dev')
        self.assertEqual(len(at.session_state['cell_result']['matches'][0]['boxes']),4)

import hashlib
import shutil
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from tests.test_ui import fixture_app
from paper_evidence.collection_view import KEY
from paper_evidence.collection import make_evidence
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf

ROOT=Path(__file__).resolve().parents[1]


def button(at,label):
    return next(b for b in at.button if b.label==label)


def lookup(at,method='Proposed'):
    next(f for f in at.text_input if f.label=='方法或模型名').set_value(method)
    next(f for f in at.text_input if f.label=='指标名或完整分组路径').set_value('Accuracy')
    button(at,'查找数值').click().run()


class CollectionUITests(unittest.TestCase):
    def test_collection_remains_exportable_when_current_pdf_is_missing(self):
        from streamlit.testing.v1 import AppTest
        source=(ROOT/'examples/demo-paper.pdf').read_bytes()
        chunk=next(c for c in chunk_blocks(parse_pdf(source).blocks) if c.table)
        item=make_evidence(source,chunk,2,1,{'name':'demo-paper.pdf'})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'eval/real').mkdir(parents=True)
            shutil.copyfile(ROOT/'app.py',root/'app.py')
            shutil.copyfile(ROOT/'eval/real/papers.json',root/'eval/real/papers.json')
            at=AppTest.from_file(str(root/'app.py'),default_timeout=30)
            at.session_state[KEY]=[item]
            with patch('urllib.request.urlopen') as request:
                at.run()
                next(f for f in at.selectbox if f.label=='查看已保存证据').select_index(1).run()
                request.assert_not_called()
            self.assertFalse(at.exception)
            self.assertTrue(any(d.label=='导出证据清单 JSON' for d in at.get('download_button')))
            self.assertTrue(any('清单仍可导出' in i.value for i in at.info))

    def test_collection_requires_explicit_action_and_does_not_call_model(self):
        at=fixture_app();lookup(at)
        self.assertNotIn(KEY,at.session_state)
        with patch('paper_evidence.cloud.CompatibleGenerator.__call__') as cloud:
            button(at,'加入证据清单').click().run()
            cloud.assert_not_called()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state[KEY]),1)
        self.assertEqual(at.session_state[KEY][0]['reference']['value'],'91.2%')
        self.assertTrue(button(at,'已在证据清单中').disabled)
        self.assertTrue(any(d.label=='导出证据清单 Markdown' for d in at.get('download_button')))

    def test_multiple_methods_removal_and_readding_preserve_other_records(self):
        at=fixture_app()
        for method in ('Proposed','Baseline'):
            lookup(at,method);button(at,'加入证据清单').click().run()
        self.assertEqual(len(at.session_state[KEY]),2)
        next(f for f in at.selectbox if f.label=='查看已保存证据').select_index(1).run()
        button(at,'移除这条证据').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state[KEY]),1)
        self.assertEqual(at.session_state[KEY][0]['reference']['row_label'],'Baseline')
        lookup(at,'Proposed');button(at,'加入证据清单').click().run()
        self.assertEqual(len(at.session_state[KEY]),2)

    def test_saved_geometry_preview_requires_matching_full_pdf_hash(self):
        at=fixture_app();lookup(at);button(at,'加入证据清单').click().run()
        next(f for f in at.selectbox if f.label=='查看已保存证据').select_index(1).run()
        next(f for f in at.checkbox if f.label=='显示已保存证据的 PDF 来源').set_value(True).run()
        self.assertFalse(at.exception)
        source=(ROOT/'examples/demo-paper.pdf').read_bytes()+b'\n'
        chunk=next(c for c in chunk_blocks(parse_pdf(source).blocks) if c.table)
        at.session_state[KEY]=[make_evidence(source,chunk,2,1,{'name':'demo-paper.pdf'})]
        at.run()
        next(f for f in at.selectbox if f.label=='查看已保存证据').select_index(1).run()
        self.assertTrue(any('文件哈希不同' in i.value for i in at.info))
        self.assertFalse(any(f.label=='显示已保存证据的 PDF 来源' for f in at.checkbox))

    def test_switching_documents_keeps_collection_and_rejects_wrong_pdf_preview(self):
        at=fixture_app();lookup(at);button(at,'加入证据清单').click().run()
        saved=at.session_state[KEY][0]
        from reportlab.pdfgen import canvas
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'independent.pdf'
            c=canvas.Canvas(str(path));c.drawString(70,700,'Independent source document');c.save()
            item={'id':'other','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'title':'Unrelated local fixture','publication':'Local test','page_url':'https://example.org/test'}
            with patch('paper_evidence.corpus.local_examples',return_value=({'Other PDF':(path,item)},[])):
                at.run()
                next(f for f in at.selectbox if f.label=='示例论文').set_value('Other PDF').run()
                next(f for f in at.selectbox if f.label=='查看已保存证据').select_index(1).run()
                self.assertTrue(any('文件哈希不同' in i.value for i in at.info))
                self.assertNotIn('cell_result',at.session_state)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state[KEY][0],saved)

    def test_browser_inspector_can_collect_without_freeform_query(self):
        at=fixture_app()
        next(f for f in at.checkbox if f.label=='打开逐页浏览').set_value(True).run()
        next(f for f in at.selectbox if f.label=='PDF 页码').set_value(2).run()
        next(f for f in at.radio if f.label=='内容筛选').set_value('表格片段').run()
        next(f for f in at.selectbox if f.label=='浏览片段').select_index(1).run()
        next(f for f in at.checkbox if f.label=='定位表格单元格').set_value(True).run()
        button(at,'加入证据清单').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state[KEY]),1)
        self.assertNotIn('result',at.session_state)

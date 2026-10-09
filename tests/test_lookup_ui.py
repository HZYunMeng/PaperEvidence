import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest
from tests.test_ui import fixture_app
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf
from paper_evidence.tables import cell_reference

ROOT=Path(__file__).resolve().parents[1]


class LookupUITests(unittest.TestCase):
    def test_new_lookup_updates_highlight_and_citation_in_the_same_table(self):
        at=fixture_app()
        next(f for f in at.text_input if f.label=='指标名或完整分组路径').set_value('Accuracy')
        for method in ('Proposed','Baseline'):
            next(f for f in at.text_input if f.label=='方法或模型名').set_value(method)
            next(b for b in at.button if b.label=='查找数值').click().run()
            self.assertFalse(at.exception)
            ref=at.session_state['cell_result']['matches'][0]
            self.assertTrue(any(f'{method} · Accuracy' in s.value for s in at.success))
            self.assertEqual(at.metric[0].value,ref['value'])
            self.assertTrue(next(f for f in at.selectbox if f.label=='数据行').disabled)

    def test_lookup_reads_source_value_without_model_or_freeform_retrieval(self):
        at=fixture_app()
        next(f for f in at.text_input if f.label=='方法或模型名').set_value('Proposed')
        next(f for f in at.text_input if f.label=='指标名或完整分组路径').set_value('Accuracy')
        with patch('paper_evidence.cloud.CompatibleGenerator.__call__') as cloud:
            next(b for b in at.button if b.label=='查找数值').click().run()
            cloud.assert_not_called()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value,'91.2%')
        self.assertNotIn('result',at.session_state)
        ref=at.session_state['cell_result']['matches'][0]
        self.assertEqual((ref['row_label'],ref['column_label'],ref['page']),('Proposed','Accuracy',2))

    def test_no_match_does_not_claim_paper_has_no_answer(self):
        at=fixture_app()
        next(f for f in at.text_input if f.label=='方法或模型名').set_value('Unknown method')
        next(f for f in at.text_input if f.label=='指标名或完整分组路径').set_value('Accuracy')
        next(b for b in at.button if b.label=='查找数值').click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.metric)
        self.assertTrue(any('不表示论文没有该数据' in w.value for w in at.warning))

    def test_multiple_sources_are_not_automatically_selected(self):
        at=fixture_app()
        table=next(c for c in chunk_blocks(parse_pdf(ROOT/'examples/demo-paper.pdf').blocks) if c.table)
        refs=[cell_reference(table,r,1) for r in (1,2)]
        next(f for f in at.text_input if f.label=='方法或模型名').set_value('Proposed')
        next(f for f in at.text_input if f.label=='指标名或完整分组路径').set_value('Accuracy')
        with patch('paper_evidence.tables.find_cells',return_value=refs):
            next(b for b in at.button if b.label=='查找数值').click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.metric)
        next(f for f in at.selectbox if f.label=='选择数值来源').set_value(1).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value,'91.2%')

    def test_first_launch_download_is_explicit_and_verified(self):
        # Isolated root with no downloaded PDFs: startup must not use the network.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'eval/real').mkdir(parents=True)
            shutil.copyfile(ROOT/'app.py',root/'app.py')
            (root/'eval/real/papers.json').write_bytes((ROOT/'eval/real/papers.json').read_bytes())
            at=AppTest.from_file(str(root/'app.py'),default_timeout=30)
            with patch('urllib.request.urlopen') as request:
                at.run()
                request.assert_not_called()
            self.assertFalse(at.exception)
            self.assertEqual(next(f for f in at.selectbox if f.label=='示例论文').value,'clip · 真实论文')
            self.assertTrue(any(b.label=='下载真实论文并开始' for b in at.button))
            self.assertFalse(any(f.label=='方法或模型名' for f in at.text_input))
            response=MagicMock();response.__enter__.return_value=response
            response.geturl.return_value='https://proceedings.mlr.press/test.pdf'
            response.read.return_value=b'%PDF-unverified'
            with patch('urllib.request.urlopen',return_value=response) as request:
                next(b for b in at.button if b.label=='下载真实论文并开始').click().run()
                request.assert_called_once()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertFalse((root/'data/real-papers/clip.pdf').exists())
            broken=root/'data/real-papers/clip.pdf';broken.parent.mkdir(parents=True)
            broken.write_bytes(b'%PDF-corrupted')
            with patch('urllib.request.urlopen') as request:
                at.run();request.assert_not_called()
            self.assertFalse(at.exception)
            self.assertTrue(any(b.label=='下载真实论文并开始' for b in at.button))
            self.assertEqual(broken.read_bytes(),b'%PDF-corrupted')

    def test_schema2_without_header_paths_remains_compatible(self):
        from paper_evidence.storage import save_index,load_index
        paper=parse_pdf(ROOT/'examples/demo-paper.pdf');chunks=chunk_blocks(paper.blocks)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old-index.json';save_index(paper,chunks,path)
            data=json.loads(path.read_text());data['schema_version']=2
            for entry in data['chunks']:
                if entry['table']:
                    entry['table'].pop('header_paths');entry['table'].pop('header_notes')
            path.write_text(json.dumps(data))
            _,loaded=load_index(path)
        table=next(c for c in loaded if c.table)
        self.assertEqual(cell_reference(table,2,1)['value'],'91.2%')
        self.assertEqual(cell_reference(table,2,1)['column_groups'],[])

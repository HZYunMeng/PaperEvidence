from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

from paper_evidence.collection import make_evidence, add_evidence, export_json, export_markdown
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf
from paper_evidence.models import TableCell

ROOT=Path(__file__).resolve().parents[1]


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(ROOT/'examples/demo-paper.pdf').read_bytes()
        cls.chunk=next(c for c in chunk_blocks(parse_pdf(cls.source).blocks) if c.table)

    def evidence(self, **document):
        return make_evidence(self.source,self.chunk,2,1,document)

    def test_snapshot_preserves_full_file_hash_source_cells_and_raw_fields(self):
        item=self.evidence(name='/private/user/paper.pdf',title='Original title')
        self.assertEqual(item['document']['sha256'],hashlib.sha256(self.source).hexdigest())
        self.assertEqual(item['document']['name'],'paper.pdf')
        self.assertEqual(item['reference']['value'],'91.2%')
        self.assertEqual(item['row_source']['fields'][1]['text'],'91.2%')
        self.assertEqual(len(item['reference']['boxes']),3)

    def test_wrong_pdf_and_unbindable_cells_are_rejected(self):
        with self.assertRaises(ValueError):make_evidence(self.source+b'\n',self.chunk,2,1)
        rows=list(self.chunk.table.rows)
        rows[2]=(rows[2][0],TableCell('91.2%',None),rows[2][2])
        bad=replace(self.chunk,table=replace(self.chunk.table,rows=tuple(rows)))
        with self.assertRaises(ValueError):make_evidence(self.source,bad,2,1)

    def test_same_physical_cell_deduplicates_across_chunk_ids_and_file_renaming(self):
        first=self.evidence(name='one.pdf')
        other=make_evidence(self.source,replace(self.chunk,id='another-fragment'),2,1,{'name':'renamed.pdf'})
        items,added=add_evidence([],first)
        self.assertTrue(added)
        items,added=add_evidence(items,other)
        self.assertFalse(added)
        self.assertEqual(len(items),1)
        self.assertEqual(items[0]['document']['name'],'one.pdf')

    def test_different_cells_and_file_revisions_remain_separate(self):
        first=self.evidence(name='same.pdf')
        baseline=make_evidence(self.source,self.chunk,1,1,{'name':'same.pdf'})
        revised=self.source+b'\n'
        chunk=next(c for c in chunk_blocks(parse_pdf(revised).blocks) if c.table)
        second=make_evidence(revised,chunk,2,1,{'name':'same.pdf'})
        self.assertEqual(len({i['id'] for i in (first,baseline,second)}),3)

    def test_collection_does_not_share_mutable_caller_snapshots(self):
        item=self.evidence()
        items,_=add_evidence([],item)
        item['reference']['value']='tampered'
        self.assertEqual(items[0]['reference']['value'],'91.2%')
        self.assertNotIn('tampered',export_json(items))

    def test_json_and_markdown_retain_source_provenance_without_pdf_bytes(self):
        item=self.evidence(name='paper.pdf')
        exported=json.loads(export_json([item]))
        self.assertEqual(exported['schema_version'],1)
        self.assertEqual(exported['items'],[item])
        text=export_markdown([item])
        for expected in ('91.2%','Proposed','Accuracy','PDF page',item['document']['sha256']):
            self.assertIn(expected,text)
        self.assertNotIn('%PDF-',text)
        self.assertIn('comparability require manual review',text)

    def test_pdf_labels_cannot_inject_markdown_rows_or_html(self):
        item=deepcopy(self.evidence(name='paper.pdf'))
        item['reference']['row_label']='A | B\n<script>run()</script> [link](bad)'
        text=export_markdown([item])
        self.assertNotIn('<script>',text)
        self.assertIn('A \\| B &lt;script&gt;',text)
        self.assertEqual(sum(line.startswith('| E') for line in text.splitlines()),1)

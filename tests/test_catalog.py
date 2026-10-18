import unittest
from dataclasses import replace

from paper_evidence.catalog import table_catalog, catalog_rows, row_source, row_label, match_label
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf
from paper_evidence.models import TableData, TableCell
from tests.test_header_tables import grouped_pdf


class CatalogTests(unittest.TestCase):
    def test_row_fragments_join_only_by_original_block_identity(self):
        chunks=chunk_blocks(parse_pdf(grouped_pdf()).blocks,max_chars=240)
        tables=table_catalog(chunks)
        self.assertEqual(len(tables),1)
        self.assertGreater(len(tables[0].chunks),1)
        self.assertEqual(len(catalog_rows(tables[0])),3)
        original=tables[0].chunks[0]
        unrelated=replace(original,id='unrelated',block_ids=('another-table',))
        self.assertEqual(len(table_catalog((*chunks,unrelated))),2)
        other_page=replace(original,page=2)
        self.assertEqual(len(table_catalog((*chunks,other_page))),2)

    def test_catalog_keeps_distinct_grouped_columns_and_skips_spans(self):
        chunks=chunk_blocks(parse_pdf(grouped_pdf(merged=True)).blocks)
        rows=catalog_rows(table_catalog(chunks)[0])
        self.assertEqual([r['column_label'] for r in rows[0]['references']],
                         ['Quality / Dev','Quality / Test','Expense / Dev','Expense / Test'])
        self.assertEqual([r['column'] for r in rows[2]['references']],[1,2])
        self.assertIn('shared cost',row_label(rows[2]['source']))
        self.assertIsNone(rows[2]['source']['fields'][3]['bbox'])

    def test_numeric_same_row_fields_distinguish_duplicate_methods_across_fragments(self):
        chunk=next(c for c in chunk_blocks(parse_pdf(grouped_pdf()).blocks) if c.table)
        header=(TableCell('Method',(70,50,110,60)),TableCell('Batch',(200,50,230,60)),TableCell('Score',(300,50,330,60)))
        rows=[]
        for i,batch in enumerate(('32','64')):
            y=100+i*20
            row=(TableCell('Encoder',(70,y,110,y+10)),TableCell(batch,(200,y,230,y+10)),TableCell('17.4',(300,y,330,y+10)))
            rows.append(replace(chunk,id=f'part{i}',block_ids=(f'original:rows{i}',),table=TableData((header,row),caption='Table 4. Shared method')))
        entries=table_catalog(rows)
        items=catalog_rows(entries[0])
        self.assertEqual(len(items),2)
        labels=[match_label(r['references'][-1],r['chunk']) for r in items]
        self.assertNotEqual(*labels)
        self.assertIn('Batch=32',labels[0]);self.assertIn('Batch=64',labels[1])
        self.assertEqual(items[0]['source']['fields'][1]['bbox'],[200,100,230,110])

    def test_unbindable_table_remains_discoverable_without_fabricated_cells(self):
        chunk=next(c for c in chunk_blocks(parse_pdf(grouped_pdf()).blocks) if c.table)
        rows=tuple(tuple(replace(c,bbox=None) for c in row) for row in chunk.table.rows)
        table=replace(chunk,table=replace(chunk.table,rows=rows))
        entries=table_catalog([table])
        self.assertEqual(len(entries),1)
        self.assertFalse(catalog_rows(entries[0]))
        self.assertEqual(entries[0].chunks[0].text,chunk.text)

    def test_group_text_is_exported_without_invented_geometry(self):
        chunk=next(c for c in chunk_blocks(parse_pdf(grouped_pdf()).blocks) if c.table)
        group=(TableCell('Separate validation setting',None),)+tuple(TableCell('',None) for _ in range(4))
        table=replace(chunk,table=replace(chunk.table,rows=(chunk.table.rows[0],group,chunk.table.rows[1])))
        source=row_source(table,2)
        self.assertEqual(source['preceding_group_text'],'Separate validation setting')
        self.assertIn('no experimental-condition interpretation',source['scope'])


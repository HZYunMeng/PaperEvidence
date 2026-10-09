import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from paper_evidence.chunking import chunk_blocks
from paper_evidence.models import TableCell
from paper_evidence.parsing import parse_pdf, highlight_page
from paper_evidence.storage import save_index, load_index
from paper_evidence.tables import cell_reference, find_cells
from eval.real_eval import has_bound_cell


def grouped_pdf(*, rules=True, merged=False):
    from reportlab.pdfgen import canvas
    out=io.BytesIO(); c=canvas.Canvas(out,pagesize=(612,792))
    c.setFont('Helvetica',9)
    c.drawString(70,730,'Table 14. Separate quality and expense measurements')
    for y in (711,663,583):c.line(70,y,545,y)
    c.drawString(80,683,'Method')
    c.drawString(313,697,'Quality')
    c.drawString(455,697,'Expense')
    if rules:
        c.line(267,686,399,686);c.line(414,686,540,686)
    for x,t in [(280,'Dev'),(353,'Test'),(429,'Dev'),(506,'Test')]:c.drawString(x,671,t)
    for i,y in enumerate((647,627,607)):
        c.setFont('Helvetica',9);c.drawString(80,y,'Encoder')
        c.setFont('Helvetica',6);c.drawString(113,y-3,f'V{i}')
        c.setFont('Helvetica',9)
        for x,t in [(280,str(17.4+i)),(353,str(28.6+i))]:c.drawString(x,y,t)
        if merged and i==2:
            c.drawString(460,y,'shared cost')
        else:
            for x,t in [(429,str(17.4+i)),(506,str(7.1+i))]:c.drawString(x,y,t)
    c.save();return out.getvalue()


class HeaderTableTests(unittest.TestCase):
    def test_unnamed_row_header_does_not_shift_numeric_columns(self):
        from reportlab.pdfgen import canvas
        out=io.BytesIO();c=canvas.Canvas(out,pagesize=(612,792));c.setFont('Helvetica',9)
        c.drawString(70,730,'Table 15. Measurements with an unnamed method column')
        for y in (711,682,620):c.line(70,y,520,y)
        for x,t in [(305,'Score'),(430,'Cost')]:c.drawString(x,696,t)
        for y,t in [(661,'Alpha'),(641,'Beta')]:
            for x,v in [(80,t),(305,'12.3'),(430,'45.6')]:c.drawString(x,y,v)
        c.save()
        table=next(c for c in chunk_blocks(parse_pdf(out.getvalue()).blocks) if c.table)
        ref=cell_reference(table,1,1)
        self.assertEqual((ref['row_label'],ref['column_label'],ref['value']),('Alpha','Score','12.3'))

    def setUp(self):
        self.data=grouped_pdf()
        self.paper=parse_pdf(self.data)
        self.chunks=chunk_blocks(self.paper.blocks)
        self.table=next(c for c in self.chunks if c.table)

    def test_subscripts_and_parent_headers_bind_to_distinct_source_regions(self):
        ref=cell_reference(self.table,2,1)
        self.assertEqual((ref['row_label'],ref['column_label'],ref['value']),('EncoderV1','Quality / Dev','18.4'))
        self.assertEqual(ref['column_groups'],['Quality'])
        self.assertEqual(len(ref['boxes']),4)
        self.assertLess(ref['boxes'][3][1],ref['boxes'][1][1])
        self.assertTrue(highlight_page(self.data,self.table,selected_boxes=ref['boxes']).startswith(b'\x89PNG'))
        expected={'page':1,'quote':'18.4','bbox':ref['boxes'][2],
                  'table':{'row_label':'EncoderV1','column_label':'Dev','column_group':'Quality'}}
        self.assertTrue(has_bound_cell(self.table,expected))
        wrong={**expected,'table':{**expected['table'],'column_group':'Expense'}}
        self.assertFalse(has_bound_cell(self.table,wrong))

    def test_repeated_leaf_names_require_disambiguation_not_first_match(self):
        all_cells=find_cells(self.chunks,'Encoder V1','Dev')
        self.assertEqual(len(all_cells),2)
        self.assertEqual({r['column_label'] for r in all_cells},{'Quality / Dev','Expense / Dev'})
        self.assertEqual(len(find_cells(self.chunks,'Encoder_V1','Quality / Dev')),1)
        self.assertEqual(find_cells(self.chunks,'Unrelated method','Quality / Dev'),[])
        self.assertEqual(find_cells(self.chunks,'','Dev'),[])

    def test_unruled_parent_headers_are_not_guessed(self):
        paper=parse_pdf(grouped_pdf(rules=False))
        self.assertFalse(any(b.table for b in paper.blocks))
        self.assertIn('28.6',' '.join(b.text for b in paper.blocks))

    def test_spanning_data_cells_keep_text_but_cannot_bind_a_value(self):
        chunks=chunk_blocks(parse_pdf(grouped_pdf(merged=True)).blocks)
        table=next(c for c in chunks if c.table)
        self.assertIn('shared cost',table.text)
        for col in (3,4):
            with self.assertRaises(ValueError):cell_reference(table,3,col)
        self.assertEqual(cell_reference(table,3,1)['value'],'19.4')

    def test_missing_parent_geometry_or_mismatched_paths_are_rejected(self):
        paths=list(self.table.table.header_paths)
        paths[1]=(TableCell('Quality',None),paths[1][-1])
        damaged=replace(self.table,table=replace(self.table.table,header_paths=tuple(paths)))
        with self.assertRaises(ValueError):cell_reference(damaged,1,1)

    def test_header_paths_survive_schema3_and_long_table_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'index.json'
            save_index(self.paper,self.chunks,path)
            _,loaded=load_index(path)
        self.assertEqual(next(c.table for c in loaded if c.table),self.table.table)
        small=chunk_blocks(self.paper.blocks,max_chars=240)
        tables=[c for c in small if c.table]
        self.assertGreater(len(tables),1)
        for chunk in tables:
            self.assertEqual(chunk.table.header_paths,self.table.table.header_paths)
            ref=cell_reference(chunk,1,1)
            self.assertTrue(all(tuple(b) in chunk.boxes for b in ref['boxes']))

    def test_numeric_header_notes_are_preserved_outside_data_rows(self):
        from reportlab.pdfgen import canvas
        out=io.BytesIO();c=canvas.Canvas(out,pagesize=(612,792));c.setFont('Helvetica',9)
        c.drawString(70,730,'Table 12. Dataset sizes below the column names')
        for y in (711,664,600):c.line(70,y,510,y)
        for x,t in [(80,'Method'),(285,'First'),(435,'Second')]:c.drawString(x,696,t)
        for x,t in [(285,'111k'),(435,'222k')]:c.drawString(x,678,t)
        for y,t in [(645,'Alpha'),(625,'Beta')]:
            for x,v in [(80,t),(285,'12.3'),(435,'45.6')]:c.drawString(x,y,v)
        c.save()
        table=next(c for c in chunk_blocks(parse_pdf(out.getvalue()).blocks) if c.table)
        self.assertEqual(len(table.table.rows),3)
        self.assertIn('111k',table.table.header_notes)
        self.assertIn('Header notes:',table.text)

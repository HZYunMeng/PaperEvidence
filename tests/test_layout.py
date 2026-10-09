"""Layout regressions use unrelated synthetic text, not benchmark answer labels."""
import io
import unittest
from dataclasses import replace

from paper_evidence.chunking import chunk_blocks
from paper_evidence.layout import words_of, text_of
from paper_evidence.models import Block, TableCell, TableData
from paper_evidence.parsing import parse_pdf
from paper_evidence.tables import cell_reference, table_value_claim
from paper_evidence.storage import save_index, load_index


def pdf_bytes(draw):
    from reportlab.pdfgen import canvas
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=(612,792))
    c.setFont('Helvetica',9)
    draw(c)
    c.save()
    return out.getvalue()


def columns(c):
    c.drawString(175,755,'A full width heading across both columns')
    c.setFont('Helvetica',9)
    for i in range(12):
        c.drawString(55,710-i*13,f'Left{i:02} alpha beta gamma delta epsilon zeta eta')
        c.drawString(315,710-i*13,f'Right{i:02} theta iota kappa lambda mu nu xi')


def horizontal(c, *, multiheader=False, caption=True):
    if caption:
        c.drawString(70,725,'Table 9. Unrelated experimental comparison')
    for y in (705,682,628):
        c.line(70,y,520,y)
    if multiheader:
        c.drawString(90,696,'Benchmark summary')
        c.drawString(90,685,'Method     Score     Cost')
    else:
        c.drawString(80,690,'Method')
        c.drawString(305,690,'Score')
        c.drawString(440,690,'Cost')
    for y,label,a,b in [(667,'Alpha encoder','17.4','3.2M'),(647,'Beta decoder','28.6','7.1M')]:
        c.drawString(80,y,label)
        c.drawString(305,y,a)
        c.drawString(440,y,b)


class LayoutTests(unittest.TestCase):
    def test_columns_are_not_interleaved_or_chunked_together(self):
        paper=parse_pdf(pdf_bytes(columns))
        texts=[b.text for b in paper.blocks]
        left=[i for i,t in enumerate(texts) if t.startswith('Left')]
        right=[i for i,t in enumerate(texts) if t.startswith('Right')]
        self.assertEqual((len(left),len(right)),(12,12))
        self.assertLess(max(left),min(right))
        self.assertTrue(all(not ('Left' in c.text and 'Right' in c.text) for c in chunk_blocks(paper.blocks)))
        self.assertTrue(any('full width heading' in t for t in texts))

    def test_single_column_prose_is_retained(self):
        def draw(c):
            for i in range(12):
                c.drawString(55,710-i*13,f'Line{i:02} the short prose contains several ordinary words without a column gutter.')
        paper=parse_pdf(pdf_bytes(draw))
        self.assertEqual(len(paper.blocks),12)
        self.assertTrue(all(b.flow=='single' for b in paper.blocks))
        self.assertTrue(all('without a column gutter.' in b.text for b in paper.blocks))

    def test_uneven_column_lengths_do_not_hide_the_gutter(self):
        def draw(c):
            for i in range(24):
                c.drawString(55,710-i*13,f'Left{i:02} alpha beta gamma delta epsilon zeta eta')
                if i < 8:
                    c.drawString(315,710-i*13,f'Right{i:02} theta iota kappa lambda mu nu xi')
        paper=parse_pdf(pdf_bytes(draw))
        self.assertEqual(sum(b.text.startswith('Right') for b in paper.blocks),8)
        self.assertTrue(all(not ('Left' in b.text and 'Right' in b.text) for b in paper.blocks))

    def test_horizontal_table_cells_keep_source_words_and_geometry(self):
        paper=parse_pdf(pdf_bytes(horizontal))
        table=next(c for c in chunk_blocks(paper.blocks) if c.table)
        self.assertIn(table.table.extraction,('horizontal-rules-word-gutters','horizontal-rules-header-bands'))
        self.assertIn('Table 9.',table.table.caption)
        ref=cell_reference(table,2,1)
        self.assertEqual((ref['row_label'],ref['column_label'],ref['value']),('Beta decoder','Score','28.6'))
        self.assertTrue(all(70<=b[0]<b[2]<=520 for b in ref['boxes']))
        self.assertTrue(all('28.6' not in b.text for b in paper.blocks if b.kind=='text'))

    def test_grid_without_table_caption_is_not_inferred(self):
        paper=parse_pdf(pdf_bytes(lambda c: horizontal(c,caption=False)))
        self.assertFalse(any(b.table for b in paper.blocks))
        self.assertTrue(any('28.6' in b.text for b in paper.blocks))

    def test_multirow_header_falls_back_without_losing_text(self):
        paper=parse_pdf(pdf_bytes(lambda c: horizontal(c,multiheader=True)))
        self.assertFalse(any(b.table and b.table.extraction=='horizontal-rules-word-gutters' for b in paper.blocks))
        self.assertIn('28.6',' '.join(b.text for b in paper.blocks))

    def test_duplicate_row_labels_include_source_context(self):
        table=next(c for c in chunk_blocks(parse_pdf(pdf_bytes(horizontal)).blocks) if c.table)
        rows=((TableCell('Method',(70,80,100,90)),TableCell('Variant',(150,80,200,90)),TableCell('Score',(300,80,330,90))),
              (TableCell('Encoder',(70,100,100,110)),TableCell('Small',(150,100,180,110)),TableCell('17.4',(300,100,330,110))),
              (TableCell('Encoder',(70,120,100,130)),TableCell('Large',(150,120,180,130)),TableCell('28.6',(300,120,330,130))))
        table=replace(table,table=TableData(rows))
        result=table_value_claim(table,2,2)
        self.assertEqual(result['text'],'Encoder / Variant=Large · Score：28.6')
        self.assertEqual(len(result['evidence'][0]['boxes']),4)
        self.assertEqual(cell_reference(table,1,2)['value'],'17.4')

    def test_ligature_physical_glyph_is_kept_once(self):
        char={'text':'ﬁ','x0':10,'x1':15,'top':10,'bottom':20,'doctop':10,'upright':True}
        words=words_of([char])
        self.assertEqual(words[0]['text'],'fi')
        self.assertEqual(len(words[0]['chars']),1)
        self.assertEqual(text_of([char,char]),'fi')
        self.assertEqual(words_of([char,char])[0]['text'],'fi')

    def test_off_page_graphics_do_not_break_crop(self):
        def draw(c):
            horizontal(c)
            c.line(310,750,700,750)
            c.line(310,720,700,720)
            c.line(310,615,700,615)
        paper=parse_pdf(pdf_bytes(draw))
        self.assertTrue(any(b.table for b in paper.blocks))

    def test_caption_and_cells_survive_index_roundtrip(self):
        import tempfile
        from pathlib import Path
        paper=parse_pdf(pdf_bytes(horizontal)); chunks=chunk_blocks(paper.blocks)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'index.json'
            save_index(paper,chunks,path)
            _,loaded=load_index(path)
        self.assertEqual(next(c.table for c in loaded if c.table),next(c.table for c in chunks if c.table))

    def test_long_table_repeats_header_without_losing_rows_or_coordinates(self):
        header=(TableCell('Method',(10,10,40,20)),TableCell('Score',(100,10,130,20)))
        rows=[header]
        for i in range(20):
            rows.append((TableCell(f'Variant-{i}',(10,40+i*15,65,50+i*15)),
                         TableCell(str(100+i),(100,40+i*15,130,50+i*15))))
        block=Block('id','doc',1,'Results','table','X'*1500,(10,10,130,335),TableData(tuple(rows),caption='Table 8. Comparison'))
        chunks=chunk_blocks([block],max_chars=180)
        self.assertGreater(len(chunks),1)
        restored=[row for c in chunks for row in c.table.rows[1:]]
        self.assertEqual(restored,rows[1:])
        for chunk in chunks:
            self.assertEqual(chunk.table.rows[0],header)
            self.assertEqual(chunk.table.caption,'Table 8. Comparison')
            self.assertTrue(all(box in chunk.boxes for row in chunk.table.rows for cell in row if (box:=cell.bbox)))
            self.assertLessEqual(len(chunk.text),180)
        self.assertEqual(cell_reference(chunks[-1],len(chunks[-1].table.rows)-1,1)['value'],'119')

    def test_split_table_does_not_validate_an_omitted_row(self):
        from eval.real_eval import has_evidence,has_bound_cell
        header=(TableCell('Method',(10,10,40,20)),TableCell('Score',(100,10,130,20)))
        rows=[header]+[(TableCell(f'Variant-{i}',(10,40+i*15,65,50+i*15)),
                        TableCell('100',(100,40+i*15,130,50+i*15))) for i in range(20)]
        block=Block('id','doc',1,'Results','table','X'*1500,(10,10,130,335),TableData(tuple(rows)))
        chunks=chunk_blocks([block],max_chars=160)
        target={'page':1,'quote':'100','bbox':rows[-1][1].bbox,'table':{'row_label':'Variant-19','column_label':'Score'}}
        self.assertFalse(has_evidence(chunks[0],target))
        self.assertFalse(has_bound_cell(chunks[0],target))
        self.assertTrue(has_evidence(chunks[-1],target))
        self.assertTrue(has_bound_cell(chunks[-1],target))


if __name__ == '__main__':
    unittest.main()

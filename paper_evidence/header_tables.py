"""Recover ruled numeric tables using header bands and source glyph geometry.

Only partial horizontal rules can establish a spanning parent header. Empty or
spanning data cells remain unbindable; uncertain layouts fall back to text.
"""
from bisect import bisect_right
from statistics import median
import re

from .layout import LETTER, _caption, box_of
from .models import TableCell, TableData


def row_chars(chars):
    chars = list({id(c):c for c in chars if c['text'].strip()}.values())
    if not chars:
        return []
    height = median(c['bottom']-c['top'] for c in chars)
    large = [c for c in chars if c['bottom']-c['top'] >= height*0.8]
    small = [c for c in chars if c not in large]
    rows = []
    for char in sorted(large,key=lambda c:(c['top']+c['bottom'])/2):
        center = (char['top']+char['bottom'])/2
        row = next((r for r in reversed(rows) if abs(center-median(
            (c['top']+c['bottom'])/2 for c in r)) <= height*0.35),None)
        if row is None:
            rows.append([char])
        else:
            row.append(char)
    for char in small:
        # A sub/superscript needs both vertical overlap and a nearby source glyph.
        compatible = [r for r in rows if any(
            min(c['bottom'],char['bottom'])-max(c['top'],char['top'])>=1
            and min(abs(char['x0']-c['x1']),abs(c['x0']-char['x1'])) <= 3 for c in r)]
        if compatible:
            nearest = min(compatible,key=lambda r:abs((char['top']+char['bottom'])/2-
                        median((c['top']+c['bottom'])/2 for c in r)))
            nearest.append(char)
        else:
            rows.append([char])
    return sorted(rows,key=lambda r:median((c['top']+c['bottom'])/2 for c in r))


def glyph_text(chars):
    from pdfplumber.utils import extract_text
    tolerance = max(3,median(c['bottom']-c['top'] for c in chars)*0.65)
    return extract_text(chars,x_tolerance=1,y_tolerance=tolerance).strip()


def phrases(chars):
    from pdfplumber.utils import extract_words
    tolerance = max(3,median(c['bottom']-c['top'] for c in chars)*0.65)
    words = extract_words(chars,x_tolerance=1,y_tolerance=tolerance,return_chars=True)
    groups = []
    for word in sorted(words,key=lambda w:w['x0']):
        glyphs = list({id(c):c for c in word['chars']}.values())
        if groups and word['x0']-box_of(groups[-1])[2] <= 4:
            groups[-1].extend(glyphs)
        else:
            groups.append(glyphs)
    return groups


def ruled_table(page,box):
    caption = _caption(page,box)
    if not caption:
        return None
    left,top,right,bottom = box
    rules = [e for e in page.edges if e['orientation']=='h' and
             abs(e['x0']-left)<=3 and abs(e['x1']-right)<=3 and top+2<e['top']<bottom-2]
    if not rules:
        return None
    separator = min(e['top'] for e in rules)
    rows = row_chars(page.crop(box).chars)
    headers = [r for r in rows if median((c['top']+c['bottom'])/2 for c in r)<separator]
    body = [r for r in rows if r not in headers]
    if not headers or len(headers)>4 or len(body)<2:
        return None
    groups = [phrases(r) for r in headers]
    labels = [[g for g in row if LETTER.search(glyph_text(g)) and
               not re.match(r'^[+−-]?\d',glyph_text(g))] for row in groups]
    leaf_index = max(range(len(labels)),key=lambda i:(len(labels[i]),i))
    leaves = labels[leaf_index]
    if len(leaves)<2 or len(leaves)>12:
        return None
    extras = [g for i,row in enumerate(labels) if i!=leaf_index for g in row]
    standalone = [g for g in extras if box_of(g)[2]<box_of(leaves[0])[0]-6]
    if len(standalone)>1:
        return None
    if standalone:
        label = standalone[0]
        leaves = [label]+leaves
        extras = [g for g in extras if g is not label]
    else:
        label = leaves[0]
    paths = [[TableCell(glyph_text(g),box_of(g))] for g in leaves]
    centers = [(box_of(g)[0]+box_of(g)[2])/2 for g in leaves]
    for parent in sorted(extras,key=lambda g:box_of(g)[1],reverse=True):
        pbox = box_of(parent)
        partial = [e for e in page.edges if e['orientation']=='h' and
                   top<e['top']<=separator and 0<=e['top']-pbox[3]<=max(6,(pbox[3]-pbox[1])*1.2) and
                   e['x0']<=pbox[0]+3 and e['x1']>=pbox[2]-3 and
                   e['x1']-e['x0']<right-left-6]
        matches = []
        for edge in partial:
            indices = [i for i,c in enumerate(centers) if i>0 and edge['x0']<=c<=edge['x1']]
            if len(indices)>=2:
                matches.append(indices)
        if len(matches)!=1 or any(len(paths[i])>1 for i in matches[0]):
            return None
        for i in matches[0]:
            paths[i].insert(0,TableCell(glyph_text(parent),pbox))
    if any(len(p)>1 for p in paths) and len({tuple(c.text for c in p) for p in paths})!=len(paths):
        return None
    data = [phrases(r) for r in body]
    # An unnamed row-label column places numbers beneath the first visible
    # metric header. Do not shift those metrics into the row-label column.
    if any(re.match(r'^[+−-]?\d',glyph_text(g)) and
           box_of(leaves[0])[0]-6 <= (box_of(g)[0]+box_of(g)[2])/2 < (centers[0]+centers[1])/2
           for row in data for g in row):
        return None
    # Locate the first numeric column from its header, excluding numeric citations
    # inside method names. Subsequent boundaries follow distinct leaf headers.
    edge = (centers[1]+centers[2])/2 if len(centers)>2 else right
    first_values = [box_of(g)[0] for row in data for g in row if
                    re.match(r'^[+−-]?\d',glyph_text(g)) and
                    box_of(leaves[1])[0]-6 <= (box_of(g)[0]+box_of(g)[2])/2 < edge]
    if not first_values:
        return None
    start = min(first_values)
    label_ends = [box_of(g)[2] for row in data for g in row if box_of(g)[2]<start-1]
    if not label_ends:
        return None
    boundaries = [(max(label_ends)+start)/2]+[(a+b)/2 for a,b in zip(centers[1:],centers[2:])]
    if boundaries != sorted(boundaries):
        return None
    header_cells = tuple(TableCell(' / '.join(c.text for c in p),p[-1].bbox) for p in paths)
    extracted, usable = [header_cells], 0
    for chars,row in zip(body,data,strict=True):
        if not any(re.search(r'\d',glyph_text(g)) for g in row):
            extracted.append((TableCell(glyph_text(chars),None),)+tuple(TableCell('',None) for _ in boundaries))
            continue
        cells = [[] for _ in leaves]
        spanning = set()
        for group in row:
            gbox = box_of(group)
            column = bisect_right(boundaries,(gbox[0]+gbox[2])/2)
            crossed = [i for i,x in enumerate(boundaries) if gbox[0]<x<gbox[2]]
            if crossed:
                first,last = crossed[0],crossed[-1]+1
                cells[first].extend(group)
                spanning.update(range(first,last+1))
            else:
                cells[column].extend(group)
        values = tuple(TableCell(glyph_text(cs),None if i in spanning else box_of(cs))
                       if cs else TableCell('',None) for i,cs in enumerate(cells))
        if not LETTER.search(values[0].text) or values[0].bbox is None:
            return None
        if any(c.bbox and re.search(r'\d',c.text) for c in values[1:]):
            usable += 1
        extracted.append(values)
    if usable<2:
        return None
    notes = '\n'.join(glyph_text(headers[i]) for i,row in enumerate(labels) if not row)
    return TableData(tuple(extracted),'horizontal-rules-header-bands',caption,
                     tuple(tuple(p) for p in paths),notes)

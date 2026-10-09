"""Conservative geometric recovery for two-column text and horizontal-rule tables.

No paper names, question labels, model calls or OCR are used. Uncertain candidates
fall back to text. Inferred column membership is a heuristic, not semantic proof.
"""
from bisect import bisect_right
from statistics import median
import re

from .models import TableCell, TableData

CAPTION = re.compile(r"^(?:Table|表)\s*\d+[.:：．]?", re.IGNORECASE)
LETTER = re.compile(r"[A-Za-z\u3400-\u9fff]")


def box_of(items):
    return (min(c['x0'] for c in items), min(c['top'] for c in items),
            max(c['x1'] for c in items), max(c['bottom'] for c in items))


def text_of(chars):
    from pdfplumber.utils import extract_text
    chars = list({id(c):c for c in chars}.values())
    return extract_text(chars, x_tolerance=1, y_tolerance=3).strip()


def words_of(chars):
    from pdfplumber.utils import extract_words
    chars = list({id(c):c for c in chars}.values())
    words = extract_words(chars, x_tolerance=1, y_tolerance=3, return_chars=True)
    # pdfplumber repeats the original char object when expanding a ligature.
    # Keep each physical glyph once before extracting the cell text again.
    for word in words:
        word['chars'] = list({id(c):c for c in word['chars']}.values())
    return words


def _merge(intervals):
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    return merged


def _ruled_regions(page):
    groups = []
    # Matching line extents are evidence of a common table region. Chart grids
    # are candidates too, and must pass header/data checks below.
    for edge in sorted(page.edges, key=lambda e: e['top']):
        if (edge['orientation'] != 'h' or edge['x1']-edge['x0'] < 80
                or edge['x0'] < page.bbox[0] or edge['x1'] > page.bbox[2]
                or not page.bbox[1] <= edge['top'] <= page.bbox[3]):
            continue
        group = next((g for g in groups if abs(g[0]['x0']-edge['x0']) <= 3
                      and abs(g[0]['x1']-edge['x1']) <= 3), None)
        if group is None:
            groups.append([edge])
        elif edge['top']-group[-1]['top'] > 2:
            group.append(edge)
    for group in groups:
        segments, segment = [], []
        left, right = min(e['x0'] for e in group), max(e['x1'] for e in group)
        for edge in group:
            if segment and edge['top'] > segment[-1]['top']+3:
                between = page.crop((left, segment[-1]['top']+1, right, edge['top']-1))
                lines = between.extract_text_lines(x_tolerance=1, return_chars=False)
                if any(CAPTION.match(l['text'].strip()) for l in lines):
                    segments.append(segment)
                    segment = []
            segment.append(edge)
        segments.append(segment)
        for rules in segments:
            if len(rules) >= 3 and rules[-1]['top']-rules[0]['top'] >= 20:
                yield (left, rules[0]['top'], right, rules[-1]['top'])


def _caption(page, box, margin=0):
    """Return nearby source text, stopping at table boundaries/large line gaps."""
    left, top, right, bottom = box
    left, right = max(page.bbox[0],left-margin), min(page.bbox[2],right+margin)
    candidates = []
    for y0, y1, above in [(max(0, top-80), top, True), (bottom, min(page.height, bottom+65), False)]:
        lines = page.crop((left, y0, right, y1)).extract_text_lines(x_tolerance=1, return_chars=False)
        starts = [i for i,l in enumerate(lines) if CAPTION.match(l['text'].strip())]
        if not starts:
            continue
        i = starts[-1] if above else starts[0]
        chosen = [lines[i]]
        for line in lines[i+1:]:
            if line['top']-chosen[-1]['bottom'] > 7 or CAPTION.match(line['text'].strip()):
                break
            chosen.append(line)
        distance = top-chosen[-1]['bottom'] if above else chosen[0]['top']-bottom
        if 0 <= distance <= 22:
            candidates.append((distance, '\n'.join(l['text'] for l in chosen)))
    if not candidates and margin == 0:
        return _caption(page,box,margin=24)
    return min(candidates, default=(0,''))[1]


def horizontal_tables(page, occupied=()):
    """Recover single-header numeric tables bounded by >=3 horizontal rules.

    Column gutters must be empty across all numeric data rows. Text is assigned
    by whole-word center; cell boxes bound actual source glyphs. Group labels
    are retained as spanning rows with no bindable geometry. Multirow headers
    and entirely borderless tables are deliberately not inferred.
    """
    result = []
    for box in _ruled_regions(page):
        if any(_overlap(box, old) > 0.5 for old in occupied):
            continue
        crop = page.crop(box)
        lines = crop.extract_text_lines(x_tolerance=1, return_chars=True)
        if len(lines) < 3:
            continue
        words = [words_of(line['chars']) for line in lines]
        numeric = [i for i,row in enumerate(words[1:], 1)
                   if sum(bool(re.search(r'\d', w['text'])) for w in row) >= 2]
        if len(numeric) < 2 or not LETTER.search(lines[0]['text']):
            continue
        # Require a nearby table caption as an additional guard against figures
        # and prose with horizontal rules. This also supplies retrieval context.
        caption = _caption(page, box)
        if not caption:
            continue
        intervals = _merge((w['x0'],w['x1']) for i in numeric for w in words[i])
        gutters = [(a[1]+b[0])/2 for a,b in zip(intervals, intervals[1:]) if b[0]-a[1] >= 4]
        if not 1 <= len(gutters) <= 11:
            continue
        count = len(gutters)+1
        rows = []
        valid = True
        for i, row in enumerate(words):
            cells = [[] for _ in range(count)]
            for word in row:
                column = bisect_right(gutters, (word['x0']+word['x1'])/2)
                cells[column].extend(word['chars'])
            if i not in numeric and i > 0:
                # Spanning group text cannot establish an individual data cell.
                rows.append((TableCell(lines[i]['text'],None),)+tuple(TableCell('',None) for _ in gutters))
                continue
            extracted = tuple(TableCell(text_of(cs), box_of(cs)) if cs else TableCell('',None) for cs in cells)
            if i == 0:
                if sum(bool(LETTER.search(c.text)) for c in extracted[1:]) != count-1:
                    valid = False
                    break
            elif (not LETTER.search(extracted[0].text)
                  or any(not c.text or '\n' in c.text for c in extracted)):
                valid = False
                break
            rows.append(extracted)
        if valid:
            result.append((box, TableData(tuple(rows), 'horizontal-rules-word-gutters', caption)))
    return result


def _overlap(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection / max(1, min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])))


def detect_gutter(lines, width):
    """Find a repeatedly empty central gap in otherwise populated text rows."""
    gaps, eligible = [], 0
    for line in lines:
        words = words_of(line['chars'])
        if (len(words) < 8 or line['x0'] >= width*0.4 or line['x1'] <= width*0.6):
            continue
        eligible += 1
        for i,(a,b) in enumerate(zip(words,words[1:])):
            if (i >= 2 and len(words)-i-1 >= 3 and 8 <= b['x0']-a['x1'] <= width*0.18
                    and width*0.4 <= (a['x1']+b['x0'])/2 <= width*0.6):
                gaps.append((a['x1'],b['x0']))
    if eligible < 6:
        return None
    scores = [(sum(a+1<=x<=b-1 for a,b in gaps), x)
              for x in range(int(width*0.4),int(width*0.6)+1)]
    support = max(s for s,_ in scores)
    if support < 6 or support/eligible < 0.35:
        return None
    # Prefer the middle of the longest best-supported interval.
    best = [x for s,x in scores if s==support]
    runs = []
    for x in best:
        if runs and x==runs[-1][-1]+1:
            runs[-1].append(x)
        else:
            runs.append([x])
    return median(max(runs,key=len))


def reading_objects(page, table_objects):
    """Return (bbox,kind,text,table,flow) in reading order, with flow boundaries."""
    boxes = [o[0] for o in table_objects]
    # Filter individual glyphs so a table line cannot swallow adjacent-column text.
    remaining = page.filter(lambda o: o.get('object_type') != 'char' or not any(
        b[0]-0.5 <= (o['x0']+o['x1'])/2 <= b[2]+0.5
        and b[1]-0.5 <= (o['top']+o['bottom'])/2 <= b[3]+0.5 for b in boxes))
    lines = remaining.extract_text_lines(x_tolerance=1, return_chars=True)
    gutter = detect_gutter(lines,page.width)
    objects = []
    for line in lines:
        chars = line['chars']
        if not chars:
            continue
        groups = [chars]
        if gutter is not None and line['x0'] < gutter < line['x1']:
            words = words_of(chars)
            left = [w for w in words if w['x1'] <= gutter]
            right = [w for w in words if w['x0'] >= gutter]
            if (left and right and len(left)+len(right)==len(words)
                    and min(w['x0'] for w in right)-max(w['x1'] for w in left) >= 8):
                groups = [[c for w in ws for c in w['chars']] for ws in (left,right)]
        for cs in groups:
            text = text_of(cs)
            if text:
                objects.append((box_of(cs),'text',text,None,''))
    objects.extend((*o,'') for o in table_objects)
    if gutter is None:
        return [(*o[:4], 'single') for o in sorted(objects,key=lambda o:(o[0][1],o[0][0]))], False
    spanning = sorted([o for o in objects if o[0][0]<gutter<o[0][2]],key=lambda o:o[0][1])
    # A full-width title/table divides the page into independent reading bands.
    result, pending = [], [o for o in objects if o not in spanning]
    for band,wide in enumerate(spanning+[None]):
        cutoff = wide[0][1] if wide else float('inf')
        current = [o for o in pending if (o[0][1]+o[0][3])/2 < cutoff]
        pending = [o for o in pending if o not in current]
        for column in ('left','right'):
            selected = [o for o in current if (o[0][0]+o[0][2])/2 < gutter] if column=='left' else [o for o in current if (o[0][0]+o[0][2])/2 >= gutter]
            result.extend((*o[:4],f'band{band}-{column}') for o in sorted(selected,key=lambda o:o[0][1]))
        if wide:
            result.append((*wide[:4],f'band{band}-wide'))
    return result, True

"""Discover parsed tables and their source labels without queries or models."""
from dataclasses import dataclass
import hashlib
import json
import re

from .tables import cell_reference


@dataclass(frozen=True)
class CatalogTable:
    id: str
    page: int
    caption: str
    chunks: tuple


def table_catalog(chunks):
    # Only the original block identity joins row fragments. Captions, headers
    # and page numbers alone cannot establish that two tables are the same.
    grouped = {}
    for chunk in chunks:
        if chunk.kind != 'table' or chunk.table is None:
            continue
        blocks = tuple(re.sub(r':rows\d+$','',bid) for bid in chunk.block_ids)
        identity = (chunk.document_id,chunk.page,blocks or (chunk.id,))
        grouped.setdefault(identity,[]).append(chunk)
    return tuple(CatalogTable(
        hashlib.sha256(repr(identity).encode()).hexdigest()[:16],
        group[0].page,group[0].table.caption,tuple(group))
        for identity,group in grouped.items())


def row_source(chunk, row):
    """Return raw same-row fields; do not infer experimental conditions."""
    fields = []
    for column,cell in enumerate(chunk.table.rows[row]):
        header = chunk.table.rows[0][column].text if column < len(chunk.table.rows[0]) else ''
        fields.append({'column':column,'column_label':header,'text':cell.text,
                       'bbox':list(cell.bbox) if cell.bbox else None})
    preceding = next((r[0].text for r in reversed(chunk.table.rows[1:row])
                      if r and r[0].text and r[0].bbox is None and
                      all(not c.text for c in r[1:])), '')
    return {'chunk_id':chunk.id,'document_id':chunk.document_id,'page':chunk.page,
            'row':row,'fields':fields,'preceding_group_text':preceding,
            'scope':'raw parsed row fields; no experimental-condition interpretation'}


def catalog_rows(entry):
    result,seen = [],set()
    for chunk in entry.chunks:
        for row in range(1,len(chunk.table.rows)):
            references = []
            for column in range(1,len(chunk.table.rows[row])):
                try:
                    references.append(cell_reference(chunk,row,column))
                except ValueError:
                    pass
            if not references:
                continue
            source = row_source(chunk,row)
            identity = json.dumps(source['fields'],ensure_ascii=False,sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            result.append({'id':hashlib.sha256(identity.encode()).hexdigest()[:16],
                           'chunk':chunk,'row':row,'references':references,'source':source})
    return tuple(result)


def row_label(source):
    fields = source['fields']
    label = fields[0]['text']
    context = [f"{f['column_label'] or '未命名列'}={f['text']}" for f in fields[1:] if f['text']]
    if source['preceding_group_text']:
        context.insert(0,source['preceding_group_text'])
    return label + (' · 同行原文：'+'; '.join(context) if context else '')


def match_label(reference, chunk):
    caption = ' '.join(reference['table_caption'].split())[:70] or '无表注'
    source = row_source(chunk,reference['row'])
    return (f"第 {reference['page']} 页 · {caption} · "
            f"{row_label(source)} · 第 {reference['column']+1} 列 "
            f"{reference['column_label']}={reference['value']}")

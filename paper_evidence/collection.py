"""Explicitly collected source cells; no ranking or experimental comparison."""
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path

from . import __version__
from .catalog import row_source
from .tables import cell_reference


def pdf_hash(source):
    data = source if isinstance(source,bytes) else Path(source).read_bytes()
    return hashlib.sha256(data).hexdigest()


def make_evidence(source, chunk, row, column, document=None):
    sha = pdf_hash(source)
    if sha[:16] != chunk.document_id:
        raise ValueError('PDF 文件与单元格来源不一致。')
    reference = cell_reference(chunk,row,column)
    identity = [sha,reference['page'],reference['boxes'][:3]]
    document = document or {}
    name = str(document.get('name','uploaded.pdf')).replace('\\','/').split('/')[-1]
    return {'id':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
            'document':{'name':name,'title':document.get('title',''),'sha256':sha},
            'reference':reference,'row_source':row_source(chunk,row),'app_version':__version__,
            'scope':'source-cell snapshot; no semantic support or comparability assessment'}


def add_evidence(items, item):
    """Keep the first snapshot of a physical cell, independent of chunk splits."""
    if any(old['id']==item['id'] for old in items):
        return items,False
    return [*items,deepcopy(item)],True


def export_json(items):
    return json.dumps({'schema_version':1,'kind':'paper-evidence-collection','items':items,
                       'scope':'user-selected parsed cells; not a comparative benchmark; PDFs not included'},
                      ensure_ascii=False,indent=2)


def _md(text):
    text = html.escape(str(text),quote=False)
    for char in ('\\','|','`','*','_','[',']'):
        text = text.replace(char,'\\'+char)
    return text.replace('\n',' ').replace('\r',' ')


def export_markdown(items):
    lines = ['# PaperEvidence — source cell collection','',
             'Selected parsed source cells. Experimental conditions and comparability require manual review.',
             'PDF files are not included. Page numbers refer to PDF pages.','',
             '| Source | File | Method | Metric | Value | PDF page |',
             '| --- | --- | --- | --- | --- | --- |']
    for i,item in enumerate(items,1):
        r=item['reference']
        values=[f'E{i}',item['document']['name'],r['row_label'],r['column_label'],r['value'],r['page']]
        lines.append('| '+' | '.join(_md(v) for v in values)+' |')
    for i,item in enumerate(items,1):
        r=item['reference'];d=item['document']
        lines.extend(['',f'## E{i}','',f"File: {_md(d['name'])}",
                      f"Title: {_md(d['title'])}",f"SHA256: `{d['sha256']}`",
                      f"App version: {_md(item['app_version'])}",
                      f"Table caption: {_md(r['table_caption'])}",
                      'Raw same-row fields: '+'; '.join(
                          _md(f"{f['column_label']}={f['text']}") for f in item['row_source']['fields']),
                      'Preceding group text: '+_md(item['row_source']['preceding_group_text']),
                      '', 'Source regions (row label, leaf header, value, then parent headers/context):',
                      '```json',json.dumps(r['boxes'],ensure_ascii=False),'```'])
    return '\n'.join(lines)+'\n'

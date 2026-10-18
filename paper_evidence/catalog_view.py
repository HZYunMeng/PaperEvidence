"""Source-first table navigation, kept separate from ranked retrieval."""
import json
import streamlit as st

from .catalog import table_catalog, catalog_rows, row_label, row_source
from .viewer import show_text, show_page


def choose_catalog_cell(source, chunks, document_id):
    """Return only an explicitly selected and submitted physical source cell."""
    with st.expander('不知道标签？从表格目录选择'):
        if not st.checkbox('打开表格目录',key=f'catalog-open-{document_id}'):
            return None
        entries = table_catalog(chunks)
        st.caption(f'识别出 {len(entries)} 个表格来源；长表的行片段合并显示。此数量不代表论文的实际表格总数。')
        if not entries:
            st.info('暂无结构化表格。可以展开逐页浏览，直接检查 PDF 原文。')
            return None
        lookup = {e.id:e for e in entries}
        selected = st.selectbox('选择表格',[None]+list(lookup),
            format_func=lambda tid:'请选择表格' if tid is None else
                f"第 {lookup[tid].page} 页 · {' '.join(lookup[tid].caption.split())[:100] or '无表注'} · 来源 {list(lookup).index(tid)+1}",
            key=f'catalog-table-{document_id}')
        if selected is None:
            return None
        entry = lookup[selected]
        rows = catalog_rows(entry)
        st.caption(f'{len(entry.chunks)} 个来源片段 · {len(rows)} 行至少有一个可绑定单元格。')
        with st.expander('查看这张表的解析原文'):
            for chunk in entry.chunks:
                show_text(chunk.text)
        if not rows:
            st.warning('这张表没有可可靠绑定的行列。保留了解析原文，请核对完整 PDF。')
            try:
                show_page(source,entry.page)
            except Exception as exc:
                st.warning(f'PDF 页面预览不可用：{exc}')
            return None
        row_lookup = {r['id']:r for r in rows}
        selected_row = st.selectbox('选择原表数据行',[None]+list(row_lookup),
            format_func=lambda rid:'请选择数据行' if rid is None else
                f"片段 {entry.chunks.index(row_lookup[rid]['chunk'])+1} 第 {row_lookup[rid]['row']+1} 行 · {row_label(row_lookup[rid]['source'])}",
            key=f'catalog-row-{document_id}-{entry.id}')
        if selected_row is None:
            return None
        item = row_lookup[selected_row]
        refs = {r['column']:r for r in item['references']}
        selected_column = st.selectbox('选择原表指标',[None]+list(refs),
            format_func=lambda col:'请选择指标' if col is None else
                f"第 {col+1} 列 · {refs[col]['column_label']}",
            key=f'catalog-column-{document_id}-{entry.id}-{selected_row}')
        st.caption('同行字段按原表展示，未自动解释为实验条件；缺少来源坐标的字段不能作为独立单元格引用。')
        if st.button('核对这个单元格',disabled=selected_column is None,
                     key=f'catalog-confirm-{document_id}-{entry.id}-{selected_row}'):
            return refs[selected_column]
    return None


def show_row_source(chunk, row, *, key_prefix):
    source = row_source(chunk,row)
    with st.expander('查看同行字段与上方分组原文'):
        st.caption('以下是解析原文，未推断实验条件、单位或语义支持。')
        if source['preceding_group_text']:
            st.text(source['preceding_group_text'])
            st.caption('上方分组文字没有独立坐标，需在 PDF 中核对。')
        st.dataframe([{'列':f['column']+1,'表头':f['column_label'],'原文':f['text'],
                       '字符区域':'有' if f['bbox'] else '无'} for f in source['fields']],hide_index=True)
        st.download_button('导出同行字段与坐标',json.dumps(source,ensure_ascii=False,indent=2),
                           file_name='source-row.json',mime='application/json',
                           key=f'{key_prefix}-row-export-{chunk.id}-{row}')

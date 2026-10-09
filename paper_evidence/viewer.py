"""Source inspection shared by search results and query-free page browsing."""
import json
import re
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from .parsing import highlight_page
from .tables import cell_reference, table_value_claim


def show_text(text):
    if text.startswith("| ") and "| ---" in text:
        table_lines = [line for line in text.splitlines() if line.startswith("| ")]
        caption = "\n".join(line for line in text.splitlines() if not line.startswith("| "))
        rows = [[cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
                for line in table_lines]
        if len(rows) >= 3 and all(len(row) == len(rows[0]) for row in rows):
            labels = [f"{label} (第 {i+1} 列)" if rows[0].count(label)>1 else label for i,label in enumerate(rows[0])]
            st.table(pd.DataFrame(rows[2:], columns=labels))
            if caption:
                st.caption(caption)
            return
    st.text(text)


def bindable_columns(chunk, row):
    columns = []
    if chunk.table and type(row) is int and 1 <= row < len(chunk.table.rows):
        for column in range(1,min(len(chunk.table.rows[row]),len(chunk.table.rows[0]))):
            try:
                cell_reference(chunk,row,column)
                columns.append(column)
            except ValueError:
                pass
    return columns


def source_reference(chunk):
    return {"chunk_id":chunk.id,"document_id":chunk.document_id,"page":chunk.page,
            "section":chunk.section,"kind":chunk.kind,"quote":chunk.text,
            "boxes":[list(b) for b in chunk.boxes],"cell_bound":False,
            "scope":"parsed source fragment and block regions; no answer support assessment"}


def source_label(chunk):
    excerpt = " ".join(chunk.text.split())[:55]
    return f"第 {chunk.page} 页 · {chunk.section} · {chunk.kind} · {excerpt}"


def show_page(source, page):
    # highlight_page only needs a page number and regions. No fake evidence ID.
    preview = highlight_page(source,SimpleNamespace(page=page,boxes=()),selected_boxes=[])
    st.image(preview,caption=f"PDF 第 {page} 页 · 完整原文",width="stretch")


def inspect_source(source, chunk, *, binding=None, key_prefix="search", locked=False):
    key = f"{key_prefix}-{chunk.document_id}-{chunk.id}"
    if binding:
        key += f"-{binding['row']}-{binding['column']}"
    boxes = None
    if chunk.table and st.checkbox("定位表格单元格",value=bool(binding),key=f"inspect-{key}",disabled=locked):
        st.caption("按解析行名和完整表头路径定位；分组表头会一并高亮。跨列或缺少来源的单元格不提供数值引用。")
        choices = [r for r in range(1,len(chunk.table.rows)) if bindable_columns(chunk,r)]
        if choices:
            row = st.selectbox("数据行",choices,
                index=choices.index(binding["row"]) if binding and binding["row"] in choices else 0,
                format_func=lambda r:f"第 {r+1} 行 · {' / '.join(c.text for c in chunk.table.rows[r][:2])}",key=f"row-{key}",disabled=locked)
            columns = bindable_columns(chunk,row)
            column = st.selectbox("数值列",columns,
                index=columns.index(binding["column"]) if binding and binding["column"] in columns else 0,
                format_func=lambda c:f"第 {c+1} 列 · {chunk.table.rows[0][c].text}",key=f"column-{key}",disabled=locked)
            reference = cell_reference(chunk,row,column)
            st.success(table_value_claim(chunk,row,column)["text"])
            boxes = reference["boxes"]
            st.download_button("导出单元格引用",json.dumps(reference,ensure_ascii=False,indent=2),
                file_name="table-cell-citation.json",mime="application/json",key=f"cell-export-{key}")
        else:
            st.warning("当前表格没有可可靠绑定的单元格，请核对 PDF 原文。")
    st.caption("黄色框标记选中单元格、行名、表头及必要的同名行上下文；未选择单元格时标记整个片段。")
    try:
        st.image(highlight_page(source,chunk,selected_boxes=boxes),caption=f"第 {chunk.page} 页",width="stretch")
    except Exception as exc:
        st.warning(f"PDF 页面预览不可用：{exc}")
    with st.expander("完整检索片段" if key_prefix=="search" else "完整解析片段"):
        st.text(chunk.text)
    st.download_button("导出原文片段与坐标",json.dumps(source_reference(chunk),ensure_ascii=False,indent=2),
        file_name="source-fragment.json",mime="application/json",key=f"source-export-{key}")

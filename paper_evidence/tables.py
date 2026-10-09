"""Bind an inspected table value to exact parsed cells, without guessing semantics."""
import re
import unicodedata


def cell_reference(chunk, row, column):
    if type(row) is not int or type(column) is not int:
        raise ValueError("表格行列必须是整数。")
    table = chunk.table
    if chunk.kind != "table" or table is None:
        raise ValueError("该片段没有结构化单元格；请重新导入论文或查看原文。")
    if (row < 1 or row >= len(table.rows) or column < 1 or column >= len(table.rows[row])
            or column >= len(table.rows[0])):
        raise ValueError("请选择数据行与数值列；当前不支持引用表头或首列作为数值。")
    cells = (table.rows[row][0], table.rows[0][column], table.rows[row][column])
    if any(not cell.text.strip() or cell.bbox is None or "\n" in cell.text for cell in cells):
        raise ValueError("存在空白、合并或多行单元格，当前无法可靠绑定；请查看 PDF。")
    label, header, value = (cell.text.strip() for cell in cells)
    path = table.header_paths[column] if len(table.header_paths)==len(table.rows[0]) else ()
    if table.header_paths and (not path or any(c.bbox is None or not c.text.strip() or '\n' in c.text for c in path)
            or ' / '.join(c.text for c in path)!=header or path[-1].bbox!=cells[1].bbox):
        raise ValueError("分组表头来源不完整或与列不一致；请查看 PDF。")
    # A matrix of chart tick labels is not enough to establish row/column meanings.
    if not re.search(r"[A-Za-z\u3400-\u9fff]",label) or not re.search(r"[A-Za-z\u3400-\u9fff]",header):
        raise ValueError("表头或行名缺少可识别文字；当前无法可靠绑定。")
    same_label = [r for r in table.rows[1:] if r and r[0].text.strip()==label]
    context = []
    if len(same_label) > 1:
        for index, other in enumerate(table.rows[row][1:], 1):
            if (index != column and other.bbox is not None and "\n" not in other.text
                    and re.search(r"[A-Za-z\u3400-\u9fff]",other.text)
                    and table.rows[0][index].text.strip()):
                context.append({"column_label":table.rows[0][index].text.strip(),
                                "value":other.text.strip(),"bbox":list(other.bbox)})
    return {"chunk_id":chunk.id,"document_id":chunk.document_id,"page":chunk.page,
            "section":chunk.section,"kind":"table","row":row,"column":column,
            "row_label":label,"column_label":header,"value":value,
            "quote":value,"quote_verified":True,"cell_bound":True,
            "boxes":[list(cell.bbox) for cell in cells]+[list(c.bbox) for c in path[:-1]]+[c["bbox"] for c in context],
            "column_leaf_label":path[-1].text if path else header,
            "column_groups":[c.text for c in path[:-1]],
            "header_path":[{"text":c.text,"bbox":list(c.bbox)} for c in path],
            "row_context":context,"table_caption":table.caption,
            "interpretation":"source header path and first-column row label" if path else
                             "first row treated as header, first column as row label"}


def table_value_claim(chunk, row, column):
    reference = cell_reference(chunk,row,column)
    context = " / " + "; ".join(f'{c["column_label"]}={c["value"]}' for c in reference["row_context"]) if reference["row_context"] else ""
    return {"text":f'{reference["row_label"]}{context} · {reference["column_label"]}：{reference["value"]}',
            "evidence":[reference],"table_value":{
                "chunk_id":chunk.id,"row":row,"column":column,"value":reference["value"]}}


def label_key(text):
    """Only normalize typography, spacing and punctuation; never infer aliases."""
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold()
                   if not c.isspace() and c not in '_()-')


def find_cells(chunks, method, metric):
    """Exact normalized label lookup, distinct from free-form evidence retrieval.

    All matching cells are returned. A repeated leaf label such as EN-DE needs
    its full group path or user disambiguation; never silently choose one.
    """
    method,metric = label_key(method),label_key(metric)
    if not method or not metric:
        return []
    found,seen = [],set()
    for chunk in chunks:
        if not chunk.table:
            continue
        for row in range(1,len(chunk.table.rows)):
            if label_key(chunk.table.rows[row][0].text)!=method:
                continue
            for column in range(1,min(len(chunk.table.rows[0]),len(chunk.table.rows[row]))):
                try:
                    reference = cell_reference(chunk,row,column)
                except ValueError:
                    continue
                if metric not in {label_key(reference['column_label']),label_key(reference['column_leaf_label'])}:
                    continue
                identity = (chunk.document_id,chunk.page,tuple(tuple(b) for b in reference['boxes']))
                if identity not in seen:
                    seen.add(identity)
                    found.append(reference)
    return found

"""Bind an inspected table value to exact parsed cells, without guessing semantics."""
import re


def cell_reference(chunk, row, column):
    if type(row) is not int or type(column) is not int:
        raise ValueError("表格行列必须是整数。")
    table = chunk.table
    if chunk.kind != "table" or table is None:
        raise ValueError("该片段没有结构化单元格；请重新导入论文或查看原文。")
    if row < 1 or row >= len(table.rows) or column < 1 or column >= len(table.rows[row]):
        raise ValueError("请选择数据行与数值列；当前不支持引用表头或首列作为数值。")
    cells = (table.rows[row][0], table.rows[0][column], table.rows[row][column])
    if any(not cell.text.strip() or cell.bbox is None or "\n" in cell.text for cell in cells):
        raise ValueError("存在空白、合并或多行单元格，当前无法可靠绑定；请查看 PDF。")
    label, header, value = (cell.text.strip() for cell in cells)
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
            "boxes":[list(cell.bbox) for cell in cells]+[c["bbox"] for c in context],
            "row_context":context,"table_caption":table.caption,
            "interpretation":"first row treated as header, first column as row label"}


def table_value_claim(chunk, row, column):
    reference = cell_reference(chunk,row,column)
    context = " / " + "; ".join(f'{c["column_label"]}={c["value"]}' for c in reference["row_context"]) if reference["row_context"] else ""
    return {"text":f'{reference["row_label"]}{context} · {reference["column_label"]}：{reference["value"]}',
            "evidence":[reference],"table_value":{
                "chunk_id":chunk.id,"row":row,"column":column,"value":reference["value"]}}

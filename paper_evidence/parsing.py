import hashlib
import io
import re
from pathlib import Path

from .models import Block, Paper, TableCell, TableData
from .layout import _overlap, horizontal_tables, reading_objects


HEADINGS = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:abstract|introduction|background|related work|methods?|methodology|"
    r"experiments?|results?|discussion|conclusions?|limitations?|references|"
    r"摘要|引言|背景|相关工作|方法|实验|结果|讨论|结论|局限性|参考文献)\s*$",
    re.IGNORECASE,
)


def _inside(box, outer):
    return box[0] >= outer[0] - 1 and box[1] >= outer[1] - 1 and \
        box[2] <= outer[2] + 1 and box[3] <= outer[3] + 1


def table_markdown(rows):
    cleaned = [[(cell or "").replace("\n", " ").replace("|", "\\|") for cell in row] for row in rows]
    if not cleaned:
        return ""
    width = max(map(len, cleaned))
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    lines = ["| " + " | ".join(row) + " |" for row in cleaned]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def structured_table(table):
    rows = table.extract()
    if len(rows) < 2 or not rows or max(map(len, rows)) < 2:
        return None
    # Sparse chart grids must not swallow their text as a purported table.
    occupied = sum(bool(cell and cell.strip()) for row in rows for cell in row)
    if occupied / sum(map(len, rows)) < 0.65:
        return None
    result = []
    for values, geometry in zip(rows, table.rows, strict=True):
        result.append(tuple(TableCell(value or "", tuple(map(float, box)) if box else None)
                            for value, box in zip(values, geometry.cells, strict=True)))
    return TableData(tuple(result))


def parse_pdf(source, name=None, *, max_pages=200):
    """Geometric text/table extraction. No OCR or vision inference."""
    import pdfplumber

    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ValueError("文件不是有效的 PDF。")
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("原型支持最大 30 MB 的 PDF。")
    doc_id = hashlib.sha256(data).hexdigest()[:16]
    name = name or ("uploaded.pdf" if isinstance(source, bytes) else Path(source).name)
    blocks, warnings = [], []
    section = "Front matter"
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        if len(pdf.pages) > max_pages:
            raise ValueError(f"原型支持最多 {max_pages} 页。")
        for page in pdf.pages:
            candidates = page.find_tables()
            objects = []
            # Prefer glyph-bounded rows over a ruled-grid cell containing many
            # data rows. Failed candidates remain available to text extraction.
            recovered = horizontal_tables(page)
            for box, structure in recovered:
                text = table_markdown([[cell.text for cell in row] for row in structure.rows])
                if structure.caption:
                    text += "\n" + structure.caption
                if structure.header_notes:
                    text += "\nHeader notes: " + structure.header_notes
                objects.append((box, "table", text, structure))
            for table in candidates:
                if any(_overlap(table.bbox, obj[0]) > 0.5 for obj in objects):
                    continue
                structure = structured_table(table)
                if structure is None:
                    warnings.append(f"第 {page.page_number} 页有稀疏或不完整网格，未作为表格索引；保留可提取文字。")
                    continue
                text = table_markdown([[cell.text for cell in row] for row in structure.rows])
                if text.strip():
                    objects.append((table.bbox, "table", text, structure))
            ordered, dual_column = reading_objects(page, objects)
            if not page.chars:
                warnings.append(f"第 {page.page_number} 页没有文本层，需要 OCR；该页未进入检索。")
            if page.images:
                warnings.append(f"第 {page.page_number} 页含 {len(page.images)} 个图片对象；图片内容尚未进入检索。")
            for index, (box, kind, text, structure, flow) in enumerate(ordered):
                if kind == "text" and HEADINGS.fullmatch(text):
                    kind = "heading"
                if kind == "heading":
                    section = text
                blocks.append(Block(
                    f"{doc_id}:p{page.page_number}:b{index}", doc_id,
                    page.page_number, section, kind, text, tuple(map(float, box)), structure, flow,
                ))
        count = len(pdf.pages)
    warnings.append("阅读顺序和横线表格按坐标启发式恢复；分组表头需明确的分组横线。无边框表格、跨页表格、公式和复杂版式仍需原文核对。")
    return Paper(doc_id, name, count, tuple(blocks), tuple(warnings))


def highlight_page(source, chunk, *, resolution=110, selected_boxes=None):
    import pdfplumber

    stream = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    with pdfplumber.open(stream) as pdf:
        page = pdf.pages[chunk.page - 1]
        image = page.to_image(resolution=resolution)
        for box in chunk.boxes if selected_boxes is None else selected_boxes:
            image.draw_rect(box, fill=(255, 218, 96, 70), stroke=(216, 132, 0), stroke_width=1)
        out = io.BytesIO()
        image.save(out, format="PNG")
    return out.getvalue()

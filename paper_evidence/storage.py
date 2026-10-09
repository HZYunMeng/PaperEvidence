import json
from dataclasses import asdict
from pathlib import Path

from .models import Chunk, TableCell, TableData


def decode_table(value):
    if value is None:
        return None
    def cells(rows):
        return tuple(tuple(TableCell(cell["text"],tuple(cell["bbox"]) if cell["bbox"] else None)
                           for cell in row) for row in rows)
    return TableData(cells(value["rows"]),value["extraction"],value.get("caption", ""),
                     cells(value.get("header_paths",())),value.get("header_notes",""))


def save_index(paper, chunks, out):
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 3, "paper": paper.to_dict(), "chunks": [asdict(c) for c in chunks]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_index(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") not in {1, 2, 3}:
        raise ValueError("不支持的索引版本。")
    chunks = tuple(Chunk(**{
        **row, "block_ids": tuple(row["block_ids"]),
        "boxes": tuple(tuple(box) for box in row["boxes"]),
        "table": decode_table(row.get("table")),
    }) for row in data["chunks"])
    return data["paper"], chunks

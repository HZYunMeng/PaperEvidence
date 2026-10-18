from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TableCell:
    text: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class TableData:
    rows: tuple[tuple[TableCell, ...], ...]
    extraction: str = "pdfplumber-lines"
    caption: str = ""
    header_paths: tuple[tuple[TableCell, ...], ...] = ()
    header_notes: str = ""


@dataclass(frozen=True)
class Block:
    id: str
    document_id: str
    page: int
    section: str
    kind: str
    text: str
    bbox: tuple[float, float, float, float]
    table: TableData | None = None
    flow: str = ""


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    page: int
    section: str
    kind: str
    text: str
    block_ids: tuple[str, ...]
    boxes: tuple[tuple[float, float, float, float], ...]
    table: TableData | None = None


@dataclass(frozen=True)
class Paper:
    id: str
    name: str
    pages: int
    blocks: tuple[Block, ...]
    warnings: tuple[str, ...]

    def to_dict(self):
        return asdict(self)

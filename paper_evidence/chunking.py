import hashlib
from dataclasses import replace

from .models import Chunk, TableData


def table_parts(block, max_chars):
    """Split long structured tables only between rows; repeat header/caption.

    Row indices refer to each returned fragment, not to the original full table.
    Physical glyph coordinates are preserved. Spanning group rows are carried
    forward as context and remain unbindable.
    """
    from .parsing import table_markdown
    if not block.table or len(block.text) <= max_chars:
        return [block]
    table, result, pending, group = block.table, [], [], None
    header = table.rows[0]

    def render(rows):
        text = table_markdown([[c.text for c in r] for r in rows])
        return text + ('\n'+table.caption if table.caption else '')

    def flush():
        if len(pending) < 2:
            return
        structure = TableData(tuple(pending),table.extraction,table.caption)
        result.append(replace(block,id=f'{block.id}:rows{len(result)}',
                              text=render(pending),table=structure))
        pending.clear()

    for row in table.rows[1:]:
        spanning = row[0].text and row[0].bbox is None and all(not c.text for c in row[1:])
        if spanning:
            group = row
        if not pending:
            pending.append(header)
            if group and group != row:
                pending.append(group)
        if len(pending)>1 and len(render(pending+[row])) > max_chars:
            flush()
            pending.append(header)
            if group and group != row:
                pending.append(group)
        pending.append(row)
    flush()
    return result


def chunk_blocks(blocks, max_chars=1400):
    """Keep page/section/flow boundaries and table rows; retain source block IDs."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    result, pending = [], []

    def flush():
        if not pending:
            return
        first = pending[0]
        text = "\n".join(b.text for b in pending)
        identity = "|".join(b.id for b in pending) + "|" + text
        result.append(Chunk(
            hashlib.sha256(identity.encode()).hexdigest()[:16],
            first.document_id, first.page, first.section, first.kind,
            text, tuple(b.id for b in pending), tuple(b.bbox for b in pending),
            first.table if len(pending) == 1 else None,
        ))
        pending.clear()

    expanded = (part for block in blocks for part in
                (table_parts(block,max_chars) if block.kind=='table' else [block]))
    for block in expanded:
        if block.kind == "heading":
            flush()
            continue
        if pending and (
            (block.document_id, block.page, block.section, block.kind, block.flow) !=
            (pending[0].document_id, pending[0].page, pending[0].section, pending[0].kind, pending[0].flow)
            or sum(len(b.text) + 1 for b in pending) + len(block.text) > max_chars
        ):
            flush()
        pending.append(block)
        # Preserve a complete table or an oversized single line rather than silently truncate it.
        if block.kind == "table" or len(block.text) >= max_chars:
            flush()
            if block.table and ':rows' in block.id:
                # A tall split table must not pretend that its omitted rows are
                # source evidence. Retain only header/current-row glyph boxes.
                result[-1] = replace(result[-1],boxes=tuple(
                    cell.bbox for row in block.table.rows for cell in row if cell.bbox))
    flush()
    return tuple(result)

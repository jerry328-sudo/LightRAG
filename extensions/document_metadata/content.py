"""Read parsed documents and their indexed context within one workspace."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal
from urllib.parse import urlsplit

import regex
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lightrag.chunk_schema import normalize_chunk_sidecar
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.utils_pipeline import resolve_sidecar_uri, strip_lightrag_doc_prefix


_MAX_EXCERPT_CHARS = 16000
_PARAGRAPH_END = re.compile(r"\n[ \t\r]*\n")
_PASSAGE_PATTERN = r"^[0-9a-f]{64}\.[0-9]{1,12}\.[0-9]{1,12}$"


class ContentOptions(BaseModel):
    scope: Literal["outline", "range", "full"] | None = Field(
        default=None,
        description="Defaults to outline, or range when a selector is given.",
    )
    q: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Search parsed text, matching one line at a time.",
    )
    match: Literal["literal", "regex"] = "literal"
    case_sensitive: bool = False
    context_lines: int = Field(default=2, ge=0, le=8)
    line_start: int | None = Field(
        default=None, ge=1, description="First parsed-text line, counted from one."
    )
    line_end: int | None = Field(
        default=None, ge=1, description="Last parsed-text line, inclusive."
    )
    page_start: int | None = Field(
        default=None,
        ge=1,
        description="First PDF page. Returns parsed blocks with page anchors, which may span pages.",
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
        description="Last PDF page, inclusive. Defaults to page_start.",
    )
    section: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Exact heading or block_id from the document outline.",
    )
    chunk_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Chunk belonging to this document, as returned by retrieval.",
    )
    neighbor_chunks: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of chunks to include on either side of chunk_id.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Reuse next_offset with the same selectors to continue.",
    )
    max_chars: int = Field(
        default=6000,
        ge=256,
        le=_MAX_EXCERPT_CHARS,
        description="Maximum text characters in a bounded response.",
    )
    limit: int = Field(
        default=12,
        ge=1,
        le=40,
        description="Maximum search hits or outline entries in the response.",
    )
    include_citations: bool = Field(
        default=False,
        description="Return a versioned passage locator with each excerpt.",
    )
    passage: str | None = Field(
        default=None,
        max_length=90,
        pattern=_PASSAGE_PATTERN,
        description=(
            "Read the exact excerpt identified by a returned passage locator. "
            "For a chunk locator, include its chunk_id. Other selectors and "
            "continuation offsets are unavailable with a passage locator."
        ),
    )


@dataclass(frozen=True)
class Block:
    block_id: str
    start: int
    end: int
    heading: str
    level: int
    pages: tuple[int, ...]


def _page_anchors(row: dict) -> tuple[int, ...]:
    anchors = set()
    positions = row.get("positions")
    if not isinstance(positions, list):
        return ()
    for position in positions:
        if not isinstance(position, dict) or position.get("type") != "bbox":
            continue
        value = position.get("anchor")
        if isinstance(value, str) and re.fullmatch(r"[0-9]{1,9}", value):
            value = int(value)
        if type(value) is int and value > 0:
            anchors.add(value)
    return tuple(sorted(anchors))


def _read_blocks(
    record: dict, document_id: str, text: str, input_dir: Path | None
) -> list[Block]:
    uri = record.get("sidecar_location")
    if input_dir is None or not isinstance(uri, str):
        return []
    try:
        parts = urlsplit(uri)
        if parts.scheme != "file" or parts.netloc not in ("", "localhost"):
            return []
        path = resolve_sidecar_uri(uri)
        if path is None:
            return []
        directory = path.resolve()
        if directory.parent not in (input_dir, input_dir / "__parsed__"):
            return []
        candidates = sorted(directory.glob("*.blocks.jsonl"))
        if len(candidates) != 1:
            return []
        file = candidates[0]
        if file.resolve().parent != directory or not file.is_file():
            return []

        blocks = []
        position = 0
        with file.open(encoding="utf-8") as handle:
            header = json.loads(next(handle, "{}"))
            if (
                not isinstance(header, dict)
                or header.get("type") != "meta"
                or header.get("doc_id") != document_id
            ):
                return []
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("type") != "content":
                    continue
                content = row.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                if blocks:
                    if not text.startswith("\n\n", position):
                        return []
                    position += 2
                if not text.startswith(content, position):
                    return []
                end = position + len(content)
                level = row.get("level")
                blocks.append(
                    Block(
                        str(row.get("blockid") or ""),
                        position,
                        end,
                        str(row.get("heading") or ""),
                        level if type(level) is int else 0,
                        _page_anchors(row),
                    )
                )
                position = end
        # Old or stale sidecars may remain after a reparse. Trust locations
        # only when the ordered source blocks reproduce this document's text.
        return blocks if position == len(text) else []
    except (OSError, RuntimeError, ValueError):
        return []


@dataclass(frozen=True)
class ParsedText:
    text: str
    blocks: list[Block]
    starts: list[int]

    @classmethod
    def from_record(
        cls, record: dict, document_id: str, text: str, input_dir: Path | None
    ) -> ParsedText:
        return cls(
            text,
            _read_blocks(record, document_id, text, input_dir),
            [0, *(index.end() for index in re.finditer("\n", text))],
        )

    def location(self, start: int, end: int) -> dict:
        overlapping = [
            block for block in self.blocks if block.start < end and block.end > start
        ]
        pages = sorted({page for block in overlapping for page in block.pages})
        result = {
            "line_start": bisect_right(self.starts, start),
            "line_end": bisect_right(self.starts, max(start, end - 1)),
        }
        if pages:
            result.update(pages=pages, page_precision="block")
        if len(overlapping) == 1 and overlapping[0].heading:
            result["section"] = overlapping[0].heading
        return result

    def bounds(self, start: int, end: int) -> tuple[int, int]:
        if start > len(self.starts):
            raise HTTPException(416, "The line range starts beyond the parsed text")
        if end < start:
            raise HTTPException(422, "line_end must be at least line_start")
        return self.starts[start - 1], self.starts[end] if end < len(
            self.starts
        ) else len(self.text)


def _outline(document: ParsedText, options: ContentOptions, has_chunks: bool) -> dict:
    entries = [block for block in document.blocks if block.heading]
    pages = {page for block in document.blocks for page in block.pages}
    areas = ["lines", "search", "full"]
    if has_chunks:
        areas.append("chunk")
    if entries:
        areas.append("section")
    if pages:
        areas.append("page")
    total = len(entries)
    end = options.offset + options.limit
    return {
        "selectors": areas,
        "line_count": len(document.starts),
        "page_range": [min(pages), max(pages)] if pages else None,
        "sections": [
            {
                "heading": block.heading[:512],
                "block_id": block.block_id,
                "level": block.level,
                "line": bisect_right(document.starts, block.start),
                "pages": list(block.pages),
            }
            for block in entries[options.offset : end]
        ],
        "next_offset": end if end < total else None,
    }


def _section_region(document: ParsedText, section: str) -> tuple[int, int]:
    hits = [
        idx for idx, block in enumerate(document.blocks) if block.block_id == section
    ]
    if not hits:
        hits = [
            idx
            for idx, block in enumerate(document.blocks)
            if block.heading and block.heading.casefold() == section.casefold()
        ]
    if not hits:
        raise HTTPException(404, "Section not found in the parsed document outline")
    if len(hits) != 1:
        raise HTTPException(
            422, "Several sections have that heading. Use the block_id from the outline"
        )
    index = hits[0]
    chosen = document.blocks[index]
    after = next(
        (
            block
            for block in document.blocks[index + 1 :]
            if block.heading and block.level <= chosen.level
        ),
        None,
    )
    return chosen.start, after.start if after else len(document.text)


def _page_regions(document: ParsedText, start: int, end: int) -> list[tuple[int, int]]:
    if end < start:
        raise HTTPException(422, "page_end must be at least page_start")
    if not any(block.pages for block in document.blocks):
        raise HTTPException(
            422,
            "This document has no page anchors. Use section, line, or keyword selectors",
        )
    ranges = []
    for block in document.blocks:
        if not any(start <= page <= end for page in block.pages):
            continue
        if ranges and ranges[-1][1] + 2 == block.start:
            ranges[-1] = (ranges[-1][0], block.end)
        else:
            ranges.append((block.start, block.end))
    return ranges


def _bounded_end(text: str, start: int, end: int, remaining: int) -> int:
    stop = min(end, start + remaining)
    if stop >= end:
        return stop
    # Preserve continuation offsets and the hard budget. A single paragraph
    # exceeding the budget must still make forward progress.
    boundary = max(
        (match.end() for match in _PARAGRAPH_END.finditer(text, start, stop)),
        default=start,
    )
    return boundary if text[start:boundary].strip() else stop


def _text_revision(text: str, anchors, document_revision: str = "") -> str:
    digest = hashlib.sha256(text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(document_revision.encode("ascii"))
    digest.update(
        json.dumps(anchors, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _document_revision(document: ParsedText) -> str:
    anchors = [
        (
            block.block_id,
            block.start,
            block.end,
            block.heading,
            block.level,
            block.pages,
        )
        for block in document.blocks
    ]
    return _text_revision(document.text, anchors)


def _citation(revision: str | None, start: int, end: int) -> dict:
    return {"passage": f"{revision}.{start}.{end}"} if revision and start < end else {}


def _passage_bounds(passage: str, revision: str, length: int) -> tuple[int, int]:
    recorded_revision, start, end = passage.split(".")
    if recorded_revision != revision:
        raise HTTPException(409, "The cited text has changed. Request a new excerpt.")
    lower, upper = int(start), int(end)
    if not 0 <= lower < upper <= length or upper - lower > _MAX_EXCERPT_CHARS:
        raise HTTPException(
            416,
            "The cited range is outside the parsed text or exceeds the excerpt limit.",
        )
    return lower, upper


def _read_regions(
    document: ParsedText,
    regions: list[tuple[int, int]],
    options: ContentOptions,
    revision: str | None = None,
) -> dict:
    skip = options.offset
    count = 0
    segments = []
    total = sum(end - start for start, end in regions)
    for start, end in regions:
        if skip >= end - start:
            skip -= end - start
            continue
        start += skip
        skip = 0
        stop = _bounded_end(document.text, start, end, options.max_chars - count)
        segments.append(
            {
                "text": document.text[start:stop],
                **document.location(start, stop),
                **_citation(revision, start, stop),
            }
        )
        count += stop - start
        if stop < end or count >= options.max_chars:
            break
    end_offset = options.offset + count
    return {
        "segments": segments,
        "next_offset": end_offset if end_offset < total else None,
    }


def _find_lines(
    document: ParsedText, options: ContentOptions, revision: str | None = None
) -> dict:
    flags = 0 if options.case_sensitive else regex.IGNORECASE
    pattern = regex.escape(options.q) if options.match == "literal" else options.q
    try:
        matcher = regex.compile(pattern, flags)
    except regex.error as error:
        raise HTTPException(422, "Invalid regular expression") from error

    lower = options.line_start or 1
    upper = min(options.line_end or len(document.starts), len(document.starts))
    if options.line_end is not None and options.line_end < lower:
        raise HTTPException(422, "line_end must be at least line_start")

    deadline = monotonic() + 0.5
    remaining = options.max_chars
    hits = []
    try:
        for line in range(lower + options.offset, upper + 1):
            start, end = document.bounds(line, line)
            remaining_time = deadline - monotonic()
            if remaining_time <= 0:
                raise TimeoutError
            line_text = document.text[start:end].removesuffix("\n")
            found = matcher.search(line_text, timeout=remaining_time)
            if found is None:
                continue
            if len(hits) >= options.limit or remaining <= 0:
                return {"matches": hits, "next_offset": line - lower}
            before, after = document.bounds(
                max(lower, line - options.context_lines),
                min(upper, line + options.context_lines),
            )
            # Center a clipped excerpt on the match so a very long line does
            # not use the budget before the matching term appears.
            excerpt_limit = min(1000, remaining)
            if after - before > excerpt_limit:
                before = max(
                    before,
                    min(
                        start + found.start() - excerpt_limit // 3,
                        after - excerpt_limit,
                    ),
                )
                after = before + excerpt_limit
            snippet = document.text[before:after]
            hits.append(
                {
                    "line": line,
                    "column": found.start() + 1,
                    "text": snippet,
                    **document.location(before, after),
                    **_citation(revision, before, after),
                }
            )
            remaining -= len(snippet)
    except TimeoutError as error:
        raise HTTPException(
            422,
            "Document search exceeded its time limit. Narrow the line range or simplify the pattern",
        ) from error
    return {"matches": hits, "next_offset": None}


async def _read_chunks(
    rag,
    status,
    document_id: str,
    document: ParsedText,
    options: ContentOptions,
    document_revision: str | None = None,
) -> dict:
    ordered = status.chunks_list or []
    try:
        anchor = ordered.index(options.chunk_id)
    except ValueError as error:
        raise HTTPException(404, "Chunk not found in this document") from error
    neighbors = 0 if options.passage else options.neighbor_chunks
    selected = ordered[max(0, anchor - neighbors) : anchor + neighbors + 1]
    rows = await rag.text_chunks.get_by_ids(selected)
    texts = {
        chunk_id: row
        for chunk_id, row in zip(selected, rows, strict=True)
        if row and isinstance(row.get("content"), str)
    }
    if options.chunk_id not in texts:
        raise HTTPException(404, "Chunk content is unavailable")
    blocks = {block.block_id: block for block in document.blocks}
    skip = options.offset
    count = 0
    total = sum(len(row["content"]) for row in texts.values())
    results = []
    for chunk_id, row in texts.items():
        text = row["content"]
        if skip >= len(text):
            skip -= len(text)
            continue
        location = {}
        # Identical stored chunks may be shared across documents. A document's
        # membership authorizes its text, but another document's anchors do not.
        if row.get("full_doc_id") == document_id:
            sidecar = normalize_chunk_sidecar(row)
            refs = [
                blocks[ref["id"]]
                for ref in (sidecar or {}).get("refs", [])
                if ref["type"] == "block" and ref["id"] in blocks
            ]
            if refs:
                location = document.location(
                    min(ref.start for ref in refs), max(ref.end for ref in refs)
                )
        revision = (
            _text_revision(text, location, document_revision)
            if document_revision
            else None
        )
        if options.passage:
            start, end = _passage_bounds(options.passage, revision, len(text))
        else:
            start = skip
            end = _bounded_end(text, start, len(text), options.max_chars - count)
        excerpt = text[start:end]
        skip = 0
        item = {
            "chunk_id": chunk_id,
            "text": excerpt,
            **location,
            **_citation(revision, start, end),
        }
        results.append(item)
        count += len(excerpt)
        if options.passage:
            return {"chunks": results, "next_offset": None}
        if end < len(text) or count >= options.max_chars:
            break
    end = options.offset + count
    return {"chunks": results, "next_offset": end if end < total else None}


def _validate_scope(options: ContentOptions) -> str:
    lines = options.line_start is not None or options.line_end is not None
    pages = options.page_start is not None or options.page_end is not None
    search = options.q is not None
    if options.passage:
        if lines or pages or search or options.section is not None or options.offset:
            raise HTTPException(
                422, "A passage locator can only be paired with its chunk ID."
            )
        if options.scope not in (None, "range"):
            raise HTTPException(422, "Passage locators require a range read.")
        return "range"
    exclusive = (pages, options.section is not None, options.chunk_id is not None)
    if sum(exclusive) > 1 or (any(exclusive) and (lines or search)):
        raise HTTPException(
            422,
            "Use a single page, section, or chunk selector. Keyword search can use line bounds",
        )
    if pages and options.page_start is None:
        raise HTTPException(422, "page_end requires page_start")
    selector = lines or search or any(exclusive)
    scope = options.scope or ("range" if selector else "outline")
    if scope != "range" and selector:
        raise HTTPException(422, f"Selectors are not available with scope={scope}")
    if scope == "full" and options.offset:
        raise HTTPException(422, "scope=full always returns the complete parsed text")
    return scope


async def _stream_text(text: str):
    for offset in range(0, len(text), 32768):
        yield text[offset : offset + 32768]


def register_content_route(
    router: APIRouter, rag, input_dir: Path | str | None = None
) -> None:
    input_root = Path(input_dir).resolve() if input_dir is not None else None

    @router.get(
        "/{document_id}/content",
        summary="Read a parsed document or inspect bounded source passages",
        responses={
            200: {
                "description": "JSON outline or bounded passages. With scope=full, a complete UTF-8 Markdown download.",
                "content": {
                    "application/json": {
                        "examples": {
                            "outline": {
                                "value": {
                                    "document_id": "doc-example",
                                    "scope": "outline",
                                    "selectors": ["lines", "search", "full", "section"],
                                    "line_count": 118,
                                    "page_range": None,
                                    "sections": [
                                        {
                                            "heading": "Introduction",
                                            "block_id": "example",
                                            "level": 1,
                                            "line": 1,
                                            "pages": [],
                                        }
                                    ],
                                    "next_offset": None,
                                },
                            },
                            "range": {
                                "value": {
                                    "document_id": "doc-example",
                                    "scope": "range",
                                    "segments": [
                                        {
                                            "text": "Excerpt",
                                            "line_start": 10,
                                            "line_end": 10,
                                        }
                                    ],
                                    "next_offset": 7,
                                },
                            },
                        },
                    },
                    "text/markdown": {"schema": {"type": "string"}},
                },
            },
            404: {"description": "Document, section, or chunk was not found"},
            409: {"description": "Parsed text is unavailable"},
            416: {"description": "The requested line range is outside the parsed text"},
        },
    )
    async def read_document(
        document_id: str, options: Annotated[ContentOptions, Query()]
    ):
        """Use selectors to limit context or explicitly request the complete parsed text.

        For continuation, repeat the same selectors and set `offset` to
        `next_offset`. It represents parsed lines for a keyword search,
        characters for a range, and entries for the outline.
        Pages are reported only when a verified source block carries page
        anchors. Multi-page blocks may include text from neighboring pages.
        """
        scope = _validate_scope(options)
        statuses, record = await asyncio.gather(
            rag.doc_status.get_full_docs_by_ids([document_id], strict=True),
            rag.full_docs.get_by_id(document_id),
        )
        status = statuses.get(document_id)
        if (
            status is None
            or getattr(status, "status", None) == "deleting"
            or (status.metadata or {}).get("is_duplicate")
        ):
            raise HTTPException(404, "Document not found")
        if not record or record.get("parse_format") == FULL_DOCS_FORMAT_PENDING_PARSE:
            raise HTTPException(409, "Parsed document text is not available")
        text = strip_lightrag_doc_prefix(
            record.get("content"), record.get("parse_format")
        )
        if not isinstance(text, str) or not text:
            raise HTTPException(409, "Parsed document text is not available")
        if scope == "full":
            return StreamingResponse(
                _stream_text(text),
                media_type="text/markdown",
                headers={
                    "Content-Disposition": 'attachment; filename="document.md"',
                    "X-Accel-Buffering": "no",
                },
            )

        document = await asyncio.to_thread(
            ParsedText.from_record, record, document_id, text, input_root
        )
        revision = (
            await asyncio.to_thread(_document_revision, document)
            if scope == "range" and (options.include_citations or options.passage)
            else None
        )
        if scope == "outline":
            payload = _outline(document, options, bool(status.chunks_list))
        elif options.q is not None:
            payload = await asyncio.to_thread(_find_lines, document, options, revision)
        elif options.chunk_id is not None:
            payload = await _read_chunks(
                rag, status, document_id, document, options, revision
            )
        elif options.passage:
            start, end = _passage_bounds(options.passage, revision, len(text))
            payload = {
                "segments": [
                    {
                        "text": text[start:end],
                        **document.location(start, end),
                        **_citation(revision, start, end),
                    }
                ],
                "next_offset": None,
            }
        else:
            if options.section is not None:
                regions = [_section_region(document, options.section)]
            elif options.page_start is not None:
                regions = _page_regions(
                    document, options.page_start, options.page_end or options.page_start
                )
            else:
                regions = [
                    document.bounds(
                        options.line_start or 1,
                        options.line_end or len(document.starts),
                    )
                ]
            payload = _read_regions(document, regions, options, revision)
        return {"document_id": document_id, "scope": scope, **payload}

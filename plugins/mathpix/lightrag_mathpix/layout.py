"""Associate Mathpix image crops with verified figure captions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any

from lightrag.sidecar.ir import IRDoc


_CROP = re.compile(r"-(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.I)
_IMAGE_LINE = re.compile(r"!\[[^\]]*\]\((?:<([^>]+)>|([^ )]+))\)")
_FIGURE_LABEL = re.compile(
    r"^\s*(?:(?:[*_{}]+|\\(?:textbf|textit)\{)\s*)*"
    r"(fig(?:ure)?\.?|scheme)\s+([A-Z]?\d+[a-z]?)\b",
    re.I,
)
_FIGURE_NOTE = re.compile(r"^\s*(?:footnote|note)\s+(?:to|for)\s+", re.I)
_REGION_KEYS = ("top_left_x", "top_left_y", "width", "height")
_FigureLabel = tuple[str, str]
_CropKey = tuple[int, int, int, int, int]


def read_pages(payload: bytes) -> dict[int, dict[str, Any]]:
    """Accept both a pages envelope and the per-page JSON lines download."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("pages", [data])
    if not isinstance(data, list):
        raise ValueError("Invalid Mathpix page layout")
    pages = {
        page["page"]: page
        for page in data
        if isinstance(page, dict)
        and type(page.get("page")) is int
        and type(page.get("page_height")) is int
        and isinstance(page.get("lines"), list)
    }
    if not pages:
        raise ValueError("Mathpix returned no page layout")
    return pages


def _label(text: str) -> _FigureLabel | None:
    match = _FIGURE_LABEL.match(text)
    if match:
        kind = "scheme" if match[1].lower() == "scheme" else "figure"
        return kind, match[2].lower()
    return None


def _crop(src: str) -> _CropKey | None:
    match = _CROP.search(src)
    if match:
        page, height, width, top, left = map(int, match.groups())
        if page > 0 and width > 0 and height > 0:
            return page, left, top, width, height
    return None


def _region(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    box = row.get("region")
    if not isinstance(box, dict):
        return None
    values = tuple(box.get(key) for key in _REGION_KEYS)
    if all(type(value) is int for value in values):
        left, top, width, height = values
        if width > 0 and height > 0:
            return left, top, width, height
    return None


def _text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("text_display") or "").strip()


@dataclass
class _Caption:
    label: _FigureLabel
    text: str
    region: tuple[int, int, int, int]
    ids: set[str]


def _page_captions(rows: list[dict[str, Any]]) -> list[_Caption]:
    children = defaultdict(list)
    for row in rows:
        if isinstance(row.get("parent_id"), str):
            children[row["parent_id"]].append(row)

    captions = []
    for row in rows:
        if row.get("type") != "figure_label" or (region := _region(row)) is None:
            continue
        row_id = row.get("id")
        child_rows = (
            [child for child in children.get(row_id, ()) if child.get("type") == "text"]
            if isinstance(row_id, str)
            else []
        )
        # Parent text may already contain the whole caption. Child text is
        # needed when the parent carries only geometry or only the figure label.
        parent_text = _text(row)
        child_text = " ".join(text for child in child_rows if (text := _text(child)))
        if child_text and not _label(child_text) and _label(parent_text):
            if child_text not in parent_text:
                child_text = f"{parent_text} {child_text}"
        text = max(
            (part for part in (parent_text, child_text) if _label(part)),
            key=len,
            default="",
        )
        if label := _label(text):
            ids = {r["id"] for r in (row, *child_rows) if isinstance(r.get("id"), str)}
            captions.append(_Caption(label, text, region, ids))
    return captions


def _markdown_captions(markdown: str) -> dict[_CropKey, set[_FigureLabel]]:
    lines = [line for line in markdown.splitlines() if line.strip()]
    matches = defaultdict(set)
    for line, following in zip(lines, lines[1:]):
        if (label := _label(following)) is None:
            continue
        for match in _IMAGE_LINE.finditer(line):
            if crop := _crop(match[1] or match[2]):
                matches[crop].add(label)
    return matches


def attach_figure_context(
    ir: IRDoc, markdown: str, pages: dict[int, dict[str, Any]]
) -> None:
    """Require exact crop geometry and corroborating Markdown for each caption."""
    markdown_labels = _markdown_captions(markdown)
    drawings = defaultdict(list)
    for block in ir.blocks:
        for drawing in block.drawings:
            if key := _crop(drawing.src):
                drawings[key[0]].append((key, drawing))

    for page_number, page_drawings in drawings.items():
        page = pages.get(page_number)
        if not page:
            continue
        rows = [row for row in page["lines"] if isinstance(row, dict)]
        regions = defaultdict(list)
        for row in rows:
            if row.get("type") in ("diagram", "chart"):
                regions[_region(row)].append(row)
        captions = _page_captions(rows)
        notes = [row for row in rows if row.get("type") == "footnote" and _text(row)]
        associations = {}
        for key, _ in page_drawings:
            _, left, top, width, height = key
            image_rows = regions[(left, top, width, height)]
            if len(image_rows) != 1:
                continue
            nearby = []
            for cap in captions:
                x, y, w, _ = cap.region
                overlap = min(left + width, x + w) - max(left, x)
                gap = y - top - height
                if overlap >= 0.75 * min(width, w) and 0 <= gap <= min(
                    155, page["page_height"] * 0.06
                ):
                    nearby.append(cap)
            if len(nearby) == 1 and markdown_labels.get(key) == {nearby[0].label}:
                associations[key] = (nearby[0], image_rows[0])

        label_counts = Counter(cap.label for cap, _ in associations.values())
        for key, drawing in page_drawings:
            if key not in associations:
                continue
            cap, image = associations[key]
            if label_counts[cap.label] != 1:
                continue
            drawing.caption = cap.text
            image_id = image.get("id")
            ids = cap.ids | ({image_id} if isinstance(image_id, str) else set())
            for row in notes:
                text = _text(row)
                explicit = _FIGURE_NOTE.match(text)
                parent = row.get("parent_id")
                if (isinstance(parent, str) and parent in ids) or (
                    explicit and _label(text[explicit.end() :]) == cap.label
                ):
                    if text not in drawing.footnotes:
                        drawing.footnotes.append(text)

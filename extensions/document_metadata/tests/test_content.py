"""Behavior of the document content API against scoped, saved records."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from extensions.document_metadata.content import register_content_route

pytestmark = pytest.mark.offline


def _document(record, *, input_dir=None, chunks=(), status="processed"):
    saved = SimpleNamespace(chunks_list=list(chunks), status=status, metadata={})
    rag = SimpleNamespace(
        doc_status=SimpleNamespace(
            get_full_docs_by_ids=AsyncMock(return_value={"doc-test": saved})
        ),
        full_docs=SimpleNamespace(get_by_id=AsyncMock(return_value=record)),
        text_chunks=SimpleNamespace(get_by_ids=AsyncMock()),
    )
    app = FastAPI()
    router = APIRouter(prefix="/documents")
    register_content_route(router, rag, input_dir)
    app.include_router(router)
    return TestClient(app), rag


def _sidecar(root, blocks, doc_id="doc-test"):
    directory = root / "__parsed__" / "manual.pdf.parsed"
    directory.mkdir(parents=True)
    rows = [{"type": "meta", "doc_id": doc_id}]
    rows.extend(
        {
            "type": "content",
            "blockid": f"block-{index}",
            "heading": heading,
            "level": level,
            "content": text,
            "positions": [{"type": "bbox", "anchor": page} for page in pages],
        }
        for index, (heading, level, text, pages) in enumerate(blocks)
    )
    (directory / "manual.blocks.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    return directory.as_uri()


def test_outline_section_pages_and_continuation_use_verified_source_blocks(tmp_path):
    first, second, third = "Overview. " * 32, "Test settings", "A new chapter"
    blocks = [
        ("Parent", 1, first, ["2"]),
        ("Child", 2, second, ["3", "4"]),
        ("Next", 1, third, [5]),
    ]
    uri = _sidecar(tmp_path, blocks)
    body = "\n\n".join((first, second, third))
    client, _ = _document(
        {
            "parse_format": "lightrag",
            "content": "{{LRdoc}}" + body,
            "sidecar_location": uri,
        },
        input_dir=tmp_path,
    )

    outline = client.get("/documents/doc-test/content", params={"limit": 1}).json()
    assert outline["page_range"] == [2, 5]
    assert outline["sections"] == [
        {
            "block_id": "block-0",
            "heading": "Parent",
            "level": 1,
            "line": 1,
            "pages": [2],
        }
    ]
    assert outline["next_offset"] == 1

    first_part = client.get(
        "/documents/doc-test/content",
        params={"section": "block-0", "max_chars": 256},
    ).json()
    second_part = client.get(
        "/documents/doc-test/content",
        params={"section": "block-0", "offset": first_part["next_offset"]},
    ).json()
    assert first_part["next_offset"] == 256
    assert (
        first_part["segments"][0]["text"] + second_part["segments"][0]["text"]
        == first + "\n\n" + second + "\n\n"
    )
    assert second_part["next_offset"] is None

    page = client.get("/documents/doc-test/content", params={"page_start": 4}).json()
    assert page["segments"] == [
        {
            "text": second,
            "line_start": 3,
            "line_end": 3,
            "pages": [3, 4],
            "page_precision": "block",
            "section": "Child",
        }
    ]


@pytest.mark.parametrize("cause", ["other-workspace", "different-id", "stale-text"])
def test_page_anchors_require_matching_body_and_confined_sidecar(tmp_path, cause):
    root = tmp_path / "this"
    source_root = tmp_path / "other" if cause == "other-workspace" else root
    uri = _sidecar(
        source_root,
        [("Heading", 1, "saved body", [12])],
        doc_id="doc-other" if cause == "different-id" else "doc-test",
    )
    body = "different body" if cause == "stale-text" else "saved body"
    client, _ = _document(
        {
            "parse_format": "lightrag",
            "content": "{{LRdoc}}" + body,
            "sidecar_location": uri,
        },
        input_dir=root,
    )

    outline = client.get("/documents/doc-test/content").json()
    assert outline["page_range"] is None
    assert "page" not in outline["selectors"]
    refusal = client.get("/documents/doc-test/content", params={"page_start": 12})
    assert refusal.status_code == 422
    assert "no page anchors" in refusal.json()["detail"]


def test_line_regex_supports_anchors_and_resumes_from_next_match():
    client, _ = _document(
        {"parse_format": "raw", "content": "header\nHeat 7\nlook here\nHEAT 8\ntail"}
    )
    options = {"q": r"^heat \d$", "match": "regex", "context_lines": 1, "limit": 1}

    first = client.get("/documents/doc-test/content", params=options).json()
    assert first["matches"] == [
        {
            "line": 2,
            "column": 1,
            "line_start": 1,
            "line_end": 3,
            "text": "header\nHeat 7\nlook here\n",
        }
    ]
    assert first["next_offset"] == 3

    second = client.get(
        "/documents/doc-test/content",
        params={**options, "offset": first["next_offset"]},
    ).json()
    assert second["matches"][0]["line"] == 4
    assert second["matches"][0]["text"] == "look here\nHEAT 8\ntail"
    assert second["next_offset"] is None


@pytest.mark.parametrize(
    ("parse_format", "expected_prefix"), [("lightrag", ""), ("raw", "{{LRdoc}}")]
)
def test_full_streaming_download_is_complete_and_only_strips_internal_markers(
    parse_format, expected_prefix
):
    body = "铅酸蓄电池。\n" * 6000
    client, _ = _document({"parse_format": parse_format, "content": "{{LRdoc}}" + body})

    result = client.get(
        "/documents/doc-test/content", params={"scope": "full", "max_chars": 256}
    )

    assert result.status_code == 200
    assert result.headers["content-type"] == "text/markdown; charset=utf-8"
    assert result.headers["content-disposition"] == 'attachment; filename="document.md"'
    assert result.text == expected_prefix + body


def test_chunk_lookup_requires_membership_and_reads_only_requested_neighbors():
    client, rag = _document(
        {"parse_format": "raw", "content": "before body after"},
        chunks=["chunk-before", "chunk-centre", "chunk-after", "chunk-outside"],
    )
    rag.text_chunks.get_by_ids.return_value = [
        {"content": "before"},
        {"content": "body"},
        {"content": "after"},
    ]
    found = client.get(
        "/documents/doc-test/content", params={"chunk_id": "chunk-centre"}
    ).json()
    assert found["chunks"] == [
        {"chunk_id": "chunk-before", "text": "before"},
        {"chunk_id": "chunk-centre", "text": "body"},
        {"chunk_id": "chunk-after", "text": "after"},
    ]
    assert found["next_offset"] is None

    foreign = client.get(
        "/documents/doc-test/content", params={"chunk_id": "chunk-foreign"}
    )
    assert foreign.status_code == 404
    rag.text_chunks.get_by_ids.assert_awaited_once_with(
        ["chunk-before", "chunk-centre", "chunk-after"]
    )

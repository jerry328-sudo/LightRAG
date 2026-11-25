from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extensions.document_metadata.query import attach_query_document_ids


pytestmark = pytest.mark.offline


async def test_query_documents_follow_stored_chunk_ownership():
    response = {
        "data": {
            "chunks": [
                {"chunk_id": "c1", "file_path": "273.pdf", "reference_id": "1"},
                {"chunk_id": "c2", "file_path": "273.pdf", "reference_id": "2"},
                {"chunk_id": "c2", "file_path": "281.pdf", "reference_id": "3"},
                {"chunk_id": "missing", "file_path": "099.pdf", "reference_id": "4"},
            ],
            "references": [{"reference_id": str(i)} for i in range(1, 5)],
        }
    }
    lookup = AsyncMock(
        return_value=[
            {"id": "c2", "file_path": "281.pdf", "full_doc_id": "doc-b"},
            None,
            {"id": "c1", "file_path": "273.pdf", "full_doc_id": "doc-a"},
        ]
    )
    retrieve = AsyncMock(return_value=response)
    rag = SimpleNamespace(
        aquery_data=retrieve, text_chunks=SimpleNamespace(get_by_ids=lookup)
    )

    attach_query_document_ids(rag)
    data = (await rag.aquery_data("battery aging", param=object()))["data"]

    retrieve.assert_awaited_once()
    lookup.assert_awaited_once_with(["c1", "c2", "missing"])
    assert [chunk.get("document_id") for chunk in data["chunks"]] == [
        "doc-a",
        None,
        "doc-b",
        None,
    ]
    assert [reference.get("document_id") for reference in data["references"]] == [
        "doc-a",
        None,
        "doc-b",
        None,
    ]

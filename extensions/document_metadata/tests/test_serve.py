import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest

from extensions.document_metadata.serve import attach_document_metadata


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_page_merges_saved_fields_without_changing_processing_metadata():
    original = SimpleNamespace(metadata={"parse_engine": "mathpix"})
    no_metadata = SimpleNamespace(metadata={"parse_engine": "native"})
    fetch = AsyncMock(return_value=([("doc-2", original), ("doc-9", no_metadata)], 99))
    saved = {
        "document_type": "paper",
        "title": "Grounded title",
        "contributors": [
            {"name": f"Author {index}", "role": "author"} for index in range(8)
        ],
        "identifiers": [{"type": "doi", "value": "10.0000/verified"}],
        "summary": "电池寿命" * 100,
        "domain_details": {"datasets": [{"name": "Large source"}]},
        "evidence": [{"claim": "Full evidence"}],
    }
    query = AsyncMock(
        return_value=[
            {
                "id": "doc-2",
                "document_metadata": json.dumps(saved),
            }
        ]
    )
    storage = SimpleNamespace(
        workspace="leadacid", get_docs_paginated=fetch, db=SimpleNamespace(query=query)
    )

    attach_document_metadata(storage)
    documents, total = await storage.get_docs_paginated(page=2)

    fetch.assert_awaited_once_with(page=2)
    workspace, ids, fields = query.await_args.args[1]
    assert workspace == "leadacid"
    assert ids == ["doc-2", "doc-9"]
    assert {"title", "contributors", "overall_confidence"} <= set(fields)
    assert {"domain_details", "evidence"}.isdisjoint(fields)
    assert total == 99
    assert [doc_id for doc_id, _ in documents] == ["doc-2", "doc-9"]
    assert documents[0][1].metadata == {
        "parse_engine": "mathpix",
        "document_metadata": {
            "document_type": "paper",
            "title": "Grounded title",
            "contributors": [*(f"Author {index}" for index in range(6)), "+2"],
            "identifiers": ["10.0000/verified"],
            "summary": saved["summary"][:239] + "…",
        },
    }
    assert original.metadata == {"parse_engine": "mathpix"}
    assert documents[1][1].metadata == {"parse_engine": "native"}


@pytest.mark.asyncio
async def test_extension_read_failure_still_serves_original_page():
    original = SimpleNamespace(metadata={"parse_engine": "mathpix"})
    fetch = AsyncMock(return_value=([("doc-2", original)], 1))
    query = AsyncMock(side_effect=asyncpg.UndefinedColumnError())
    storage = SimpleNamespace(
        workspace="default", get_docs_paginated=fetch, db=SimpleNamespace(query=query)
    )

    attach_document_metadata(storage)
    documents, total = await storage.get_docs_paginated()

    query.assert_awaited_once()
    assert documents == [("doc-2", original)]
    assert total == 1

from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from extensions.document_metadata.enrich import enrich, enrich_record, validate_batch


pytestmark = pytest.mark.offline


def entry():
    return {
        "workspace": "default",
        "document_id": "doc-0123",
        "file_path": "007.pdf",
        "fields": {
            "published_date": "2024-02-29",
            "title": "Registry title",
            "type_details.journal": "Verified Journal",
        },
        "sources": {
            field: {"kind": "doi_registry", "locator": "https://doi.org/10.5555/123"}
            for field in ("published_date", "title", "type_details.journal")
        },
    }


def test_enrichment_keeps_existing_values_and_assessment_provenance():
    original = {
        "title": "Stored title",
        "published_date": None,
        "type_details": {"volume": "42"},
        "assessment_source": {"kind": "okf_json"},
    }
    result, fields = enrich_record(original, entry())
    assert fields == ["published_date", "type_details.journal"]
    assert result == {
        **original,
        "published_date": "2024-02-29",
        "type_details": {"volume": "42", "journal": "Verified Journal"},
        "field_sources": {
            "published_date": entry()["sources"]["published_date"],
            "type_details": {"journal": entry()["sources"]["type_details.journal"]},
        },
    }
    assert original["type_details"] == {"volume": "42"}
    assert enrich_record(result, entry()) == (result, [])


@pytest.mark.parametrize("date_value", ["", "2025-02-29", "0000", "2025-1"])
def test_invalid_calendar_dates_fail_validation(date_value):
    document = entry()
    document["fields"]["published_date"] = date_value
    with pytest.raises(ValueError, match="published_date"):
        validate_batch({"records": [document]})


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.asyncio
async def test_write_is_bound_to_the_document_and_original_record(dry_run):
    @asynccontextmanager
    async def transaction(*, readonly):
        assert readonly is dry_run
        yield

    original = json.dumps(
        {"title": "Local title", "type_details": {"journal": "Local journal"}}
    )
    db = MagicMock(transaction=transaction)
    db.fetchval = AsyncMock(side_effect=[original, "doc-0123"])
    result = await enrich(db, "default", {"records": [entry()]}, dry_run=dry_run)
    assert result == {
        "workspace": "default",
        "dry_run": dry_run,
        "requested_documents": 1,
        "documents_with_additions": 1,
        "fields_added": {"published_date": 1},
    }
    assert db.fetchval.await_count == (1 if dry_run else 2)
    assert db.fetchval.await_args_list[0].args[1:] == ("default", "doc-0123", "007.pdf")
    if not dry_run:
        query, *args = db.fetchval.await_args.args
        assert "echem_document_metadata = $4::jsonb" in query
        assert args[:4] == ["default", "doc-0123", "007.pdf", original]
        assert json.loads(args[4])["published_date"] == "2024-02-29"


@pytest.mark.asyncio
async def test_concurrent_change_fails_loudly():
    @asynccontextmanager
    async def transaction(*, readonly):
        assert readonly is False
        yield

    db = MagicMock(transaction=transaction)
    db.fetchval = AsyncMock(side_effect=["{}", None])
    with pytest.raises(RuntimeError, match="Metadata changed during enrichment"):
        await enrich(db, "default", {"records": [entry()]}, dry_run=False)
    assert db.fetchval.await_count == 2

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from extensions.document_metadata.documents import register_document_routes


pytestmark = pytest.mark.offline


def _setup():
    connection = SimpleNamespace(
        execute=AsyncMock(), fetchval=AsyncMock(), fetch=AsyncMock(return_value=[])
    )

    @asynccontextmanager
    async def transaction(*, readonly):
        assert readonly
        yield

    connection.transaction = transaction

    async def process(callback):
        return await callback(connection)

    db = SimpleNamespace(query=AsyncMock(), _run_with_retry=process)
    rag = SimpleNamespace(doc_status=SimpleNamespace(db=db, workspace="workspace-2"))
    app = FastAPI()
    router = APIRouter(prefix="/documents")
    register_document_routes(router, rag)
    app.include_router(router)
    return TestClient(app), connection, db


def test_search_returns_bounded_source_matches_and_page_cursor():
    client, connection, _ = _setup()
    connection.fetch.return_value = [
        {
            "id": "doc-88",
            "file_path": "example.pdf",
            "metadata": '{"title":"Warm cells"}',
            "body_snippet": "The value 25.0 was measured at 25 °C.",
            "body_line": 57,
            "match_field": "summary",
            "metadata_snippet": "Measurements for 25.0 cycles",
        },
        {"id": "doc-95"},
    ]
    response = client.post(
        "/documents/search",
        json={
            "query": "25.0",
            "document_ids": ["doc-88", "doc-95"],
            "filters": {
                "title": "Warm",
                "author": "Smith",
                "doi": "10.4444/valid",
                "document_type": "paper",
                "year": 2030,
                "research_task": "RUL_prediction",
                "overall_confidence": "high",
            },
            "cursor": "doc-77",
            "limit": 1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["next_cursor"] == "doc-88"
    assert len(payload["documents"]) == 1
    hit = payload["documents"][0]
    assert hit["document_id"] == "doc-88"
    assert hit["metadata"] == {"title": "Warm cells"}
    assert [
        (match["source"], match["line"], match["field"]) for match in hit["matches"]
    ] == [
        ("content", 57, None),
        ("metadata", None, "summary"),
    ]
    assert connection.fetch.await_args.args[1:] == (
        "workspace-2",
        "doc-77",
        ["doc-88", "doc-95"],
        "25.0",
        "qi",
        True,
        True,
        "Warm",
        "Smith",
        "10.4444/valid",
        "paper",
        "2030",
        "RUL_prediction",
        "high",
        2,
    )
    connection.fetchval.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [(asyncpg.InvalidRegularExpressionError, 422), (asyncpg.QueryCanceledError, 408)],
)
def test_search_bounds_database_regex_and_maps_rejected_patterns(
    failure, expected_code
):
    client, connection, _ = _setup()
    connection.fetchval.side_effect = failure
    response = client.post(
        "/documents/search", json={"query": "(.*)+", "match": "regex"}
    )
    assert response.status_code == expected_code
    connection.execute.assert_awaited_once_with(
        "SET LOCAL statement_timeout = '3000ms'"
    )
    assert connection.fetchval.await_args.args[1:] == ("(.*)+", "ni")
    connection.fetch.assert_not_called()


def test_metadata_discovery_and_projection_preserve_only_stored_values():
    client, _, db = _setup()
    metadata = {
        "title": "Temperature experiment",
        "domain_details": {"evaluation": {"groups": 3}, "datasets": [{"name": "SNL"}]},
        "evidence": [{"source_location": {"page": 9}}],
        "overall_confidence": None,
    }
    db.query.return_value = {"file_path": "temperature.pdf", "metadata": metadata}
    url = "/documents/doc-94/metadata"
    full = client.get(url)
    assert full.status_code == 200 and full.json()["metadata"] == metadata
    fields = client.get(url, params={"view": "fields"})
    assert fields.status_code == 200
    assert "metadata" not in fields.json()
    assert {field["path"]: field["type"] for field in fields.json()["fields"]} == {
        "title": "string",
        "domain_details": "object",
        "domain_details.evaluation": "object",
        "domain_details.evaluation.groups": "integer",
        "domain_details.datasets": "array",
        "evidence": "array",
        "overall_confidence": "null",
    }
    projected = client.get(
        url,
        params=[
            ("fields", "domain_details.evaluation.groups"),
            ("fields", "evidence"),
            ("fields", "identifiers"),
        ],
    )
    assert projected.status_code == 200
    assert projected.json()["metadata"] == {
        "domain_details": {"evaluation": {"groups": 3}},
        "evidence": [{"source_location": {"page": 9}}],
    }
    assert db.query.await_args.args[1] == ["workspace-2", "doc-94"]


def test_metadata_distinguishes_unknown_document_from_unfilled_record():
    client, _, db = _setup()
    db.query.side_effect = [None, {"file_path": "unfilled.pdf", "metadata": None}]
    assert client.get("/documents/doc-7/metadata").status_code == 404
    unfilled = client.get("/documents/doc-9/metadata")
    assert unfilled.status_code == 200
    assert unfilled.json() == {
        "file_path": "unfilled.pdf",
        "document_id": "doc-9",
        "metadata": None,
    }

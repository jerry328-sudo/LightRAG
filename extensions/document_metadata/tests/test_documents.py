from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from extensions.document_metadata import documents
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
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
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


def _search_row(document_id, metadata):
    return {
        "id": document_id,
        "file_path": f"{document_id}.pdf",
        "metadata": metadata,
        "body_snippet": None,
        "metadata_snippet": None,
    }


def test_search_projects_metadata_of_multiple_documents_and_passes_structured_filters():
    client, connection, _ = _setup()
    connection.fetch.return_value = [
        _search_row(
            "doc-41",
            {
                "title": "Battery results",
                "type_details": {"journal": "Energy Journal", "volume": "3"},
                "field_sources": {
                    "published_date": {"source": "publisher"},
                    "language": {},
                },
                "evidence_strength": "strong",
            },
        ),
        _search_row(
            "doc-42",
            {"title": "Transfer study", "source_urls": ["https://example.com"]},
        ),
    ]
    response = client.post(
        "/documents/search",
        json={
            "document_ids": ["doc-41", "doc-42"],
            "fields": ["title", "type_details.journal", "field_sources.published_date"],
            "filters": {
                "document_subtype": "research_article",
                "evidence_strength": "strong",
                "year_from": 2020,
                "year_to": 2024,
                "published_from": "2020-01-01",
                "published_to": "2024-04-02",
                "cross_domain_validation": False,
                "method_tag": "zero-shot learning",
                "generalization_target": "cross_dataset",
            },
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {
                "document_id": "doc-41",
                "file_path": "doc-41.pdf",
                "metadata": {
                    "title": "Battery results",
                    "type_details": {"journal": "Energy Journal"},
                    "field_sources": {"published_date": {"source": "publisher"}},
                },
                "matches": [],
            },
            {
                "document_id": "doc-42",
                "file_path": "doc-42.pdf",
                "metadata": {"title": "Transfer study"},
                "matches": [],
            },
        ],
        "next_cursor": None,
    }
    args = connection.fetch.await_args.args
    assert args[1:8] == (
        "workspace-2",
        None,
        ["doc-41", "doc-42"],
        None,
        "qi",
        False,
        False,
    )
    assert args[16:] == (
        True,
        "research_article",
        "2020",
        "2024",
        "2020-01-01",
        "2024-04-02",
        "strong",
        False,
        "zero-shot learning",
        "cross_dataset",
    )


def test_compact_catalogue_allows_200_rows_and_retains_cursor():
    client, connection, _ = _setup()
    row_metadata = {"summary": "z" * 800, "document_type": "paper", "evidence": [1]}
    connection.fetch.return_value = [
        _search_row(f"doc-{number:03d}", row_metadata) for number in range(1, 202)
    ]
    response = client.post(
        "/documents/search", json={"fields": ["summary", "document_type"], "limit": 200}
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["documents"]) == 200 and payload["next_cursor"] == "doc-200"
    assert payload["documents"][0]["metadata"] == {
        "summary": "z" * 512,
        "document_type": "paper",
    }
    assert connection.fetch.await_args.args[15] == 201


def test_filter_only_catalogue_supports_100_rich_rows_and_year_aliases():
    client, connection, _ = _setup()
    projected = {
        "title": "Capacity results",
        "summary": "Study summary",
        "domain_details": {"key_contributions": ["Validates the forecast"]},
        "limitations": {"scope_limitations": ["Single test site"]},
    }
    source = {
        **projected,
        "id": "not-the-document-id",
        "evidence_strength": "strong",
        "domain_details": {**projected["domain_details"], "discussion": "omitted"},
    }
    connection.fetch.return_value = [
        _search_row(f"doc-{number:03d}", source) for number in range(1, 102)
    ]
    response = client.post(
        "/documents/search",
        json={
            "filters": {
                "year_min": 2024,
                "year_max": 2026,
                "evidence_strength": "strong",
                "research_task": "RUL_prediction",
            },
            "fields": [
                "id",
                "title",
                "summary",
                "domain_details.key_contributions",
                "limitations",
            ],
            "limit": 100,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["documents"]) == 100
    assert payload["next_cursor"] == "doc-100"
    assert payload["documents"][0] == {
        "document_id": "doc-001",
        "file_path": "doc-001.pdf",
        "metadata": projected,
        "matches": [],
    }
    args = connection.fetch.await_args.args
    assert (args[13], args[15], args[18], args[19], args[22]) == (
        "RUL_prediction",
        101,
        "2024",
        "2026",
        "strong",
    )


def test_projected_catalogue_byte_budget_emits_cursor_without_skipping_row(monkeypatch):
    client, connection, _ = _setup()
    monkeypatch.setattr(documents, "_PROJECTED_PAGE_MAX_BYTES", 950)
    text = "段" * 180
    metadata = {"limitations": {"scope_limitations": [text]}}
    first, second = _search_row("doc-01", metadata), _search_row("doc-02", metadata)
    connection.fetch.side_effect = [[first, second], [second]]
    body = {"fields": ["limitations"], "limit": 100}
    page = client.post("/documents/search", json=body)
    assert page.status_code == 200 and len(page.content) < 950
    assert len(page.json()["documents"]) == 1
    assert page.json()["documents"][0]["metadata"] == metadata
    assert page.json()["next_cursor"] == "doc-01"
    next_page = client.post(
        "/documents/search", json={**body, "cursor": page.json()["next_cursor"]}
    )
    assert [row["document_id"] for row in next_page.json()["documents"]] == ["doc-02"]
    assert next_page.json()["next_cursor"] is None
    assert connection.fetch.await_args.args[2] == "doc-01"


def test_projected_catalogue_rejects_single_row_above_byte_budget(monkeypatch):
    client, connection, _ = _setup()
    monkeypatch.setattr(documents, "_PROJECTED_PAGE_MAX_BYTES", 500)
    connection.fetch.return_value = [_search_row("doc-01", {"evidence": ["x" * 700]})]
    response = client.post("/documents/search", json={"fields": ["evidence"]})
    assert response.status_code == 413
    assert "Select fewer fields" in response.json()["detail"]
    connection.fetch.assert_awaited_once()


@pytest.mark.parametrize(
    "body",
    [
        {"limit": 51},
        {"limit": 51, "query": "text", "fields": ["title"]},
        {"limit": 201, "fields": ["evidence"]},
        {"fields": ["type_details..journal"]},
        {"filters": {"published_from": "2024-02-30"}},
        {"filters": {"year_from": 2024, "year_to": 2023}},
        {"filters": {"year_from": 2024, "year_min": 2023}},
        {"filters": {"year_to": 2024, "year_max": 2025}},
        {"filters": {"year_min": 2024, "year_max": 2023}},
    ],
)
def test_document_search_rejects_invalid_bounds_and_unbounded_pages(body):
    client, connection, _ = _setup()
    assert client.post("/documents/search", json=body).status_code == 422
    connection.fetch.assert_not_awaited()

import json
from unittest.mock import AsyncMock

import pytest

from extensions.document_metadata import worker


pytestmark = pytest.mark.offline


def test_parsed_text_preserves_body_and_only_adds_successful_workspace_analysis(
    tmp_path,
):
    inputs = tmp_path / "inputs"
    directory = inputs / "__parsed__" / "report.parsed"
    directory.mkdir(parents=True)
    (directory / "report.blocks.jsonl").touch()
    (directory / "report.tables.json").write_text(
        json.dumps(
            {
                "tables": {
                    "table-a": {
                        "heading": "Results",
                        "llm_analyze_result": {
                            "status": "success",
                            "description": "Retained analysis",
                        },
                    },
                    "table-b": {
                        "llm_analyze_result": {
                            "status": "error",
                            "description": "Unusable analysis",
                        }
                    },
                }
            }
        )
    )
    body = "Full beginning.\n\nFull end."
    doc = {
        "content": "{{LRdoc}}" + body,
        "parse_format": "lightrag",
        "sidecar_location": directory.as_uri(),
    }

    result = worker.parsed_text(doc, inputs)

    assert result.startswith(body)
    assert "Retained analysis" in result
    assert "Unusable analysis" not in result
    assert worker.parsed_text(doc, inputs / "otherworkspace") == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "previous_attempts, expected_status", [(0, "pending"), (2, "failed")]
)
async def test_failed_extraction_records_distinct_retry_status(
    monkeypatch, tmp_path, previous_attempts, expected_status
):
    save = AsyncMock()
    complete = AsyncMock(side_effect=RuntimeError("raw provider error"))
    monkeypatch.setattr(worker.store, "finish_document", save)
    doc = {
        "id": "doc-88",
        "content": "parsed text",
        "parse_format": "raw",
        "sidecar_location": None,
        "attempts": previous_attempts,
    }

    await worker.handle_document(None, "workspace", doc, tmp_path, complete, "model")

    complete.assert_awaited_once()
    save.assert_awaited_once()
    _db, workspace, saved_doc, metadata, job = save.await_args.args
    assert workspace == "workspace"
    assert saved_doc == doc
    assert metadata is None
    assert job["status"] == expected_status
    assert job["attempts"] == previous_attempts + 1
    assert job["error"] == "RuntimeError"
    assert ("after" in job) is (expected_status == "pending")

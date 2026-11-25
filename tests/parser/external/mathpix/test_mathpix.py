"""Offline tests of the installable Mathpix parser plugin."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock
from zipfile import ZipFile

import httpx
import pytest

from lightrag import pipeline
from lightrag.parser import registry
from lightrag.parser.base import ParseContext
from lightrag.parser.debug import build_debug_rag
from lightrag.parser.exceptions import ParsePipelineCancelled
from lightrag.parser.routing import validate_parser_routing_config
from lightrag.utils_pipeline import resolve_sidecar_uri

pytestmark = pytest.mark.offline

PDF_ID = "mathpix-fixture-4"
PLUGIN = Path(__file__).resolve().parents[4] / "plugins" / "mathpix"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+aoh0AAAAASUVORK5CYII="
)
CROP_LEFT = "sample-2_50_400_100_100.png"
CROP_RIGHT = "sample-2_61_400_100_610.png"


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.syspath_prepend(str(PLUGIN))
    monkeypatch.setenv("MATHPIX_APP_ID", "test-only-app-id")
    monkeypatch.setenv("MATHPIX_APP_KEY", "test-only-app-key")
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
    monkeypatch.setattr(registry, "_INSTANCE_CACHE", {})
    from lightrag_mathpix import register

    register()
    return registry.get_parser("mathpix")


@pytest.fixture
def document(tmp_path, monkeypatch):
    archive = AsyncMock()
    monkeypatch.setattr(pipeline, "archive_docx_source_after_full_docs_sync", archive)
    pdf = tmp_path / "battery-study.pdf"
    pdf.write_bytes(b"synthetic PDF payload")
    rag = build_debug_rag()
    rag._resolve_source_file_for_parser = lambda _: str(pdf)
    ctx = ParseContext(
        rag,
        "doc-mathpix-test",
        "battery-study.pdf",
        {"parse_engine": "mathpix"},
        pipeline_cancel_event=Event(),
    )
    return ctx, pdf, archive


@pytest.fixture
def bundle():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "result/battery-study.md",
            "# Capacity evidence\n\nThe initial capacity was measured twice.\n\n"
            "| Material | Cycles |\n| --- | --- |\n| Graphite | 90 |\n\n"
            "$$\nx^2 + y^2 = z^2\n$$\n\n"
            f"![](images/{CROP_LEFT})\n\n**Figure 7.** Cycling performance.\n\n"
            f"![](images/{CROP_RIGHT})\n\n**Figure 9.** Impedance response.\n",
        )
        archive.writestr(f"result/images/{CROP_LEFT}", PNG)
        archive.writestr(f"result/images/{CROP_RIGHT}", PNG)
    return buffer.getvalue()


@pytest.fixture
def page_layout():
    def region(left, top, width, height):
        return dict(top_left_x=left, top_left_y=top, width=width, height=height)

    return {
        "page": 2,
        "page_height": 1000,
        "lines": [
            {"type": "chart", "id": "left", "region": region(100, 100, 400, 50)},
            {
                "type": "figure_label",
                "id": "left-label",
                "region": region(100, 160, 380, 18),
            },
            {
                "type": "text",
                "parent_id": "left-label",
                "text": "Figure 7. Cycling performance.",
            },
            {"type": "chart", "id": "right", "region": region(610, 100, 400, 61)},
            {
                "type": "figure_label",
                "text": "Figure 9. Impedance response.",
                "region": region(610, 172, 380, 18),
            },
            {
                "type": "footnote",
                "parent_id": "left-label",
                "text": "The capacity is normalized by mass.",
            },
            {"type": "footnote", "text": "An unrelated page note."},
        ],
    }


@pytest.fixture
def mathpix_api(monkeypatch, engine, bundle, page_layout):
    from lightrag_mathpix.client import API_URL, MathpixClient

    calls = []
    scenarios = {}

    async def fake_api(request):
        assert request.url.host == "api.mathpix.com"
        assert request.headers["app_id"] == "test-only-app-id"
        assert request.headers["app_key"] == "test-only-app-key"
        call = (request.method, request.url.path)
        calls.append(call)

        if call == ("POST", "/v3/pdf"):
            body = await request.aread()
            assert b'filename="battery-study.pdf"' in body
            assert b'"md.zip": true' in body
            return httpx.Response(200, json={"pdf_id": PDF_ID})
        if call == ("GET", f"/v3/pdf/{PDF_ID}"):
            event = scenarios.pop("cancel_on_first_poll", None)
            if event is not None:
                event.set()
                return httpx.Response(200, json={"status": "split"})
            return httpx.Response(200, json={"status": "completed"})
        if call == ("GET", f"/v3/converter/{PDF_ID}"):
            state = scenarios.pop("conversion_state", "completed")
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "conversion_status": {"md.zip": {"status": state}},
                },
            )
        if call == ("GET", f"/v3/pdf/{PDF_ID}.lines.json"):
            return scenarios.pop("layout_response", None) or httpx.Response(
                200,
                content='{"page":1,"page_height":1000,"lines":[]}\n'
                + json.dumps(page_layout),
            )
        assert call == ("GET", f"/v3/pdf/{PDF_ID}.md.zip")
        return httpx.Response(200, content=bundle)

    monkeypatch.setattr(
        MathpixClient,
        "open",
        lambda service: httpx.AsyncClient(
            base_url=API_URL,
            headers=service.headers,
            transport=httpx.MockTransport(fake_api),
        ),
    )
    return calls, scenarios


def test_routes_pdf_to_registered_plugin_only_with_both_credentials(
    engine, monkeypatch
):
    rules = "pdf:mathpix-iteP,*:legacy-R"
    validate_parser_routing_config(rules)
    monkeypatch.delenv("MATHPIX_APP_KEY")
    with pytest.raises(ValueError, match="MATHPIX_APP_ID and MATHPIX_APP_KEY"):
        validate_parser_routing_config(rules)


@pytest.mark.asyncio
async def test_ingests_paper_sidecars_and_reuses_its_bundle(
    engine, document, mathpix_api
):
    ctx, pdf, archive = document
    calls, _ = mathpix_api
    first = await engine.parse(ctx)
    record = await ctx.rag.full_docs.get_by_id(ctx.doc_id)
    directory = Path(first.blocks_path).parent
    assert first.parse_format == "lightrag"
    assert first.parse_engine == "mathpix"
    assert "The initial capacity was measured twice." in record["content"]
    assert record["parse_engine"] == "mathpix"
    assert resolve_sidecar_uri(record["sidecar_location"]) == directory
    assert "Graphite" in (directory / "battery-study.tables.json").read_text()
    assert "x^2 + y^2 = z^2" in (directory / "battery-study.equations.json").read_text()
    drawing_items = json.loads((directory / "battery-study.drawings.json").read_text())[
        "drawings"
    ]
    drawings = {Path(item["src"]).name: item for item in drawing_items.values()}
    assert {
        name: (item["caption"], item["footnotes"]) for name, item in drawings.items()
    } == {
        CROP_LEFT: (
            "Figure 7. Cycling performance.",
            ["The capacity is normalized by mass."],
        ),
        CROP_RIGHT: ("Figure 9. Impedance response.", []),
    }
    assert [
        p.read_bytes() for p in (directory / "battery-study.blocks.assets").iterdir()
    ] == [PNG]
    archive.assert_awaited_once_with(str(pdf))
    assert calls == [
        ("POST", "/v3/pdf"),
        ("GET", f"/v3/pdf/{PDF_ID}"),
        ("GET", f"/v3/converter/{PDF_ID}"),
        ("GET", f"/v3/pdf/{PDF_ID}.md.zip"),
        ("GET", f"/v3/pdf/{PDF_ID}.lines.json"),
    ]

    second = await engine.parse(ctx)
    assert second.parse_stage_skipped
    assert second.content == first.content
    assert len(calls) == 5


@pytest.mark.parametrize("layout_issue", ["ambiguous", "unavailable"])
@pytest.mark.asyncio
async def test_falls_back_without_guessing_a_caption(
    engine, document, mathpix_api, page_layout, layout_issue
):
    ctx, _, _ = document
    calls, scenarios = mathpix_api
    if layout_issue == "ambiguous":
        page_layout["lines"].append(
            {
                "type": "figure_label",
                "text": "Figure 8. Competing caption.",
                "region": page_layout["lines"][1]["region"],
            }
        )
    else:
        scenarios["layout_response"] = httpx.Response(503)
    result = await engine.parse(ctx)
    drawings = json.loads(
        Path(result.blocks_path).with_name("battery-study.drawings.json").read_text()
    )["drawings"]
    assert {
        Path(item["src"]).name: (item["caption"], item["footnotes"])
        for item in drawings.values()
    } == {
        CROP_LEFT: ("", []),
        CROP_RIGHT: (
            "Figure 9. Impedance response." if layout_issue == "ambiguous" else "",
            [],
        ),
    }
    assert calls.count(("GET", f"/v3/pdf/{PDF_ID}.lines.json")) == 1


@pytest.mark.asyncio
async def test_pipeline_cancellation_keeps_task_for_retry(
    engine, document, mathpix_api
):
    ctx, _, archive = document
    calls, scenarios = mathpix_api
    scenarios["cancel_on_first_poll"] = ctx.pipeline_cancel_event

    with pytest.raises(ParsePipelineCancelled, match="cancelled"):
        await engine.parse(ctx)
    assert await ctx.rag.full_docs.get_by_id(ctx.doc_id) is None
    archive.assert_not_awaited()
    journal = ctx.resolve("mathpix").parsed_dir / ".mathpix" / "job.json"
    assert json.loads(journal.read_text())["pdf_id"] == PDF_ID

    ctx.pipeline_cancel_event.clear()
    assert (await engine.parse(ctx)).parse_engine == "mathpix"
    assert calls.count(("POST", "/v3/pdf")) == 1
    assert calls.count(("GET", f"/v3/pdf/{PDF_ID}")) == 2
    archive.assert_awaited_once()


@pytest.mark.asyncio
async def test_format_failure_preserves_paid_ocr_task(engine, document, mathpix_api):
    ctx, _, archive = document
    calls, scenarios = mathpix_api
    scenarios["conversion_state"] = "error"

    with pytest.raises(RuntimeError, match="ZIP conversion failed"):
        await engine.parse(ctx)
    assert ("GET", f"/v3/pdf/{PDF_ID}.md.zip") not in calls
    archive.assert_not_awaited()

    assert (await engine.parse(ctx)).parse_engine == "mathpix"
    assert calls.count(("POST", "/v3/pdf")) == 1
    assert calls.count(("GET", f"/v3/pdf/{PDF_ID}.md.zip")) == 1

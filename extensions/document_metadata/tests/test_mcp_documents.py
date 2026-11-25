from dataclasses import dataclass
from html import escape
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import HTMLResponse

from extensions.document_metadata.mcp_documents import (
    _compact_document_results,
    _extend_app,
)


pytestmark = pytest.mark.offline


@dataclass(frozen=True)
class Operation:
    name: str
    method: str
    path: str
    scope: str = "kb:read"


class GatewayError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


@pytest.mark.asyncio
async def test_metadata_tools_keep_the_complete_http_contract_and_forward_fields():
    hidden = (
        "get_ollama_version",
        "list_ollama_models",
        "list_running_ollama_models",
        "ollama_generate",
        "ollama_chat",
        "get_ui_customization",
        "get_ui_asset",
        "get_webui_url",
    )
    visible = (
        Operation("retrieve_kb_data", "post", "/query/data"),
        Operation("clear_documents", "delete", "/documents", "kb:admin"),
    )
    operations = SimpleNamespace(
        OPERATIONS=(*visible, *(Operation(name, "get", f"/{name}") for name in hidden)),
        Operation=Operation,
        READ="kb:read",
    )

    class Gateway:
        request = AsyncMock(return_value={"document_id": "doc-manual"})

    application = SimpleNamespace(LightRAGGateway=Gateway)

    def production_app():
        app = Starlette()
        app.state.gateway = application.LightRAGGateway()
        return app

    application.production_app = production_app
    gateway_module = SimpleNamespace(KNOWLEDGE_BASES={}, GatewayError=GatewayError)
    app = _extend_app(application, operations, gateway_module)
    _extend_app(application, operations, gateway_module)

    assert {op.name for op in operations.OPERATIONS} > set(hidden)
    exposed = application.OPERATIONS_BY_NAME
    assert set(exposed) == {
        "retrieve_kb_data",
        "clear_documents",
        "search_documents",
        "get_document_metadata",
        "read_document",
        "get_document_image",
    }
    assert len(operations.OPERATIONS) == len(visible) + len(hidden) + 4
    assert exposed["clear_documents"].scope == "kb:admin"
    assert exposed["read_document"].scope == "kb:read"
    assert application.LightRAGGateway is Gateway

    fields = ["title", "domain_details.experiment"]
    result = await app.state.gateway.request(
        "lead-acid", "get", "/documents/doc-manual/metadata", query={"fields": fields}
    )
    assert result == {"document_id": "doc-manual"}
    Gateway.request.assert_awaited_once_with(
        "lead-acid",
        "get",
        "/documents/doc-manual/metadata",
        body=None,
        query={"fields": fields},
        files=None,
    )
    Gateway.request.reset_mock()
    application.tool_result = lambda data, *, message=None, **_: {
        "structuredContent": data,
        "content": message,
    }
    _compact_document_results(application)
    response = {"documents": [{"document_id": "doc-1", "metadata": {"title": "One"}}]}
    Gateway.request.return_value = response
    catalogue = await app.state.gateway.request(
        "lead-acid",
        "post",
        "/documents/search",
        body={"fields": ["title"]},
        query={"response_format": "structured"},
    )
    Gateway.request.assert_awaited_once_with(
        "lead-acid",
        "post",
        "/documents/search",
        body={"fields": ["title"]},
        query={},
        files=None,
    )
    wire = application.tool_result(catalogue)
    assert wire["structuredContent"] == response
    assert (
        wire["content"]
        == "Document count is 1. Result data is available in structuredContent."
    )
    assert json.loads(application.tool_result(response)["content"]) == response


class _Markdown(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self):
        yield b"# Beginning\n"
        yield b"End of the full document.\n"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_full_document_tool_and_oauth_download_stream():
    markdown = _Markdown()
    upstream = AsyncMock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/markdown; charset=utf-8"},
            stream=markdown,
        )
    )
    token = None

    class Gateway:
        request = AsyncMock(return_value={"document_id": "doc-manual"})

        def __init__(self, http):
            self.http = http
            self.origin = "https://example.test"
            self.keys = {"lead-acid": "upstream-test-key"}

    operations = SimpleNamespace(OPERATIONS=(), Operation=Operation, READ="kb:read")
    gateway_module = SimpleNamespace(
        KNOWLEDGE_BASES={"lead-acid": "http://upstream.test/lead-acid"},
        GatewayError=GatewayError,
    )
    application = SimpleNamespace(
        LightRAGGateway=Gateway, get_access_token=lambda: token
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:

        def production_app():
            app = Starlette()
            app.state.gateway = application.LightRAGGateway(backend)
            return app

        application.production_app = production_app
        app = _extend_app(application, operations, gateway_module)
        path = "/documents/doc-manual/content"
        result = await app.state.gateway.request(
            "lead-acid", "get", path, query={"scope": "full"}
        )
        assert (
            result["download_url"]
            == "https://example.test/mcp/download/lead-acid/doc-manual"
        )
        assert "content" not in result and result["media_type"] == "text/markdown"
        Gateway.request.assert_awaited_once_with(
            "lead-acid", "get", path, query={"scope": "outline"}
        )

        with pytest.raises(GatewayError) as exc:
            await app.state.gateway.request(
                "lead-acid", "get", path, query={"scope": "full", "page_start": 3}
            )
        assert exc.value.status == 422
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as front:
            assert (await front.get(result["download_url"])).status_code == 401
            token = SimpleNamespace(subject="reader", scopes=["kb:write"])
            assert (await front.get(result["download_url"])).status_code == 403
            upstream.assert_not_awaited()

            token.scopes = ["kb:read"]
            response = await front.get(result["download_url"])
            assert response.status_code == 200
            assert response.text == "# Beginning\nEnd of the full document.\n"
            assert response.headers["Cache-Control"] == "private, no-store"
            assert (
                response.headers["Content-Disposition"]
                == 'attachment; filename="document.md"'
            )

    upstream.assert_awaited_once()
    request = upstream.await_args.args[0]
    assert request.url.path == "/lead-acid/documents/doc-manual/content"
    assert request.url.params.multi_items() == [("scope", "full")]
    assert request.headers["X-API-Key"] == "upstream-test-key"
    assert markdown.closed


@pytest.mark.asyncio
async def test_citation_signs_in_to_named_workspace_and_opens_versioned_excerpt():
    passage = "a" * 64 + ".28.59"
    excerpt = {
        "passage": passage,
        "text": "Pressure < 7 bar",
        "line_start": 3,
        "line_end": 4,
    }
    records = {}
    token = None

    def put(kind, key, value, ttl, *, cap=None):
        records[(kind, key)] = value
        return True

    class Gateway:
        origin = "https://private.test"
        authenticate = AsyncMock(return_value=True)
        request = AsyncMock(
            side_effect=[
                {"document_id": "doc-reference", "segments": [dict(excerpt)]},
                {"document_id": "doc-reference", "segments": [dict(excerpt)]},
                GatewayError(409, "The cited text has changed."),
            ]
        )

    operations = SimpleNamespace(OPERATIONS=(), Operation=Operation, READ="kb:read")
    gateway_module = SimpleNamespace(
        KNOWLEDGE_BASES={
            "lead-acid": "http://lead.test",
            "battery-life": "http://battery.test",
        },
        GatewayError=GatewayError,
    )
    browser_records = SimpleNamespace(
        put=put,
        get=lambda kind, key: records.get((kind, key)),
        take=lambda kind, key: records.pop((kind, key), None),
    )
    tickets = iter(("test-ticket", "reader-session", "battery-ticket"))

    def cookie(response, key, value, age):
        response.set_cookie(key, value, secure=True)
        return response

    def tool_result(data, *, message=None, failed=False, meta=None):
        return {"content": message or json.dumps(data), "structuredContent": data}

    def web_page(title, body, status=200):
        return HTMLResponse(f"<h1>{escape(title)}</h1>{body}", status_code=status)

    web = SimpleNamespace(
        page=web_page,
        credentials=lambda title, detail, ticket, address, button: web_page(
            title, detail + f'<form action="{escape(address, quote=True)}"></form>'
        ),
        HEADERS={"Cache-Control": "no-store"},
    )
    application = SimpleNamespace(
        LightRAGGateway=Gateway,
        get_access_token=lambda: token,
        Records=lambda: browser_records,
        new_secret=lambda: next(tickets),
        browser_cookie=cookie,
        tool_result=tool_result,
        tool_descriptor=lambda name, description, schema: (description, schema),
    )

    def production_app():
        application.Records()
        app = Starlette()
        app.state.gateway = application.LightRAGGateway()
        return app

    application.production_app = production_app
    app = _extend_app(application, operations, gateway_module, web)
    schema = {
        "$defs": {
            "DocumentSearchFilters": {
                "properties": {"year_from": {}, "evidence_strength": {}}
            }
        }
    }
    description, advertised = application.tool_descriptor(
        "search_documents", "Search documents", schema
    )
    assert "body.document_ids with body.fields" in description
    assert "year_from, evidence_strength" in description
    assert advertised["$defs"] == schema["$defs"]
    assert "properties" not in schema
    assert advertised["properties"]["response_format"]["enum"] == [
        "legacy_json",
        "structured",
    ]
    result = await app.state.gateway.request(
        "lead-acid", "get", "/documents/doc-reference/content", query={"line_start": 3}
    )
    link = result["segments"][0]["citation_url"]
    assert (
        link
        == f"https://private.test/mcp/citation/lead-acid/doc-reference?passage={passage}"
    )
    assert "passage" not in result["segments"][0]
    args, kwargs = Gateway.request.await_args
    assert args == ("lead-acid", "get", "/documents/doc-reference/content")
    assert kwargs["query"] == {"line_start": 3, "include_citations": True}

    wire = application.tool_result(result)
    assert json.loads(wire["content"]) == wire["structuredContent"] == result
    assert len(wire["content"]) < len(json.dumps(result, ensure_ascii=False))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=Gateway.origin
    ) as front:
        tracked_link = link + "&utm_source=chatgpt.com&utm_medium=referral"
        assert (await front.get(tracked_link + "&scope=full")).status_code == 404
        assert (await front.get(tracked_link + f"&passage={passage}")).status_code == 404
        login = await front.get(tracked_link)
        assert login.status_code == 200 and "登录" in login.text
        assert f'<form action="{link.removeprefix(Gateway.origin)}">' in login.text
        assert "Pressure" not in login.text
        token = SimpleNamespace(subject="reader", scopes=["kb:write"])
        assert (await front.get(tracked_link)).status_code == 403
        token = None
        fields = {
            "ticket": "test-ticket",
            "username": "reader",
            "password": "test-only",
        }
        assert (await front.post(link, data=fields)).status_code == 403
        assert Gateway.request.await_count == 1
        Gateway.authenticate.assert_not_awaited()

        approved = await front.post(
            link, data=fields, headers={"Origin": Gateway.origin}
        )
        assert approved.status_code == 303
        assert Gateway.origin + approved.headers["location"] == link
        Gateway.authenticate.assert_awaited_once_with(
            "reader", "test-only", kb="lead-acid"
        )
        opened = await front.get(approved.headers["location"])
        assert opened.status_code == 200
        assert "解析文本行 3–4" in opened.text and "Pressure &lt; 7 bar" in opened.text
        assert "PDF 页码" not in opened.text
        assert Gateway.request.await_args.kwargs["query"] == {
            "include_citations": False,
            "passage": passage,
        }

        other = await front.get(link.replace("/lead-acid/", "/battery-life/"))
        assert other.status_code == 200 and "登录" in other.text
        assert Gateway.request.await_count == 2
        assert (await front.get(link)).status_code == 409

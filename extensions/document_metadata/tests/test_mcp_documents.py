from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.applications import Starlette

from extensions.document_metadata.mcp_documents import _extend_app


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
    }
    assert len(operations.OPERATIONS) == len(visible) + len(hidden) + 3
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

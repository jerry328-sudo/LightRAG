"""Extend the separately deployed MCP service with the document read API."""

import re
from urllib.parse import quote

import httpx
from starlette.responses import JSONResponse, StreamingResponse


_NEW_OPERATIONS = (
    ("search_documents", "post", "/documents/search"),
    ("get_document_metadata", "get", "/documents/{document_id}/metadata"),
    ("read_document", "get", "/documents/{document_id}/content"),
)
_HIDDEN_OPERATIONS = {
    "get_ollama_version",
    "list_ollama_models",
    "list_running_ollama_models",
    "ollama_generate",
    "ollama_chat",
    "get_ui_customization",
    "get_ui_asset",
    "get_webui_url",
}
_CONTENT_PATH = re.compile(r"^/documents/([^/]+)/content$")
_DOCUMENT_ID = re.compile(r"(?!\.{1,2}$)[^/\\%\x00-\x1f\x7f]{1,512}")


class _FullTextResponse(StreamingResponse):
    def __init__(self, upstream):
        self.upstream = upstream
        super().__init__(
            upstream.aiter_bytes(),
            headers={
                "Content-Type": upstream.headers["content-type"],
                "Content-Disposition": 'attachment; filename="document.md"',
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.upstream.aclose()


async def _download(gateway, endpoint, kb, document_id):
    path = f"/documents/{quote(document_id, safe='')}/content"
    request = gateway.http.build_request(
        "GET",
        endpoint + path,
        params={"scope": "full"},
        headers={"X-API-Key": gateway.keys[kb], "Accept": "text/markdown"},
    )
    try:
        upstream = await gateway.http.send(request, stream=True)
    except httpx.TimeoutException:
        return JSONResponse({"detail": "The document download timed out."}, 504)
    except httpx.HTTPError:
        return JSONResponse({"detail": "The knowledge base is unavailable."}, 502)

    status = upstream.status_code
    content_type = upstream.headers.get("content-type", "").split(";", 1)[0]
    if status == 200 and content_type == "text/markdown":
        return _FullTextResponse(upstream)

    await upstream.aclose()
    if status == 404:
        return JSONResponse(
            {"detail": "The document or its content was not found."}, 404
        )
    return JSONResponse(
        {"detail": "The knowledge base could not supply the document."}, 502
    )


def _register_operations(application, operations):
    additions = tuple(operations.Operation(*entry) for entry in _NEW_OPERATIONS)
    existing = {op.name: op for op in operations.OPERATIONS}
    for op in additions:
        if op.name in existing and existing[op.name] != op:
            raise RuntimeError(f"Conflicting MCP operation: {op.name}")
    operations.OPERATIONS = (
        *operations.OPERATIONS,
        *(op for op in additions if op.name not in existing),
    )
    # The upstream contract still audits the complete HTTP surface. Only the
    # MCP catalogue and dispatch table omit the tools the user disabled.
    application.OPERATIONS = tuple(
        op for op in operations.OPERATIONS if op.name not in _HIDDEN_OPERATIONS
    )
    application.OPERATIONS_BY_NAME = {op.name: op for op in application.OPERATIONS}


def _extend_app(application, operations, gateway_module):
    _register_operations(application, operations)
    base_gateway = application.LightRAGGateway
    endpoints = gateway_module.KNOWLEDGE_BASES
    gateway = None

    class DocumentGateway(base_gateway):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nonlocal gateway
            gateway = self

        async def request(self, kb, method, path, *, body=None, query=None, files=None):
            match = _CONTENT_PATH.fullmatch(path) if method == "get" else None
            if match is None or (query or {}).get("scope") != "full":
                return await super().request(
                    kb, method, path, body=body, query=query, files=files
                )
            if query.keys() != {"scope"}:
                raise gateway_module.GatewayError(
                    422, "Full document downloads take no range selectors."
                )
            # Check existence without transferring the full document to the model.
            outline = await super().request(
                kb, method, path, query={"scope": "outline"}
            )
            return {
                "document_id": outline["document_id"],
                "scope": "full",
                "download_url": f"{self.origin}/mcp/download/{kb}/{match.group(1)}",
                "media_type": "text/markdown",
                "authentication": "OAuth Bearer token with kb:read scope",
            }

    # The MCP factory constructs its gateway synchronously. Reuse that instance
    # for downloads so it has the same client lifecycle and upstream credentials.
    application.LightRAGGateway = DocumentGateway
    try:
        app = application.production_app()
    finally:
        application.LightRAGGateway = base_gateway
    if gateway is None:
        raise RuntimeError("The MCP factory did not initialize its gateway")

    async def download(request):
        token = application.get_access_token()
        if token is None or not token.subject:
            return JSONResponse(
                {"detail": "An MCP OAuth access token is required."},
                401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if operations.READ not in token.scopes:
            return JSONResponse({"detail": "The token needs kb:read scope."}, 403)
        kb = request.path_params["kb"]
        document_id = request.path_params["document_id"]
        if kb not in endpoints or not _DOCUMENT_ID.fullmatch(document_id):
            return JSONResponse(
                {"detail": "Invalid knowledge base or document ID."}, 404
            )
        return await _download(gateway, endpoints[kb], kb, document_id)

    app.add_route("/mcp/download/{kb}/{document_id}", download, methods=["GET"])
    return app


def production_app():
    # The deployment installs this module alongside these gateway modules.
    import app as application
    import gateway
    import operations

    return _extend_app(application, operations, gateway)

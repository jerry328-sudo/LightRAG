"""Extend the separately deployed MCP service with the document read API."""

import base64
import json
import re
from html import escape
from urllib.parse import quote, unquote, urlencode

import httpx
from starlette.responses import JSONResponse, RedirectResponse, Response, StreamingResponse


_NEW_OPERATIONS = (
    ("search_documents", "post", "/documents/search"),
    ("get_document_metadata", "get", "/documents/{document_id}/metadata"),
    ("read_document", "get", "/documents/{document_id}/content"),
    ("get_document_image", "get", "/documents/{document_id}/images/{image_id}"),
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
_IMAGE_PATH = re.compile(r"^/documents/([^/]+)/images/([A-Za-z0-9_-]{1,128})$")
_DOCUMENT_READ_PATH = re.compile(r"^/documents/[^/]+/(?:metadata|content)$")
_CATALOGUE_PATHS = {"/documents/search", "/documents/paginated"}
_DOCUMENT_TOOLS = {
    "search_documents",
    "list_documents",
    "get_document_metadata",
    "read_document",
}
_RESPONSE_FORMATS = ("legacy_json", "structured")
_DOCUMENT_ID = re.compile(r"(?!\.{1,2}$)[^/\\%\x00-\x1f\x7f]{1,512}")
_PASSAGE = re.compile(r"[0-9a-f]{64}\.[0-9]{1,12}\.[0-9]{1,12}")
_CITATION_TICKET = "__Secure-echem-citation-ticket"
_CITATION_SESSION = "__Secure-echem-citation-browser"
_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_MAX_IMAGE_BYTES = 4 * 1024 * 1024


def _citation_path(kb, document_path, locator):
    return f"/mcp/citation/{kb}/{document_path}?{urlencode(locator)}"


def _add_citation_urls(payload, origin, kb, document_path):
    if not isinstance(payload, dict):
        return payload
    for key in ("segments", "matches", "chunks"):
        for item in payload.get(key, ()):
            passage = item.get("passage")
            if isinstance(passage, str) and _PASSAGE.fullmatch(passage):
                locator = {"passage": passage}
                if key == "chunks":
                    locator["chunk_id"] = item["chunk_id"]
                item["citation_url"] = origin + _citation_path(
                    kb, document_path, locator
                )
                del item["passage"]
    return payload


class _StructuredResult(dict):
    """Keep the serialization preference out of the document data itself."""


class _ImageResult(dict):
    def __init__(self, document_id, image_id, url, data, media_type):
        super().__init__(
            document_id=document_id, image_id=image_id, image_url=url, media_type=media_type
        )
        self.data = data


def _compact_document_results(application):
    original = application.tool_result
    if getattr(original, "_document_metadata_compact", False):
        return

    def tool_result(data, *, message=None, failed=False, meta=None):
        if isinstance(data, _ImageResult) and not failed:
            from mcp.types import ImageContent

            result = original(dict(data), message=message, meta=meta)
            result.content.append(ImageContent(
                type="image", data=base64.b64encode(data.data).decode("ascii"),
                mimeType=data["media_type"],
            ))
            return result
        if message is None and not failed and isinstance(data, _StructuredResult):
            message = "Result data is available in structuredContent."
            documents = data.get("documents")
            if isinstance(documents, list):
                message = f"Document count is {len(documents)}. {message}"
        elif (
            message is None
            and not failed
            and isinstance(data, dict)
            and ("document_id" in data or "documents" in data)
        ):
            # Clients with legacy text-only result support still receive the
            # full JSON. Whitespace is the only difference in that copy.
            message = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return original(data, message=message, failed=failed, meta=meta)

    tool_result._document_metadata_compact = True
    application.tool_result = tool_result


def _describe_document_tools(application):
    original = application.tool_descriptor
    if getattr(original, "_document_metadata_description", False):
        return

    def tool_descriptor(name, description, schema, *args, **kwargs):
        if name == "search_documents":
            filters = (
                schema.get("$defs", {})
                .get("DocumentSearchFilters", {})
                .get("properties", {})
            )
            description += (
                ". Omit body.query for a catalogue. Use body.document_ids with "
                "body.fields to read several records at once. Select dotted metadata "
                "fields such as title, published_date, type_details.journal, or "
                "evidence_strength. Follow next_cursor via body.cursor."
            )
            if filters:
                description += " Available body.filters: " + ", ".join(filters) + "."
        elif name == "list_documents":
            description += (
                ". This lists processing and import status. For a research catalogue "
                "with selected fields, use search_documents."
            )
        elif name == "read_document":
            description += (
                ". Specify lines, pages, a section, or a chunk for a bounded read. "
                "The resulting citation_url opens the exact passage upon sign-in."
            )
        elif name == "retrieve_kb_data":
            description += (
                " Image-description hits include image_id and document_id. "
                "Pass both with kb to get_document_image to see the extracted image."
            )
        elif name == "get_document_image":
            description += (
                " Use the document_id and image_id from retrieve_kb_data. "
                "Returns one extracted image and a sign-in protected image_url."
            )
        if name in _DOCUMENT_TOOLS:
            schema = {
                **schema,
                "properties": {
                    **schema.get("properties", {}),
                    "response_format": {
                        "type": "string",
                        "enum": list(_RESPONSE_FORMATS),
                        "default": "legacy_json",
                        "description": (
                            "Use structured to receive the full data once in structuredContent "
                            "and a short text summary. legacy_json keeps a full text copy."
                        ),
                    },
                },
            }
            description += (
                " Use response_format=structured if you can read structuredContent "
                "and want to avoid a second full JSON copy."
            )
        return original(name, description, schema, *args, **kwargs)

    tool_descriptor._document_metadata_description = True
    application.tool_descriptor = tool_descriptor


def _citation_locators(request):
    query = request.query_params
    passage = query.get("passage", "")
    if not _PASSAGE.fullmatch(passage) or len(query.getlist("passage")) != 1:
        return None
    locator = {"passage": passage}
    if "chunk_id" in query:
        chunk = query["chunk_id"]
        if len(query.getlist("chunk_id")) != 1 or not 0 < len(chunk) <= 256:
            return None
        locator["chunk_id"] = chunk
    # Chat clients can append UTM tracking. Keep it out of the locator and login URL.
    if any(key not in locator and not key.startswith("utm_") for key in query):
        return None
    return locator


async def _show_citation(gateway, gateway_module, web, kb, document_path, locator):
    try:
        result = await gateway.request(
            kb,
            "get",
            f"/documents/{document_path}/content",
            query={"include_citations": False, **locator},
        )
    except gateway_module.GatewayError as problem:
        if problem.status == 409:
            return web.page(
                "引用已过期", "<p>该文献的解析版本已变更，请重新读取该段原文。</p>", 409
            )
        status = problem.status if problem.status < 500 else 502
        return web.page("无法读取原文", "<p>暂时无法读取这段原文。</p>", status)
    key = "chunks" if "chunk_id" in locator else "segments"
    excerpts = result.get(key) or []
    if not excerpts:
        return web.page("找不到原文", "<p>该段原文已不可用。</p>", 404)
    excerpt = excerpts[0]
    location = []
    if excerpt.get("pages"):
        location.append(
            "PDF 页码（按解析分块） " + ", ".join(map(str, excerpt["pages"]))
        )
    if excerpt.get("line_start"):
        location.append(f"解析文本行 {excerpt['line_start']}–{excerpt['line_end']}")
    if excerpt.get("section"):
        location.append(excerpt["section"])
    if not location and "chunk_id" in locator:
        location.append("分块 " + locator["chunk_id"])
    return web.page(
        "知识库原文",
        f"<p>{escape(' · '.join(location))}</p><pre>{escape(excerpt['text'])}</pre>",
    )


async def _fetch_image(gateway, gateway_module, kb, path):
    try:
        async with gateway.http.stream(
            "GET", gateway_module.KNOWLEDGE_BASES[kb] + path,
            headers={"X-API-Key": gateway.keys[kb]},
        ) as upstream:
            if upstream.status_code in (404, 413, 415):
                raise gateway_module.GatewayError(upstream.status_code, {
                    404: "The extracted image was not found.",
                    413: "The image exceeds the 4 MiB response limit.",
                    415: "The image format is unsupported.",
                }[upstream.status_code])
            media_type = upstream.headers.get("content-type", "").split(";", 1)[0]
            if upstream.status_code != 200 or media_type not in _IMAGE_TYPES:
                raise gateway_module.GatewayError(502, "The knowledge base could not supply the image.")
            data = bytearray()
            async for chunk in upstream.aiter_bytes(chunk_size=65536):
                if len(data) + len(chunk) > _MAX_IMAGE_BYTES:
                    raise gateway_module.GatewayError(413, "The image exceeds the 4 MiB response limit.")
                data.extend(chunk)
            if not data:
                raise gateway_module.GatewayError(502, "The knowledge base returned an empty image.")
            return bytes(data), media_type
    except httpx.TimeoutException as error:
        raise gateway_module.GatewayError(504, "The image request timed out.") from error
    except httpx.HTTPError as error:
        raise gateway_module.GatewayError(502, "The knowledge base is unavailable.") from error


async def _open_citation(
    request, application, operations, gateway_module, gateway, records, web
):
    kb = request.path_params["kb"]
    document = request.path_params["document_id"]
    image_id = request.path_params.get("image_id")
    if kb not in gateway_module.KNOWLEDGE_BASES or not _DOCUMENT_ID.fullmatch(document):
        return web.page("无效的原文引用", "<p>请重新读取原文以获得有效链接。</p>", 404)
    document_path = quote(document, safe="")
    if image_id is None:
        locator = _citation_locators(request)
        if locator is None:
            return web.page("无效的原文引用", "<p>请重新读取原文以获得有效链接。</p>", 404)
        address = _citation_path(kb, document_path, locator)
    else:
        image_path = f"/documents/{document_path}/images/{image_id}"
        if not _IMAGE_PATH.fullmatch(image_path):
            return web.page("无效的图片引用", "<p>请重新检索此图片。</p>", 404)
        address = f"/mcp/images/{kb}/{document_path}/{image_id}"

    async def show():
        if image_id is None:
            return await _show_citation(gateway, gateway_module, web, kb, document_path, locator)
        try:
            data, media_type = await _fetch_image(gateway, gateway_module, kb, image_path)
        except gateway_module.GatewayError as problem:
            return web.page("无法读取图片", "<p>该图片暂时不可用。</p>", problem.status)
        return Response(data, media_type=media_type, headers=web.HEADERS)

    token = application.get_access_token()
    if token is not None:
        if not token.subject or operations.READ not in token.scopes:
            return web.page("权限不足", "<p>此令牌需要知识库读取权限。</p>", 403)
        return await show()

    session = records.get(
        "citation_browser", request.cookies.get(_CITATION_SESSION, "")
    )
    if session and session["kb"] == kb:
        return await show()

    if request.method == "GET":
        ticket = application.new_secret()
        if not records.put("citation_login", ticket, {"path": address}, 600, cap=500):
            return web.page("暂时无法登录", "<p>请稍后重新尝试。</p>", 429)
        response = web.credentials(
            "查看知识库原文" if image_id is None else "查看知识库图片",
            "<p>用所选知识库的账号登录后查看内容。</p>",
            ticket,
            address,
            "登录并查看",
        )
        return application.browser_cookie(response, _CITATION_TICKET, ticket, 600)

    ticket = request.cookies.get(_CITATION_TICKET, "")
    pending = records.get("citation_login", ticket)
    if (
        request.headers.get("origin") != gateway.origin
        or not pending
        or pending["path"] != address
    ):
        return web.page("请求已失效", "<p>请重新打开原文链接。</p>", 403)
    form = await request.form()
    username, password = form.get("username"), form.get("password")
    if (
        form.get("ticket") != ticket
        or not isinstance(username, str)
        or not isinstance(password, str)
        or not 0 < len(username) <= 128
        or not 0 < len(password) <= 512
    ):
        return web.page("登录失败", "<p>请检查登录信息。</p>", 400)
    try:
        accepted = await gateway.authenticate(username, password, kb=kb)
    except (gateway_module.GatewayError, httpx.HTTPError):
        return web.page("暂时无法登录", "<p>请稍后重新尝试。</p>", 503)
    if not accepted:
        return web.page("登录失败", "<p>请检查所选知识库的登录信息。</p>", 401)
    if not records.take("citation_login", ticket):
        return web.page("请求已失效", "<p>请重新打开原文链接。</p>", 403)
    session = application.new_secret()
    records.put("citation_browser", session, {"kb": kb, "subject": username}, 3600)
    response = RedirectResponse(address, status_code=303, headers=web.HEADERS)
    return application.browser_cookie(response, _CITATION_SESSION, session, 3600)


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


def _extend_app(application, operations, gateway_module, web=None):
    _register_operations(application, operations)
    base_gateway = application.LightRAGGateway
    endpoints = gateway_module.KNOWLEDGE_BASES
    gateway = None
    records = None
    records_factory = application.Records if web is not None else None

    def capture_records(*args, **kwargs):
        nonlocal records
        records = records_factory(*args, **kwargs)
        return records

    class DocumentGateway(base_gateway):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nonlocal gateway
            gateway = self

        async def request(self, kb, method, path, *, body=None, query=None, files=None):
            mode = (query or {}).get("response_format")
            if mode is not None:
                is_document_read = (method == "post" and path in _CATALOGUE_PATHS) or (
                    method == "get" and _DOCUMENT_READ_PATH.fullmatch(path)
                )
                if not is_document_read or mode not in _RESPONSE_FORMATS:
                    raise gateway_module.GatewayError(
                        422, "Unsupported document response format."
                    )
                query = {
                    key: value
                    for key, value in query.items()
                    if key != "response_format"
                }
            result = await self._read_request(
                kb, method, path, body=body, query=query, files=files
            )
            if mode == "structured" and isinstance(result, dict):
                return _StructuredResult(result)
            return result

        async def _read_request(self, kb, method, path, *, body, query, files):
            image = _IMAGE_PATH.fullmatch(path) if method == "get" else None
            if image is not None:
                document_path, image_id = image.groups()
                data, media_type = await _fetch_image(self, gateway_module, kb, path)
                return _ImageResult(
                    unquote(document_path), image_id,
                    f"{self.origin}/mcp/images/{kb}/{document_path}/{image_id}", data, media_type,
                )
            match = _CONTENT_PATH.fullmatch(path) if method == "get" else None
            if match is None:
                return await super().request(
                    kb, method, path, body=body, query=query, files=files
                )
            if (query or {}).get("scope") != "full":
                if web is None or (query or {}).get("include_citations") is False:
                    return await super().request(
                        kb, method, path, body=body, query=query, files=files
                    )
                result = await super().request(
                    kb,
                    method,
                    path,
                    body=body,
                    query={**(query or {}), "include_citations": True},
                    files=files,
                )
                return _add_citation_urls(result, self.origin, kb, match.group(1))
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
    if records_factory is not None:
        application.Records = capture_records
    try:
        app = application.production_app()
    finally:
        application.LightRAGGateway = base_gateway
        if records_factory is not None:
            application.Records = records_factory
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
    if web is not None:
        if records is None:
            raise RuntimeError("The MCP factory did not initialize its browser records")
        _compact_document_results(application)
        _describe_document_tools(application)

        async def citation(request):
            return await _open_citation(
                request, application, operations, gateway_module, gateway, records, web
            )

        app.add_route(
            "/mcp/citation/{kb}/{document_id}", citation, methods=["GET", "POST"]
        )
        app.add_route(
            "/mcp/images/{kb}/{document_id}/{image_id}", citation, methods=["GET", "POST"]
        )
    return app


def production_app():
    # The deployment installs this module alongside these gateway modules.
    import app as application
    import gateway
    import operations
    import web

    return _extend_app(application, operations, gateway, web)

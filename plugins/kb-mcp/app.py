"""Streamable HTTP MCP service for the two deployed LightRAG instances."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from mcp import types
from mcp.server.auth.handlers.metadata import MetadataHandler
from mcp.server.auth.handlers.token import TokenHandler
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware, get_access_token
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.middleware.client_auth import ClientAuthenticator
from mcp.server.auth.routes import build_metadata, build_resource_metadata_url, create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from auth import CLAUDE_CALLBACK, OAuthProvider, Records, new_secret
from gateway import GatewayError, KNOWLEDGE_BASES, LightRAGGateway
from operations import ADMIN, KB_INPUT, OPERATIONS, READ, SCOPES, WRITE, load_contract, upstream_request
from web import HEADERS, credentials, page

LOG = logging.getLogger("echem_kb_mcp")
TITLES = {"battery-life": "电池寿命预测", "lead-acid": "铅酸电池"}
SCOPE_LABELS = {READ: "查询文档和图谱", WRITE: "上传、修改和删除资料", ADMIN: "修复、强制恢复和清空知识库"}
TICKET_COOKIE = "__Secure-echem-mcp-ticket"
BROWSER_COOKIE = "__Secure-echem-mcp-browser"
SOURCE_SECONDS = 7 * 86400
TOKEN_PATH = "/mcp/oauth/token"
ISSUER_PATH = "/mcp/oauth"
ICON_PATH = "/mcp/icon.png"
OPERATIONS_BY_NAME = {op.name: op for op in OPERATIONS}
ID_INPUT = {"type": "string", "pattern": r"^[A-Za-z0-9_-]{43}$"}
QUERY_INPUT = {"type": "string", "minLength": 2, "maxLength": 2048}


def tool_result(data, *, message=None, failed=False, meta=None):
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message or json.dumps(data, ensure_ascii=False))],
        structuredContent=data,
        isError=failed,
        _meta=meta,
    )


def tool_descriptor(name, description, schema, scope=READ, destructive=False, extra_meta=None):
    security = [{"type": "oauth2", "scopes": [READ] if scope == READ else [READ, scope]}]
    return types.Tool(
        name=name,
        description=description,
        inputSchema=schema,
        annotations=types.ToolAnnotations(
            readOnlyHint=scope == READ, destructiveHint=destructive, openWorldHint=False,
        ),
        _meta={"securitySchemes": security, **(extra_meta or {})},
        securitySchemes=security,
    )


def browser_cookie(response, key, value, age):
    response.set_cookie(key, value, secure=True, httponly=True, samesite="lax", path="/mcp/", max_age=age)
    return response


def create_app(*, origin, state_path, api_keys, admin_users, upstream_transport=None):
    host = urlsplit(origin).netloc
    resource = AnyHttpUrl(origin + "/mcp")
    issuer = AnyHttpUrl(origin + ISSUER_PATH)
    metadata_url = build_resource_metadata_url(resource)
    records = Records(state_path)
    provider = OAuthProvider(origin, records)
    registration = ClientRegistrationOptions(enabled=True, valid_scopes=SCOPES, default_scopes=SCOPES)
    revocation = RevocationOptions(enabled=True)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(180, connect=5), follow_redirects=False, trust_env=False,
        transport=upstream_transport,
    )
    gateway = LightRAGGateway(client, api_keys, origin)
    server = Server(
        "Echem 知识库", version="1.0.0", website_url=origin,
        instructions="Search private research libraries. Specify kb explicitly for every write or administrative action.",
        icons=[types.Icon(src=origin + ICON_PATH, mimeType="image/png", sizes=["256x256"])],
    )
    session_manager = StreamableHTTPSessionManager(
        server, stateless=True, json_response=True,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[host, host + ":443", "127.0.0.1:9623"],
            allowed_origins=[origin, "https://chatgpt.com"],
        ),
    )
    tool_list = []

    @asynccontextmanager
    async def lifespan(_app):
        async with client:
            specs = []
            for endpoint in KNOWLEDGE_BASES.values():
                reply = await client.get(endpoint + "/openapi.json", timeout=10)
                reply.raise_for_status()
                specs.append(reply.json())
            contract = load_contract(specs[0])
            if load_contract(specs[1]) != contract:
                raise RuntimeError("LightRAG instances have different MCP input contracts.")
            openapi = specs[0]
            for op in OPERATIONS:
                description = openapi["paths"][op.path][op.method]["summary"]
                if op.confirm:
                    description += ". Requires a separate approval in the browser."
                tool_list.append(tool_descriptor(
                    op.name, description, contract[op.name], op.scope, op.destructive,
                    {"openai/fileParams": ["file"]} if op.name == "upload_document" else None,
                ))
            tool_list.extend([
                tool_descriptor("search", "Search both knowledge bases for citable source excerpts.", {
                    "type": "object", "properties": {"query": QUERY_INPUT},
                    "required": ["query"], "additionalProperties": False,
                }),
                tool_descriptor("search_kb", (
                    "Search a single knowledge base for citable source excerpts. "
                    "Image hits include document_id and image_id for get_document_image."
                ), {
                    "type": "object", "properties": {"query": QUERY_INPUT, "kb": KB_INPUT},
                    "required": ["query", "kb"], "additionalProperties": False,
                }),
                tool_descriptor("fetch", (
                    "Fetch the full indexed excerpt for a result ID returned by search or search_kb. "
                    "For image hits, metadata includes kb, document_id, and image_id."
                ), {
                    "type": "object", "properties": {"id": ID_INPUT},
                    "required": ["id"], "additionalProperties": False,
                }),
                tool_descriptor("get_admin_action_status", "Check whether the browser approved or completed an administrative action.", {
                    "type": "object", "properties": {"id": ID_INPUT},
                    "required": ["id"], "additionalProperties": False,
                }, ADMIN),
            ])
            async with session_manager.run():
                yield

    @server.list_tools()
    async def list_tools():
        return tool_list

    async def run_operation(op, arguments):
        kb = arguments["kb"]
        if op.name == "upload_document":
            return await gateway.upload(kb, arguments["file"])
        path, query = upstream_request(op, arguments)
        return await gateway.request(kb, op.method, path, body=arguments.get("body"), query=query)

    async def search_one(kb, query):
        result = await gateway.request(
            kb, "post", "/query/data",
            body={"query": query, "mode": "mix", "top_k": 12, "chunk_top_k": 5},
        )
        return result.get("data", {}).get("chunks", [])[:5]

    async def source_search(args, subject):
        targets = [args["kb"]] if "kb" in args else list(KNOWLEDGE_BASES)
        responses = await asyncio.gather(
            *(search_one(kb, args["query"]) for kb in targets), return_exceptions=True,
        )
        hits, errors = [], {}
        for kb, response in zip(targets, responses, strict=True):
            if isinstance(response, GatewayError):
                errors[kb] = str(response)
                continue
            if isinstance(response, BaseException):
                raise response
            for chunk in response:
                text = chunk.get("content")
                if not isinstance(text, str) or not text.strip():
                    continue
                reference = new_secret()
                title = str(chunk.get("file_path") or "文献片段")[:512]
                url = origin + "/mcp/source/" + reference
                metadata = {"kb": kb, "chunk_id": chunk.get("chunk_id")}
                hit = {"id": reference, "title": title, "url": url}
                if chunk.get("document_id") and chunk.get("image_id"):
                    image = {"document_id": chunk["document_id"], "image_id": chunk["image_id"]}
                    metadata.update(image)
                    if "kb" in args:
                        hit.update(image)
                data = {
                    "subject": subject, "id": reference, "title": title, "text": text[:25000],
                    "url": url, "metadata": metadata,
                }
                if not records.put("source", reference, data, SOURCE_SECONDS, cap=3000):
                    raise GatewayError(429, "Too many saved search results. Use retrieve_kb_data for full raw results.")
                hits.append(hit)
        if errors and not hits:
            return tool_result({"errors": errors}, failed=True)
        return tool_result({"results": hits, **({"errors": errors} if errors else {})})

    @server.call_tool()
    async def call_tool(name, arguments):
        token = get_access_token()
        if token is None or not token.subject:
            return tool_result({"error": "Account authorization required."}, failed=True)
        op = OPERATIONS_BY_NAME.get(name)
        scope = op.scope if op else (ADMIN if name == "get_admin_action_status" else READ)
        if scope not in token.scopes:
            challenge = f'Bearer resource_metadata="{metadata_url}", error="insufficient_scope", scope="{READ} {scope}"'
            return tool_result(
                {"error": f"Authorize the {scope} permission to use this tool."},
                failed=True, meta={"mcp/www_authenticate": [challenge]},
            )
        if scope == ADMIN and token.subject not in admin_users:
            return tool_result({"error": "Knowledge base administrator permission is required."}, failed=True)

        try:
            if name in ("search", "search_kb"):
                return await source_search(arguments, token.subject)
            if name == "fetch":
                found = records.get("source", arguments["id"])
                if not found or found["subject"] != token.subject:
                    raise GatewayError(404, "This source is unavailable. Search again to refresh the citation.")
                return tool_result({key: value for key, value in found.items() if key != "subject"})
            if name == "get_admin_action_status":
                action = records.get("admin_result", arguments["id"]) or records.get("approval", arguments["id"])
                if not action or action["subject"] != token.subject:
                    raise GatewayError(404, "Approval has expired or the action does not exist.")
                return tool_result({key: value for key, value in action.items() if key not in {"subject", "arguments"}})
            if op is None:
                raise GatewayError(404, "Unknown tool.")
            if op.confirm:
                ticket = new_secret()
                action = {"subject": token.subject, "name": name, "arguments": arguments, "status": "awaiting_approval"}
                if not records.put("approval", ticket, action, 600, cap=200):
                    raise GatewayError(429, "Too many outstanding approvals.")
                return tool_result({
                    "status": "awaiting_approval", "approval_id": ticket,
                    "approval_url": origin + "/mcp/approve/" + ticket, "kb": arguments["kb"],
                })
            if scope != READ:
                LOG.info("Tool requested subject=%s action=%s kb=%s", token.subject, name, arguments["kb"])
            result = await run_operation(op, arguments)
            return tool_result(result if isinstance(result, dict) else {"data": result})
        except GatewayError as problem:
            return tool_result({"status": problem.status, "error": str(problem)}, failed=True)

    async def token_endpoint(request: Request):
        form = await request.form()
        if form.getlist("resource") != [str(resource)]:
            return JSONResponse({"error": "invalid_target"}, status_code=400, headers={"Cache-Control": "no-store"})
        return await TokenHandler(provider, ClientAuthenticator(provider)).handle(request)

    async def browser_sign_in(request, ticket, kind):
        if request.headers.get("origin") != origin or request.cookies.get(TICKET_COOKIE) != ticket:
            return None, page("请求已失效", "<p>请重新打开授权或确认页面。</p>", 403)
        form = await request.form()
        username, password = form.get("username"), form.get("password")
        if (
            form.get("ticket") != ticket or not isinstance(username, str) or not isinstance(password, str)
            or not 0 < len(username) <= 128 or not 0 < len(password) <= 512
        ):
            return None, page("登录失败", "<p>请重新打开页面并检查登录信息。</p>", 400)
        if kind == "approval" and (username != request.state.expected_user or username not in admin_users):
            return None, page("权限不足", "<p>请使用发起请求的管理员账号。</p>", 403)
        try:
            accepted = await gateway.authenticate(username, password)
        except (GatewayError, httpx.HTTPError):
            return None, page("暂时无法登录", "<p>请稍后重新尝试。</p>", 503)
        if not accepted:
            return None, page("登录失败", "<p>请返回上一页并检查用户名和密码。</p>", 401)
        return username, None

    async def consent(request: Request):
        ticket = request.query_params.get("ticket", "")
        pending = records.get("pending", ticket)
        if not pending:
            return page("授权请求已失效", "<p>请在 MCP 客户端重新发起授权。</p>", 410)
        if request.method == "GET":
            scopes = "、".join(SCOPE_LABELS[scope] for scope in pending["scopes"])
            callback = pending["params"]["redirect_uri"]
            callback_url = urlsplit(callback)
            client_name = "Claude" if callback == CLAUDE_CALLBACK else "ChatGPT"
            response = credentials(
                "授权连接 Echem 知识库",
                f"<p>你正在连接 {client_name}。许可范围为{escape(scopes)}，适用于两个知识库。</p>",
                ticket, "/mcp/consent?ticket=" + ticket, "登录并授权",
                callback_origin=f"{callback_url.scheme}://{callback_url.netloc}",
            )
            return browser_cookie(response, TICKET_COOKIE, ticket, 600)

        username, failure = await browser_sign_in(request, ticket, "consent")
        if failure is not None:
            return failure
        if ADMIN in pending["scopes"] and username not in admin_users:
            return page("权限不足", "<p>此账号没有知识库管理员权限。</p>", 403)
        callback = provider.grant(ticket, username)
        if not callback:
            return page("授权请求已失效", "<p>请在 MCP 客户端重新发起授权。</p>", 410)
        browser_session = new_secret()
        records.put("browser", browser_session, {"subject": username}, 3600)
        response = RedirectResponse(callback, status_code=303, headers=HEADERS)
        return browser_cookie(response, BROWSER_COOKIE, browser_session, 3600)

    async def approve(request: Request):
        ticket = request.path_params["ticket"]
        action = records.get("approval", ticket)
        if not action:
            return page("确认请求已失效", "<p>请查看执行状态，确认后再提交新的请求。</p>", 410)
        op = OPERATIONS_BY_NAME[action["name"]]
        kb = action["arguments"]["kb"]
        if request.method == "GET":
            parameters = escape(json.dumps(action["arguments"], ensure_ascii=False, indent=2))
            title = f"确认管理操作 · {TITLES[kb]}"
            labels = {"clear_documents": "清空知识库", "force_reset_recovery": "强制恢复流水线", "repair_source_conflict": "修复文档来源冲突"}
            response = credentials(
                title, f"<p>目标操作为<strong>{labels[op.name]}</strong>。确认前请检查参数和目标库。</p><pre>{parameters}</pre>",
                ticket, "/mcp/approve/" + ticket, "确认并执行",
            )
            return browser_cookie(response, TICKET_COOKIE, ticket, 600)

        request.state.expected_user = action["subject"]
        username, failure = await browser_sign_in(request, ticket, "approval")
        if failure is not None:
            return failure
        if records.take("approval", ticket) is None:
            return page("确认请求已使用", "<p>请查看执行状态。</p>", 409)
        outcome = {"subject": username, "kb": kb, "name": op.name}
        records.put("admin_result", ticket, {**outcome, "status": "submitted"}, 3600)
        LOG.info("Browser-approved action subject=%s action=%s kb=%s", username, op.name, kb)
        try:
            result = await run_operation(op, action["arguments"])
            outcome.update(status="completed", result=result)
        except GatewayError as error:
            state = "unknown" if error.status >= 500 else "failed"
            outcome.update(status=state, error=str(error))
        records.put("admin_result", ticket, outcome, 3600)
        detail = escape(json.dumps({key: value for key, value in outcome.items() if key != "subject"}, ensure_ascii=False, indent=2))
        return page("操作执行结果", f"<pre>{detail}</pre>")

    async def source(request: Request):
        reference = request.path_params["reference"]
        found = records.get("source", reference)
        if not found:
            return page("引文已失效", "<p>请回到对话重新检索。</p>", 404)
        browser = records.get("browser", request.cookies.get(BROWSER_COOKIE, ""))
        if browser and browser["subject"] == found["subject"]:
            return render_source(found)
        if request.method == "GET":
            ticket = new_secret()
            records.put("source_login", ticket, {"reference": reference}, 600, cap=500)
            response = credentials(
                "查看知识库引文", "<p>用现有知识库账号登录后查看引文。</p>",
                ticket, "/mcp/source/" + reference + "?ticket=" + ticket, "登录并查看",
            )
            return browser_cookie(response, TICKET_COOKIE, ticket, 600)
        ticket = request.query_params.get("ticket", "")
        link = records.get("source_login", ticket)
        if not link or link["reference"] != reference:
            return page("请求已失效", "<p>请重新打开引文。</p>", 403)
        username, failure = await browser_sign_in(request, ticket, "source")
        if failure is not None:
            return failure
        if username != found["subject"] or records.take("source_login", ticket) is None:
            return page("权限不足", "<p>请使用发起检索的知识库账号。</p>", 403)
        session = new_secret()
        records.put("browser", session, {"subject": username}, 3600)
        return browser_cookie(render_source(found), BROWSER_COOKIE, session, 3600)

    def render_source(found):
        kb = found["metadata"]["kb"]
        return page(found["title"], f'<p><a href="/{kb}/webui/">{TITLES[kb]}</a></p><pre>{escape(found["text"])}</pre>')

    async def icon(_request):
        return FileResponse(Path(__file__).with_name("icon.png"), media_type="image/png")

    async def discovery(request):
        metadata = build_metadata(issuer, AnyHttpUrl(origin), registration, revocation)
        metadata.token_endpoint_auth_methods_supported.append("none")
        return await MetadataHandler(metadata).handle(request)

    authentication_routes = create_auth_routes(provider, issuer, client_registration_options=registration, revocation_options=revocation)
    routes = [
        Route(ISSUER_PATH + route.path, endpoint=route.endpoint, methods=sorted(route.methods))
        for route in authentication_routes if not route.path.startswith("/.well-known") and route.path != "/token"
    ]
    routes.extend(create_protected_resource_routes(
        resource, [issuer], SCOPES, resource_name="Echem 知识库",
    ))
    routes.extend([
        Route("/.well-known/oauth-authorization-server" + ISSUER_PATH, discovery),
        Route(TOKEN_PATH, token_endpoint, methods=["POST"]),
        Route("/mcp", RequireAuthMiddleware(session_manager.handle_request, [READ], metadata_url), methods=["GET", "POST", "DELETE"]),
        Route("/mcp/consent", consent, methods=["GET", "POST"]),
        Route("/mcp/approve/{ticket}", approve, methods=["GET", "POST"]),
        Route("/mcp/source/{reference}", source, methods=["GET", "POST"]),
        Route(ICON_PATH, icon, methods=["GET"]),
    ])
    return Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[
            Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(provider, resource_server_url=resource)),
            Middleware(AuthContextMiddleware),
        ],
    )


def production_app():
    return create_app(
        origin=os.environ["MCP_ORIGIN"].rstrip("/"),
        state_path=os.environ["MCP_STATE_DB"],
        api_keys={kb: os.environ["MCP_" + kb.replace("-", "_").upper() + "_KEY"] for kb in KNOWLEDGE_BASES},
        admin_users=set(os.environ["MCP_ADMIN_USERS"].split(",")),
    )

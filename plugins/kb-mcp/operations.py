"""The public LightRAG operations offered by the MCP adapter."""

from dataclasses import dataclass
from urllib.parse import quote

READ, WRITE, ADMIN = "kb:read", "kb:write", "kb:admin"
SCOPES = [READ, WRITE, ADMIN]
KB_INPUT = {"type": "string", "enum": ["battery-life", "lead-acid"]}


@dataclass(frozen=True)
class Operation:
    name: str
    method: str
    path: str
    scope: str = READ
    destructive: bool = False
    confirm: bool = False


OPERATIONS = (
    Operation("scan_documents", "post", "/documents/scan", WRITE),
    Operation("get_scan_status", "get", "/documents/scan/status/{track_id}"),
    Operation("list_source_conflicts", "get", "/documents/source_conflicts"),
    Operation("repair_source_conflict", "post", "/documents/source_conflicts/repair", ADMIN, True, True),
    Operation("upload_document", "post", "/documents/upload", WRITE),
    Operation("insert_text", "post", "/documents/text", WRITE),
    Operation("insert_texts", "post", "/documents/texts", WRITE),
    Operation("clear_documents", "delete", "/documents", ADMIN, True, True),
    Operation("get_pipeline_status", "get", "/documents/pipeline_status"),
    Operation("delete_documents", "delete", "/documents/delete_document", WRITE, True),
    Operation("get_document_status", "get", "/documents/track_status/{track_id}"),
    Operation("list_documents", "post", "/documents/paginated"),
    Operation("get_document_counts", "get", "/documents/status_counts"),
    Operation("list_file_types", "get", "/documents/supported_file_types"),
    Operation("retry_failed_documents", "post", "/documents/reprocess_failed", WRITE),
    Operation("force_reset_recovery", "post", "/documents/recovery/force_reset", ADMIN, True, True),
    Operation("cancel_pipeline", "post", "/documents/cancel_pipeline", WRITE, True),
    Operation("ask_kb", "post", "/query"),
    Operation("ask_kb_stream", "post", "/query/stream"),
    Operation("retrieve_kb_data", "post", "/query/data"),
    Operation("list_graph_labels", "get", "/graph/label/list"),
    Operation("list_popular_graph_labels", "get", "/graph/label/popular"),
    Operation("search_graph_labels", "get", "/graph/label/search"),
    Operation("get_knowledge_graph", "get", "/graphs"),
    Operation("graph_entity_exists", "get", "/graph/entity/exists"),
    Operation("edit_graph_entity", "post", "/graph/entity/edit", WRITE, True),
    Operation("edit_graph_relation", "post", "/graph/relation/edit", WRITE, True),
    Operation("create_graph_entity", "post", "/graph/entity/create", WRITE),
    Operation("create_graph_relation", "post", "/graph/relation/create", WRITE),
    Operation("merge_graph_entities", "post", "/graph/entities/merge", WRITE, True),
    Operation("delete_graph_entity", "delete", "/graph/entity/delete", WRITE, True),
    Operation("delete_graph_relation", "delete", "/graph/relation/delete", WRITE, True),
    Operation("get_ui_customization", "get", "/ui/customization"),
    Operation("get_ui_asset", "get", "/ui/customization/assets/{asset_hash}/{asset_id}"),
    Operation("get_ollama_version", "get", "/api/version"),
    Operation("list_ollama_models", "get", "/api/tags"),
    Operation("list_running_ollama_models", "get", "/api/ps"),
    Operation("ollama_generate", "post", "/api/generate"),
    Operation("ollama_chat", "post", "/api/chat"),
    Operation("get_webui_url", "get", "/"),
    Operation("verify_kb_credentials", "get", "/auth/verify"),
    Operation("get_kb_auth_status", "get", "/auth-status"),
    Operation("get_kb_health", "get", "/health"),
)

# Login is bound to the browser authorization flow. The model never handles passwords.
OAUTH_OPERATION = ("post", "/login")

FILE_INPUT = {
    "type": "object",
    "properties": {
        "download_url": {"type": "string", "maxLength": 8192},
        "file_id": {"type": "string"},
        "mime_type": {"type": "string", "maxLength": 128},
        "file_name": {"type": "string", "maxLength": 255},
    },
    "required": ["download_url", "file_id"],
    "additionalProperties": False,
}

OLLAMA_INPUT = {
    "model": {"type": "string"},
    "stream": {"type": "boolean", "default": False},
    "options": {"type": "object"},
    "system": {"type": "string"},
}


def _tool_schema(openapi, operation):
    details = openapi["paths"][operation.path][operation.method]
    definitions = {}
    components = openapi["components"]["schemas"]

    def resolve(value):
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref")
        if ref:
            name = ref.removeprefix("#/components/schemas/")
            if name not in definitions:
                definitions[name] = {}
                definitions[name] = resolve(components[name])
            return {"$ref": "#/$defs/" + name}
        return {
            key: resolve(item)
            for key, item in value.items()
            if key not in ("example", "examples")
        }

    properties = {"kb": KB_INPUT}
    required = ["kb"]
    for parameter in details.get("parameters", []):
        name = parameter["name"]
        properties[name] = resolve(parameter["schema"])
        if parameter["in"] == "path":
            properties[name].update(maxLength=512, pattern=r"^(?!\.{1,2}$)[^/\\%]+$")
        if parameter.get("required"):
            required.append(name)
    request = details.get("requestBody", {})
    if operation.name == "upload_document":
        properties["file"] = FILE_INPUT
        required.append("file")
    elif "application/json" in request.get("content", {}):
        properties["body"] = resolve(request["content"]["application/json"]["schema"])
        if request.get("required"):
            required.append("body")
    elif operation.name in ("ollama_generate", "ollama_chat"):
        field = "prompt" if operation.name == "ollama_generate" else "messages"
        field_schema = {"type": "string"} if field == "prompt" else {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"}, "content": {"type": "string"},
                    "images": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role", "content"],
            },
        }
        properties["body"] = {
            "type": "object",
            "properties": {**OLLAMA_INPUT, field: field_schema},
            "required": ["model", field],
        }
        required.append("body")
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if definitions:
        schema["$defs"] = definitions
    return schema


def load_contract(openapi):
    expected = {(op.method, op.path) for op in OPERATIONS} | {OAUTH_OPERATION}
    advertised = {
        (method, path)
        for path, methods in openapi["paths"].items()
        for method in methods
        if method in {"get", "post", "put", "delete", "patch", "head", "options"}
    }
    if expected != advertised:
        raise ValueError(f"LightRAG API changed. Missing: {expected - advertised}. New: {advertised - expected}.")
    return {op.name: _tool_schema(openapi, op) for op in OPERATIONS}


def upstream_request(operation, arguments):
    path = operation.path
    for key in arguments.keys() - {"kb", "body", "file"}:
        placeholder = "{" + key + "}"
        if placeholder in path:
            path = path.replace(placeholder, quote(str(arguments[key]), safe=""))
    query = {
        key: value
        for key, value in arguments.items()
        if key not in {"kb", "body", "file"} and value is not None
        and "{" + key + "}" not in operation.path
    }
    return path, query

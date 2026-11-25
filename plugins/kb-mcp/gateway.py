"""Requests to the existing LightRAG instances and ChatGPT attachment transfer."""

import base64
import json
import tempfile
from pathlib import PurePath
from urllib.parse import urlsplit

import httpx

KNOWLEDGE_BASES = {
    "battery-life": "http://127.0.0.1:9621/battery-life",
    "lead-acid": "http://127.0.0.1:9622/lead-acid",
}
MAX_UPLOAD = 100 * 1024 * 1024
MAX_RESPONSE = 8 * 1024 * 1024


class GatewayError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class LightRAGGateway:
    def __init__(self, http, api_keys, origin):
        self.http = http
        self.keys = api_keys
        self.origin = origin

    async def authenticate(self, username, password, kb="battery-life"):
        response = await self.http.post(
            KNOWLEDGE_BASES[kb] + "/login",
            data={"username": username, "password": password},
            timeout=8,
        )
        if response.status_code == 429:
            raise GatewayError(429, "Too many sign-in attempts.")
        if response.status_code >= 500:
            raise GatewayError(502, "LightRAG sign-in is temporarily unavailable.")
        if response.status_code != 200:
            return False
        data = response.json()
        return bool(data.get("access_token")) and data.get("auth_mode") != "disabled"

    async def request(self, kb, method, path, *, body=None, query=None, files=None):
        url = KNOWLEDGE_BASES[kb] + path
        try:
            async with self.http.stream(
                method, url, json=body, params=query, files=files,
                headers={"X-API-Key": self.keys[kb]},
            ) as response:
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(buffer) + len(chunk) > MAX_RESPONSE:
                        raise GatewayError(413, "LightRAG returned too much data. Request a shorter answer or fewer results.")
                    buffer.extend(chunk)
                status = response.status_code
                location = response.headers.get("location")
        except httpx.TimeoutException as error:
            raise GatewayError(504, "LightRAG timed out. Check status before retrying a write.") from error
        except httpx.HTTPError as error:
            raise GatewayError(502, "The LightRAG instance is unavailable.") from error

        if status >= 400:
            if status >= 500:
                raise GatewayError(status, "LightRAG could not complete the operation.")
            try:
                detail = json.loads(buffer).get("detail")
            except (ValueError, AttributeError):
                detail = None
            if isinstance(detail, list):
                detail = "; ".join(
                    " / ".join(map(str, item.get("loc", []))) + ": " + item.get("msg", "invalid input")
                    for item in detail if isinstance(item, dict)
                )
            raise GatewayError(status, detail[:2000] if isinstance(detail, str) else "LightRAG rejected the request.")

        if 300 <= status < 400:
            if path == "/" and location and location.startswith("/"):
                return {"url": self.origin + f"/{kb}/webui/"}
            raise GatewayError(502, "LightRAG returned an unexpected redirect.")
        if not buffer:
            return {"status": status}
        if content_type in {"application/x-ndjson", "application/ndjson"}:
            try:
                return {"lines": [json.loads(line) for line in buffer.splitlines() if line.strip()]}
            except ValueError as error:
                raise GatewayError(502, "LightRAG returned an invalid response stream.") from error
        if content_type == "application/json":
            try:
                payload = json.loads(buffer)
            except ValueError as error:
                raise GatewayError(502, "LightRAG returned invalid JSON.") from error
            if path == "/auth-status" and isinstance(payload, dict):
                payload.pop("access_token", None)
            return payload
        if path.startswith("/ui/customization/assets/"):
            return {"content_type": content_type, "data_base64": base64.b64encode(buffer).decode("ascii")}
        raise GatewayError(502, "LightRAG returned an unexpected content type.")

    async def upload(self, kb, file):
        url = urlsplit(file["download_url"])
        if (
            url.scheme != "https" or url.username or url.password
            or not url.hostname or not url.hostname.endswith(".oaiusercontent.com")
            or url.netloc.lower() not in {url.hostname, f"{url.hostname}:443"}
        ):
            raise GatewayError(400, "Use a ChatGPT attachment from its file transfer service.")
        filename = file.get("file_name") or PurePath(url.path).name
        if not filename or filename in {".", ".."} or any(c in filename for c in "/\\\r\n\0"):
            raise GatewayError(400, "The attachment has an invalid filename.")
        try:
            with tempfile.TemporaryFile() as spool:
                async with self.http.stream("GET", file["download_url"], timeout=30) as response:
                    if response.status_code != 200:
                        raise GatewayError(400, "ChatGPT could not supply this attachment. Select it again.")
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_UPLOAD:
                            raise GatewayError(413, "This file exceeds LightRAG's 100 MiB upload limit.")
                        spool.write(chunk)
                    media_type = file.get("mime_type") or response.headers.get("content-type", "application/octet-stream")
                spool.seek(0)
                return await self.request(kb, "post", "/documents/upload", files={"file": (filename, spool, media_type)})
        except httpx.HTTPError as error:
            raise GatewayError(502, "Could not transfer the ChatGPT attachment.") from error

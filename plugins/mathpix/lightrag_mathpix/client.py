"""Mathpix v3 PDF submission, status polling, and result download."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from pathlib import Path
from threading import Event
from typing import Any

import httpx

from lightrag.parser.exceptions import ParsePipelineCancelled
from lightrag.parser.external._common import (
    download_deadline_seconds,
    download_max_bytes,
    download_timeout,
    stream_capped_get,
)

from . import CREDENTIAL_NAMES

API_URL = "https://api.mathpix.com"
RESULT_FORMAT = "md.zip"
LAYOUT_MAX_BYTES = 64 * 1024 * 1024
CONVERSION_OPTIONS = {
    "conversion_formats": {RESULT_FORMAT: True},
    "math_inline_delimiters": ["$", "$"],
    "math_display_delimiters": ["$$", "$$"],
}
_PDF_ID = re.compile(r"[A-Za-z0-9_-]+")
_ERROR_ID = re.compile(r"[a-z0-9_]+")


class MathpixTaskFailed(RuntimeError):
    """Mathpix has confirmed a terminal failure for the saved task."""


def check_cancelled(event: Event | None) -> None:
    if event is not None and event.is_set():
        raise ParsePipelineCancelled("Mathpix parsing was cancelled")


def valid_pdf_id(value: object) -> bool:
    return isinstance(value, str) and bool(_PDF_ID.fullmatch(value))


def _check_response(response: httpx.Response, operation: str) -> None:
    # Response bodies can contain document excerpts and are not safe to log.
    if not response.is_success:
        raise RuntimeError(f"Mathpix {operation} returned HTTP {response.status_code}")


def _read_response(response: httpx.Response, operation: str) -> dict[str, Any]:
    _check_response(response, operation)
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(f"Mathpix {operation} returned invalid JSON") from None
    if not isinstance(body, dict):
        raise RuntimeError(f"Mathpix {operation} returned an invalid response")
    if body.get("error") and body.get("status") != "error":
        raise RuntimeError(f"Mathpix {operation} failed ({_error_id(body)})")
    return body


def _error_id(body: dict[str, Any]) -> str:
    detail = body.get("error_info")
    code = detail.get("id") if isinstance(detail, dict) else None
    return code if isinstance(code, str) and _ERROR_ID.fullmatch(code) else "unknown"


def _positive_seconds(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


class MathpixClient:
    def __init__(self) -> None:
        self.app_id, key = (os.getenv(name, "").strip() for name in CREDENTIAL_NAMES)
        if not self.app_id or not key:
            raise RuntimeError("Mathpix requires MATHPIX_APP_ID and MATHPIX_APP_KEY")
        self.headers = {"app_id": self.app_id, "app_key": key}
        self.poll_interval = _positive_seconds("MATHPIX_POLL_INTERVAL_SECONDS", "3")
        self.poll_timeout = _positive_seconds("MATHPIX_POLL_TIMEOUT_SECONDS", "600")

    def open(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=API_URL,
            headers=self.headers,
            timeout=httpx.Timeout(30, connect=10),
            follow_redirects=False,
        )

    async def submit(
        self, client: httpx.AsyncClient, source: Path, document_name: str
    ) -> str:
        with source.open("rb") as pdf:
            response = await client.post(
                "/v3/pdf",
                files={"file": (document_name, pdf, "application/pdf")},
                data={"options_json": json.dumps(CONVERSION_OPTIONS)},
            )
        payload = _read_response(response, "submission")
        pdf_id = payload.get("pdf_id")
        if not valid_pdf_id(pdf_id):
            raise RuntimeError("Mathpix submission did not return a valid PDF ID")
        return pdf_id

    async def _wait(
        self,
        client: httpx.AsyncClient,
        path: str,
        *,
        event: Event | None,
        converted: bool,
    ) -> None:
        while True:
            check_cancelled(event)
            response = await client.get(path)
            body = _read_response(response, "status")
            status = body.get("status")
            if status == "error":
                if converted:
                    raise RuntimeError("Mathpix Markdown ZIP conversion failed")
                raise MathpixTaskFailed(f"Mathpix task failed ({_error_id(body)})")
            if converted:
                formats = body.get("conversion_status") or {}
                result = formats.get(RESULT_FORMAT) or {}
                if result.get("status") in {"error", "failed"}:
                    raise RuntimeError("Mathpix Markdown ZIP conversion failed")
                if result.get("status") == "completed":
                    return
            elif status == "completed":
                return

            end = asyncio.get_running_loop().time() + self.poll_interval
            while (remaining := end - asyncio.get_running_loop().time()) > 0:
                check_cancelled(event)
                await asyncio.sleep(min(remaining, 0.5))

    async def download(
        self, client: httpx.AsyncClient, pdf_id: str, *, event: Event | None
    ) -> bytes:
        try:
            async with asyncio.timeout(self.poll_timeout):
                await self._wait(
                    client, f"/v3/pdf/{pdf_id}", event=event, converted=False
                )
                await self._wait(
                    client, f"/v3/converter/{pdf_id}", event=event, converted=True
                )
        except TimeoutError:
            raise TimeoutError(
                f"Mathpix task timed out ({pdf_id}); retry will check it again"
            ) from None
        check_cancelled(event)
        async with download_timeout(download_deadline_seconds()):
            response, content = await stream_capped_get(
                client,
                f"/v3/pdf/{pdf_id}.{RESULT_FORMAT}",
                max_bytes=download_max_bytes(),
                operation="Mathpix Markdown ZIP download",
            )
        _check_response(response, "download")
        return content

    async def download_layout(self, pdf_id: str, *, event: Event | None) -> bytes:
        check_cancelled(event)
        async with self.open() as client, download_timeout(10):
            response, content = await stream_capped_get(
                client,
                f"/v3/pdf/{pdf_id}.lines.json",
                max_bytes=LAYOUT_MAX_BYTES,
                operation="Mathpix PDF layout download",
            )
        _check_response(response, "layout download")
        check_cancelled(event)
        return content

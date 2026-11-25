"""Convert Mathpix Markdown and bundled images to LightRAG sidecars."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Event
import time
from typing import Any
from zipfile import BadZipFile

import httpx

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.parser.base import BaseParser, ParseContext, ParseResult
from lightrag.parser.external._common import (
    compute_size_and_hash,
    download_max_bytes,
    env_int,
)
from lightrag.parser.external._zip import result_bundle_limits, safe_extract_zip
from lightrag.parser.markdown.extract import extract_markdown
from lightrag.parser.markdown.ir_builder import NativeMarkdownIRBuilder
from lightrag.parser.markdown.parser import (
    DEFAULT_NATIVE_MD_IMAGE_MAX_TOTAL_BYTES,
    _MarkdownImageResolver,
)
from lightrag.sidecar import write_sidecar
from lightrag.sidecar.ir import IRDoc
from lightrag.utils import logger
from lightrag.utils_pipeline import make_lightrag_doc_content, sidecar_uri_for

from .client import (
    CONVERSION_OPTIONS,
    LAYOUT_MAX_BYTES,
    MathpixClient,
    MathpixTaskFailed,
    check_cancelled,
    valid_pdf_id,
)
from .layout import attach_figure_context, read_pages


def _read_job(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        raise RuntimeError("Invalid saved Mathpix task record") from None
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("signature"), str)
        or not valid_pdf_id(record.get("pdf_id"))
    ):
        raise RuntimeError("Invalid saved Mathpix task record")
    return record


def _atomic_write(path: Path, data: bytes) -> None:
    staging = path.with_suffix(".tmp")
    staging.write_bytes(data)
    os.replace(staging, path)


def _read_bundle(path: Path) -> bytes:
    max_bytes = download_max_bytes()
    with path.open("rb") as archive:
        if max_bytes is None:
            return archive.read()
        content = archive.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("Cached Mathpix bundle exceeds the configured download limit")
    return content


def _build_ir(
    payload: bytes, cache_dir: Path, document_name: str, event: Event | None
) -> tuple[IRDoc, str]:
    check_cancelled(event)
    max_entries, max_total_bytes = result_bundle_limits()
    with TemporaryDirectory(prefix="unpack-", dir=cache_dir) as extracted:
        root = Path(extracted)
        safe_extract_zip(
            payload, root, max_entries=max_entries, max_total_bytes=max_total_bytes
        )
        markdown = list(root.rglob("*.md"))
        if len(markdown) != 1:
            raise ValueError("Mathpix result must contain one Markdown document")
        md_path = markdown[0]
        resolver = _MarkdownImageResolver(
            bundle_root=md_path.parent,
            warnings={},
            download_enabled=False,
            download_required=False,
            timeout=0,
            max_bytes=env_int("NATIVE_MD_IMAGE_MAX_BYTES", 25 * 1024 * 1024),
            max_svg_pixels=env_int("NATIVE_MD_IMAGE_MAX_SVG_PIXELS", 16_000_000),
            max_total_bytes=env_int(
                "NATIVE_MD_IMAGE_MAX_TOTAL_BYTES",
                DEFAULT_NATIVE_MD_IMAGE_MAX_TOTAL_BYTES,
            ),
            max_requests=0,
            total_timeout=0,
            cancel_events=(event,),
        )
        md_text = md_path.read_text(encoding="utf-8-sig")
        parsed = extract_markdown(md_text, image_resolver=resolver)
    if not parsed.blocks:
        raise ValueError("Mathpix returned no usable document content")
    ir = NativeMarkdownIRBuilder().normalize(
        parsed.blocks,
        document_name=document_name,
        asset_dir_name=f"{Path(document_name).stem}.blocks.assets",
        parse_metadata={
            "md_tables": parsed.tables,
            "md_equations": parsed.equations,
            "md_drawings": parsed.drawings,
            "md_assets": parsed.assets,
        },
    )
    return ir, md_text


def _read_cached_layout(path: Path) -> dict[int, dict]:
    with path.open("rb") as layout:
        payload = layout.read(LAYOUT_MAX_BYTES + 1)
    if len(payload) > LAYOUT_MAX_BYTES:
        raise ValueError("Cached Mathpix layout exceeds the download limit")
    return read_pages(payload)


async def _enrich_figures(
    ir: IRDoc,
    markdown: str,
    service: MathpixClient,
    pdf_id: str,
    path: Path,
    event: Event | None,
) -> None:
    if not any(block.drawings for block in ir.blocks):
        return
    try:
        pages = await asyncio.to_thread(_read_cached_layout, path)
    except (FileNotFoundError, ValueError):
        try:
            payload = await service.download_layout(pdf_id, event=event)
            pages = await asyncio.to_thread(read_pages, payload)
        except (RuntimeError, httpx.HTTPError, TimeoutError, ValueError) as exc:
            check_cancelled(event)
            logger.warning(
                "Mathpix figure layout unavailable (%s); using Markdown",
                type(exc).__name__,
            )
            return
        await asyncio.to_thread(_atomic_write, path, payload)
    check_cancelled(event)
    await asyncio.to_thread(attach_figure_context, ir, markdown, pages)


class MathpixParser(BaseParser):
    """Keep the paid task and its validated ZIP under the document's sidecar."""

    engine_name = "mathpix"

    async def parse(self, ctx: ParseContext) -> ParseResult:
        check_cancelled(ctx.pipeline_cancel_event)
        source = ctx.resolve(self.engine_name)
        _, source_hash = await asyncio.to_thread(
            compute_size_and_hash, source.source_path
        )
        service = MathpixClient()
        signature = hashlib.sha256(
            json.dumps([source_hash, service.app_id, CONVERSION_OPTIONS]).encode(
                "utf-8"
            )
        ).hexdigest()
        # Use the standard .parsed tree so delete-file also removes the task and
        # raw result. Sidecar regeneration therefore has to preserve this folder.
        cache_dir = source.parsed_dir / ".mathpix"
        cache_dir.mkdir(parents=True, exist_ok=True)
        job_path = cache_dir / "job.json"
        result_path = cache_dir / "result.zip"
        layout_path = cache_dir / "lines.json"
        job = _read_job(job_path)
        if job is None or job["signature"] != signature:
            result_path.unlink(missing_ok=True)
            layout_path.unlink(missing_ok=True)
            job = None

        cached = job is not None and result_path.is_file()
        if cached:
            try:
                payload = await asyncio.to_thread(_read_bundle, result_path)
                ir, markdown = await asyncio.to_thread(
                    _build_ir,
                    payload,
                    cache_dir,
                    source.document_name,
                    ctx.pipeline_cancel_event,
                )
            except BadZipFile:
                result_path.unlink()
                cached = False
        if not cached:
            try:
                async with service.open() as client:
                    if job is None:
                        check_cancelled(ctx.pipeline_cancel_event)
                        pdf_id = await service.submit(
                            client, source.source_path, source.document_name
                        )
                        job = {"signature": signature, "pdf_id": pdf_id}
                        _atomic_write(job_path, json.dumps(job).encode("utf-8"))
                    payload = await service.download(
                        client, job["pdf_id"], event=ctx.pipeline_cancel_event
                    )
            except MathpixTaskFailed:
                # A confirmed terminal failure can be submitted again by a
                # subsequent explicit retry. Timeouts keep the saved task.
                job_path.unlink(missing_ok=True)
                layout_path.unlink(missing_ok=True)
                raise
            except httpx.HTTPError as error:
                raise RuntimeError(
                    f"Mathpix connection failed ({type(error).__name__})"
                ) from None
            ir, markdown = await asyncio.to_thread(
                _build_ir,
                payload,
                cache_dir,
                source.document_name,
                ctx.pipeline_cancel_event,
            )
            await asyncio.to_thread(_atomic_write, result_path, payload)

        assert job is not None
        await _enrich_figures(
            ir, markdown, service, job["pdf_id"], layout_path, ctx.pipeline_cancel_event
        )
        check_cancelled(ctx.pipeline_cancel_event)
        for artifact in source.parsed_dir.iterdir():
            if artifact == cache_dir:
                continue
            if artifact.is_dir() and not artifact.is_symlink():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
        parsed = write_sidecar(
            ir,
            parsed_dir=source.parsed_dir,
            doc_id=ctx.doc_id,
            engine=self.engine_name,
            clean_parsed_dir=False,
        )
        if not parsed["content"].strip():
            raise ValueError("Mathpix returned no usable document content")
        await ctx.rag._persist_parsed_full_docs(
            ctx.doc_id,
            {
                "content": make_lightrag_doc_content(parsed["content"]),
                "file_path": ctx.file_path,
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "sidecar_location": sidecar_uri_for(source.parsed_dir),
                "parse_engine": self.engine_name,
                "update_time": int(time.time()),
            },
        )
        await ctx.archive_source(str(source.source_path))
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_LIGHTRAG,
            content=parsed["content"],
            blocks_path=parsed["blocks_path"],
            parse_engine=self.engine_name,
            parse_stage_skipped=cached,
        )

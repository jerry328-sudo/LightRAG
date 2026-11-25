"""Serve the raster assets belonging to a workspace's parsed documents."""

import asyncio
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Path as PathParameter
from fastapi.responses import Response

from lightrag.utils_pipeline import resolve_sidecar_uri


IMAGE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@contextmanager
def _open_artifact(path):
    if path.resolve(strict=True) != path:
        raise FileNotFoundError
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise FileNotFoundError
        yield handle


def _valid_signature(data, media_type):
    if media_type == "image/webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return data.startswith({
        "image/jpeg": b"\xff\xd8\xff",
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/gif": (b"GIF87a", b"GIF89a"),
    }[media_type])


def _load_image(input_root, uri, document_id, image_id):
    if not isinstance(uri, str) or urlsplit(uri).netloc not in ("", "localhost"):
        raise FileNotFoundError
    location = resolve_sidecar_uri(uri)
    if location is None:
        raise FileNotFoundError
    directory = location.resolve(strict=True)
    if directory.parent not in (input_root, input_root / "__parsed__"):
        raise FileNotFoundError
    blocks = list(directory.glob("*.blocks.jsonl"))
    if len(blocks) != 1:
        raise FileNotFoundError
    with _open_artifact(blocks[0]) as handle:
        header = json.loads(handle.readline(65536))
    if not isinstance(header, dict) or header.get("type") != "meta" or header.get("doc_id") != document_id:
        raise FileNotFoundError
    stem = blocks[0].name.removesuffix(".blocks.jsonl")
    with _open_artifact(directory / f"{stem}.drawings.json") as handle:
        manifest = handle.read(_MAX_MANIFEST_BYTES + 1)
    if len(manifest) > _MAX_MANIFEST_BYTES:
        raise HTTPException(413, "The document image manifest is too large")
    manifest = json.loads(manifest)
    drawings = manifest.get("drawings") if isinstance(manifest, dict) else None
    item = drawings.get(image_id) if isinstance(drawings, dict) else None
    asset_path = item.get("path") if isinstance(item, dict) and item.get("id") == image_id else None
    if not isinstance(asset_path, str):
        raise FileNotFoundError
    parts = Path(asset_path).parts
    if len(parts) != 2 or parts[0] != f"{stem}.blocks.assets" or parts[1] == "..":
        raise FileNotFoundError
    asset = directory.joinpath(*parts)
    media_type = _MEDIA_TYPES.get(asset.suffix.lower())
    if media_type is None:
        raise HTTPException(415, "This image format cannot be returned")
    with _open_artifact(asset) as handle:
        if os.fstat(handle.fileno()).st_size > MAX_IMAGE_BYTES:
            raise HTTPException(413, "The image exceeds the 4 MiB response limit")
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "The image exceeds the 4 MiB response limit")
    if not _valid_signature(data, media_type):
        raise HTTPException(415, "The image does not match its declared format")
    return data, media_type


def register_image_route(router: APIRouter, rag, input_dir: Path):
    input_root = Path(input_dir).resolve()

    @router.get(
        "/{document_id}/images/{image_id}",
        summary="Read an extracted image from a document",
        response_class=Response,
        responses={
            200: {"content": {mime: {"schema": {"type": "string", "format": "binary"}} for mime in set(_MEDIA_TYPES.values())}},
            404: {"description": "Document or image was not found"},
            413: {"description": "Image exceeds the 4 MiB limit"},
            415: {"description": "Unsupported image format"},
        },
    )
    async def get_document_image(
        document_id: str,
        image_id: Annotated[str, PathParameter(pattern=IMAGE_ID_PATTERN)],
    ):
        """Use the image_id from an image hit in semantic retrieval."""
        statuses, record = await asyncio.gather(
            rag.doc_status.get_full_docs_by_ids([document_id], strict=True),
            rag.full_docs.get_by_id(document_id),
        )
        status = statuses.get(document_id)
        if not record or status is None or status.status == "deleting" or (status.metadata or {}).get("is_duplicate"):
            raise HTTPException(404, "Document not found")
        try:
            data, media_type = await asyncio.to_thread(
                _load_image, input_root, record.get("sidecar_location"), document_id, image_id
            )
        except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
            raise HTTPException(404, "Extracted image not found") from exc
        return Response(
            data,
            media_type=media_type,
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )

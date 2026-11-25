import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException

from extensions.document_metadata import images


pytestmark = pytest.mark.offline

DOCUMENT_ID = "doc-paper"
IMAGE_ID = "im-figure-1"
IMAGE_PATH = f"/documents/{DOCUMENT_ID}/images/{IMAGE_ID}"
JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")


@pytest.fixture
def image_api(tmp_path):
    input_root = tmp_path / "inputs"
    parsed = input_root / "__parsed__" / "paper.pdf.parsed"
    assets = parsed / "paper.blocks.assets"
    assets.mkdir(parents=True)
    image = assets / "figure.jpg"
    image.write_bytes(JPEG)
    (parsed / "paper.blocks.jsonl").write_text(
        json.dumps({"type": "meta", "doc_id": DOCUMENT_ID}) + "\n"
    )

    def write_manifest(path):
        (parsed / "paper.drawings.json").write_text(
            json.dumps({"drawings": {IMAGE_ID: {"id": IMAGE_ID, "path": path}}})
        )

    write_manifest("paper.blocks.assets/figure.jpg")
    records = {DOCUMENT_ID: {"sidecar_location": parsed.as_uri()}}
    get_records = AsyncMock(side_effect=records.get)
    status = SimpleNamespace(status="processed", metadata={})
    rag = SimpleNamespace(
        doc_status=SimpleNamespace(get_full_docs_by_ids=AsyncMock(return_value={DOCUMENT_ID: status})),
        full_docs=SimpleNamespace(get_by_id=get_records),
    )

    async def authorize(x_api_key: str | None = Header(default=None)):
        if x_api_key != "fixture-token":
            raise HTTPException(401)

    app = FastAPI()
    router = APIRouter(prefix="/documents", dependencies=[Depends(authorize)])
    images.register_image_route(router, rag, input_root)
    app.include_router(router)
    return app, image, write_manifest


async def test_read_image_returns_stored_bytes_only_after_authentication(image_api):
    app, _, _ = image_api
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.get(IMAGE_PATH)).status_code == 401
        reply = await client.get(IMAGE_PATH, headers={"X-API-Key": "fixture-token"})
        missing = await client.get(IMAGE_PATH.replace(DOCUMENT_ID, "doc-other"), headers={"X-API-Key": "fixture-token"})
    assert reply.status_code == 200
    assert reply.headers["content-type"] == "image/jpeg"
    assert reply.headers["cache-control"] == "private, no-store"
    assert reply.content == JPEG
    assert missing.status_code == 404


@pytest.mark.parametrize("artifact", ["traversal", "symlink", "wrong-format", "oversized"])
async def test_read_image_rejects_unsafe_or_unsupported_artifacts(image_api, tmp_path, monkeypatch, artifact):
    app, image, write_manifest = image_api
    if artifact == "traversal":
        write_manifest("paper.blocks.assets/../../secret.jpg")
    elif artifact == "symlink":
        outside = tmp_path / "secret.jpg"
        outside.write_bytes(JPEG)
        image.unlink()
        image.symlink_to(outside)
    elif artifact == "wrong-format":
        image.write_bytes(b"<html>not a JPEG</html>")
    else:
        monkeypatch.setattr(images, "MAX_IMAGE_BYTES", len(JPEG) - 1)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.get(IMAGE_PATH, headers={"X-API-Key": "fixture-token"})
    assert response.status_code == {"traversal": 404, "symlink": 404, "wrong-format": 415, "oversized": 413}[artifact]

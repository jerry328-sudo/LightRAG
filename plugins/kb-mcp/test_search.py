import httpx
import pytest

import app as gateway_app
from auth import OAuthProvider, Records


pytestmark = pytest.mark.offline


async def test_search_kb_and_fetch_preserve_extracted_image_identity(tmp_path, monkeypatch):
    origin = "https://kb.example"
    state = tmp_path / "oauth.sqlite3"
    token = OAuthProvider(origin, Records(state)).issue("fixture-client", "reader", ["kb:read"])
    monkeypatch.setattr(gateway_app, "OPERATIONS", ())
    monkeypatch.setattr(gateway_app, "load_contract", lambda _: {})

    def upstream(request):
        if request.url.path.endswith("/openapi.json"):
            return httpx.Response(200, json={"paths": {}})
        assert request.url.path.endswith("/query/data")
        chunks = [
            {
                "file_path": "cell-study.pdf", "content": "Impedance spectra description",
                "chunk_id": "drawing-chunk", "document_id": "doc-study", "image_id": "im-spectrum",
            },
            {"file_path": "notes.pdf", "content": "Ordinary text", "chunk_id": "text-chunk"},
        ] if request.url.path.startswith("/battery-life/") else []
        return httpx.Response(200, json={"data": {"chunks": chunks}})

    app = gateway_app.create_app(
        origin=origin,
        state_path=state,
        api_keys={"battery-life": "fixture-key", "lead-acid": "fixture-key"},
        admin_users=set(),
        upstream_transport=httpx.MockTransport(upstream),
    )
    headers = {"Authorization": "Bearer " + token.access_token, "Accept": "application/json, text/event-stream"}
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url=origin, headers=headers,
    ) as client:
        async def call(name, arguments):
            response = await client.post("/mcp", json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            })
            assert response.status_code == 200
            result = response.json()["result"]
            assert not result.get("isError")
            return result["structuredContent"]

        query = {"query": "cell impedance"}
        image, text = (await call("search_kb", {"kb": "battery-life", **query}))["results"]
        assert image["document_id"] == "doc-study" and image["image_id"] == "im-spectrum"
        assert set(text) == {"id", "title", "url"}
        excerpt = await call("fetch", {"id": image["id"]})
        assert excerpt["text"] == "Impedance spectra description"
        assert excerpt["metadata"] == {
            "kb": "battery-life", "chunk_id": "drawing-chunk",
            "document_id": "doc-study", "image_id": "im-spectrum",
        }

        image, _ = (await call("search", query))["results"]
        assert set(image) == {"id", "title", "url"}
        assert (await call("fetch", {"id": image["id"]}))["metadata"] == excerpt["metadata"]

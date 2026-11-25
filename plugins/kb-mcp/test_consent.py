import base64
import hashlib
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from app import create_app

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback", "client_name", "auth_method"),
    [
        ("https://claude.ai/api/mcp/auth_callback", "Claude", "client_secret_post"),
        ("https://chatgpt.com/connector_platform_oauth_redirect", "ChatGPT", "none"),
    ],
)
async def test_consent_allows_the_registered_client_callback(
    tmp_path, callback, client_name, auth_method
):
    origin = "https://kb.example"
    resource = origin + "/mcp"

    def log_in(request):
        assert request.url.path == "/battery-life/login"
        assert parse_qs(request.content.decode()) == {
            "username": ["test-reader"], "password": ["fixture-password"]
        }
        return httpx.Response(200, json={"access_token": "fixture-token"})

    app = create_app(
        origin=origin,
        state_path=tmp_path / "oauth.sqlite3",
        api_keys={},
        admin_users=set(),
        upstream_transport=httpx.MockTransport(log_in),
    )
    verifier = "fixture-verifier" * 3
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=origin) as client:
        registration = await client.post("/mcp/oauth/register", json={
            "redirect_uris": [callback],
            "grant_types": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_method": auth_method,
            "scope": "kb:read",
        })
        assert registration.status_code == 201
        registered = registration.json()
        start = await client.get("/mcp/oauth/authorize", params={
            "client_id": registered["client_id"],
            "redirect_uri": callback,
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": resource,
            "scope": "kb:read",
            "state": "fixture-state",
        })
        consent_url = start.headers["location"]
        consent = await client.get(consent_url)
        assert consent.status_code == 200
        form_action = next(
            directive.strip()
            for directive in consent.headers["Content-Security-Policy"].split(";")
            if directive.strip().startswith("form-action ")
        )
        assert form_action.split() == ["form-action", "'self'", f"https://{urlsplit(callback).netloc}"]
        assert f"你正在连接 {client_name}。" in consent.text

        grant = await client.post(consent_url, headers={"Origin": origin}, data={
            "ticket": parse_qs(urlsplit(consent_url).query)["ticket"][0],
            "username": "test-reader",
            "password": "fixture-password",
        })
        assert grant.status_code == 303
        destination, _, query = grant.headers["location"].partition("?")
        assert destination == callback
        parameters = parse_qs(query)
        assert parameters["state"] == ["fixture-state"]

        token_request = {
            "grant_type": "authorization_code",
            "code": parameters["code"][0],
            "client_id": registered["client_id"],
            "redirect_uri": callback,
            "code_verifier": verifier,
            "resource": resource,
        }
        if "client_secret" in registered:
            token_request["client_secret"] = registered["client_secret"]
        token = await client.post("/mcp/oauth/token", data=token_request)
        assert token.status_code == 200
        assert token.json()["scope"] == "kb:read"

from types import SimpleNamespace
from unittest.mock import Mock

from mcp.server.auth.provider import RegistrationError
import pytest

from auth import OAuthProvider


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redirect",
    [
        "https://claude.ai/api/mcp/auth_callback",
        "https://chatgpt.com/connector_platform_oauth_redirect",
        "https://chatgpt.com/connector/oauth/callback_abc",
    ],
)
async def test_registration_accepts_known_chat_client_callbacks(redirect):
    records = Mock()
    payload = {"client_id": "registered-client", "redirect_uris": [redirect]}
    client = SimpleNamespace(
        client_id=payload["client_id"],
        redirect_uris=payload["redirect_uris"],
        model_dump=Mock(return_value=payload),
    )
    await OAuthProvider("https://kb.example", records).register_client(client)
    records.put.assert_called_once_with(
        "client", payload["client_id"], payload, 365 * 86400, cap=500
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redirect",
    [
        "http://claude.ai/api/mcp/auth_callback",
        "https://claude.ai/api/mcp/auth_callback/",
        "https://claude.ai/api/mcp/auth_callback?next=x",
        "https://claude.ai/api/mcp/auth_callback#different",
        "https://claude.ai.example.org/api/mcp/auth_callback",
        "https://chatgpt.com/api/mcp/auth_callback",
    ],
)
async def test_registration_rejects_unapproved_redirects(redirect):
    records = Mock()
    client = SimpleNamespace(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback", redirect]
    )
    with pytest.raises(RegistrationError, match="invalid_redirect_uri"):
        await OAuthProvider("https://kb.example", records).register_client(client)
    records.put.assert_not_called()

"""OAuth state and the LightRAG account authorization adapter."""

import hashlib
import json
import re
import secrets
import sqlite3
import time
from contextlib import closing
from urllib.parse import urlencode, urlsplit

from mcp.server.auth.provider import (
    AccessToken, AuthorizationCode, AuthorizationParams, AuthorizeError,
    RefreshToken, RegistrationError, TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from operations import READ, SCOPES

ACCESS_SECONDS = 3600
REFRESH_SECONDS = 30 * 86400
CALLBACK_PATTERN = re.compile(r"/connector/oauth/[A-Za-z0-9_-]{1,160}")
CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"


def new_secret():
    return secrets.token_urlsafe(32)


class Records:
    def __init__(self, path):
        self.path = path
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, key TEXT NOT NULL, data TEXT NOT NULL,
                    expires INTEGER NOT NULL, PRIMARY KEY (kind, key)
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS records_expiry ON records (expires)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=3)

    @staticmethod
    def key(key):
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, kind, key):
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT data FROM records WHERE kind=? AND key=? AND expires>?",
                (kind, self.key(key), time.time()),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def change(self, *, write=(), consume=None, limit=None):
        now = int(time.time())
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM records WHERE expires<=?", (now,))
            if consume:
                deleted = db.execute(
                    "DELETE FROM records WHERE kind=? AND key=? RETURNING data",
                    (consume[0], self.key(consume[1])),
                ).fetchone()
                if not deleted:
                    return None
            if limit:
                kind, count = limit
                total = db.execute("SELECT count(*) FROM records WHERE kind=?", (kind,)).fetchone()[0]
                if total >= count:
                    return None
            db.executemany(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)",
                [(kind, self.key(key), json.dumps(value), now + ttl) for kind, key, value, ttl in write],
            )
        return json.loads(deleted[0]) if consume else True

    def put(self, kind, key, value, ttl, *, cap=None):
        limit = (kind, cap) if cap else None
        return self.change(write=[(kind, key, value, ttl)], limit=limit)

    def take(self, kind, key):
        return self.change(consume=(kind, key))


class OAuthProvider:
    def __init__(self, origin, records, *, allowed_callback_hosts=("chatgpt.com",)):
        self.origin = origin
        self.resource = origin + "/mcp"
        self.records = records
        self.allowed_callback_hosts = allowed_callback_hosts

    async def get_client(self, client_id):
        data = self.records.get("client", client_id)
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def register_client(self, client_info):
        for callback in client_info.redirect_uris or []:
            callback_uri = str(callback)
            if callback_uri == CLAUDE_CALLBACK:
                continue
            url = urlsplit(callback_uri)
            if not (
                url.scheme == "https"
                and url.netloc in self.allowed_callback_hosts
                and not (url.query or url.fragment)
                and (url.path == "/connector_platform_oauth_redirect" or CALLBACK_PATTERN.fullmatch(url.path))
            ):
                raise RegistrationError("invalid_redirect_uri", "Register a supported OAuth callback.")
        if not self.records.put("client", client_info.client_id, client_info.model_dump(mode="json"), 365 * 86400, cap=500):
            raise RegistrationError("invalid_client_metadata", "Registration capacity reached.")

    async def authorize(self, client, params):
        if params.resource != self.resource or not re.fullmatch(r"[A-Za-z0-9_-]{43}", params.code_challenge):
            raise AuthorizeError("invalid_request", "An exact MCP resource and S256 PKCE challenge are required.")
        scopes = params.scopes or [READ]
        if not set(scopes) <= set(SCOPES):
            raise AuthorizeError("invalid_scope", "The requested scope is unsupported.")
        ticket = new_secret()
        data = {"client_id": client.client_id, "params": params.model_dump(mode="json"), "scopes": scopes}
        if not self.records.put("pending", ticket, data, 600, cap=500):
            raise AuthorizeError("temporarily_unavailable", "Authorization capacity reached.")
        return self.origin + "/mcp/consent?" + urlencode({"ticket": ticket})

    def grant(self, ticket, subject):
        pending = self.records.take("pending", ticket)
        if pending is None:
            return None
        params = AuthorizationParams.model_validate(pending["params"])
        code = new_secret()
        data = AuthorizationCode(
            code=code, client_id=pending["client_id"], scopes=pending["scopes"],
            expires_at=time.time() + 120, code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri, redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource, subject=subject,
        )
        self.records.put("code", code, data.model_dump(mode="json", exclude={"code"}), 120)
        fields = {"code": code}
        if params.state is not None:
            fields["state"] = params.state
        return str(params.redirect_uri) + "?" + urlencode(fields)

    async def load_authorization_code(self, client, authorization_code):
        data = self.records.get("code", authorization_code)
        if data and data["client_id"] == client.client_id:
            return AuthorizationCode(code=authorization_code, **data)
        return None

    async def exchange_authorization_code(self, client, authorization_code):
        return self.issue(
            client.client_id, authorization_code.subject, authorization_code.scopes,
            consume=("code", authorization_code.code),
        )

    async def load_refresh_token(self, client, refresh_token):
        data = self.records.get("refresh", refresh_token)
        if data and data["client_id"] == client.client_id:
            return RefreshToken(token=refresh_token, **data)
        return None

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        return self.issue(
            client.client_id, refresh_token.subject, scopes, consume=("refresh", refresh_token.token),
        )

    def issue(self, client_id, subject, scopes, *, consume=None):
        now = int(time.time())
        access_token, refresh_token = new_secret(), new_secret()
        fields = dict(client_id=client_id, subject=subject, scopes=scopes, resource=self.resource)
        records = [
            ("access", access_token, {**fields, "expires_at": now + ACCESS_SECONDS}, ACCESS_SECONDS),
            ("refresh", refresh_token, {**fields, "expires_at": now + REFRESH_SECONDS}, REFRESH_SECONDS),
        ]
        if not self.records.change(write=records, consume=consume):
            raise TokenError("invalid_grant", "Grant was used or expired.")
        return OAuthToken(
            access_token=access_token, token_type="Bearer", expires_in=ACCESS_SECONDS,
            scope=" ".join(scopes), refresh_token=refresh_token,
        )

    async def load_access_token(self, token):
        data = self.records.get("access", token)
        if data and data["resource"] == self.resource:
            return AccessToken(token=token, **data)
        return None

    async def verify_token(self, token):
        return await self.load_access_token(token)

    async def revoke_token(self, token):
        self.records.take("access" if isinstance(token, AccessToken) else "refresh", token.token)

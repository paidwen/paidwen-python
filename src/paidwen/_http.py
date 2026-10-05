from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from ._contract import Tool

MAX_BODY_BYTES = 64 * 1024
REDACTED = "REDACTED"

SENSITIVE = frozenset(
    key.lower()
    for key in (
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        "auth",
        "credentials",
        "mysql_pwd",
        "privatekey",
        "private_key",
        "token",
        "session",
        "csrftoken",
        "sessionid",
        "x_csrftoken",
        "x_forwarded_for",
        "set_cookie",
        "cookie",
        "authorization",
        "proxy-authorization",
        "x_api_key",
        "aiohttp_session",
        "connect.sid",
        "csrf_token",
        "csrf",
        "_csrf",
        "_csrf_token",
        "PHPSESSID",
        "_session",
        "symfony",
        "user_session",
        "_xsrf",
        "XSRF-TOKEN",
        "client_secret",
        "access_token",
        "refresh_token",
        "id_token",
        "X-Amz-Signature",
        "X-Amz-Credential",
        "X-Amz-Security-Token",
        "sig",
        "X-Goog-Signature",
    )
)


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: REDACTED if isinstance(key, str) and key.lower() in SENSITIVE else redact(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [redact(inner) for inner in value]
    return value


def entries(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    for key, value in pairs:
        previous = collected.get(key)
        if previous is None:
            collected[key] = value
        elif isinstance(previous, list):
            previous.append(value)
        else:
            collected[key] = [previous, value]
    return collected


def body_of(content: bytes | str | None, content_type: str | None) -> Any:
    if content is None or len(content) == 0 or len(content) > MAX_BODY_BYTES:
        return None
    kind = (content_type or "").split(";")[0].strip().lower()
    text = content.decode("utf-8", "replace") if isinstance(content, bytes) else content
    if kind == "application/x-www-form-urlencoded":
        return entries(parse_qsl(text, keep_blank_values=True))
    if kind == "application/json" or kind.endswith("+json"):
        try:
            return json.loads(text)
        except ValueError:
            return None
    return None


def describe(method: str, url: str, body: Any = None) -> tuple[Tool, dict[str, Any]]:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None and not (
        (parts.scheme == "http" and parts.port == 80)
        or (parts.scheme == "https" and parts.port == 443)
    ):
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    method = method.upper()
    params: dict[str, Any] = {
        "method": method,
        "url": f"{parts.scheme}://{host}{path}",
        "query": redact(entries(parse_qsl(parts.query, keep_blank_values=True))),
    }
    if body is not None:
        params["body"] = redact(body)
    tool: Tool = {"name": f"{method} {path}", "kind": "http", "server": host}
    return tool, params


def matches(pattern: str, host: str) -> bool:
    parts = pattern.lower().split("*")
    value = host.lower()
    if len(parts) == 1:
        return value == parts[0]
    if not value.startswith(parts[0]):
        return False
    position = len(parts[0])
    for part in parts[1:-1]:
        found = value.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return len(value) - len(parts[-1]) >= position and value.endswith(parts[-1])


def ignored(patterns: Iterable[str], host: str) -> bool:
    return any(matches(pattern, host) for pattern in patterns)

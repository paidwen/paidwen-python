from __future__ import annotations

import importlib
import json
import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ._context import Carrier, missing
from ._errors import PaidwenError
from ._http import MAX_BODY_BYTES, body_of, describe, entries, server_of

if TYPE_CHECKING:
    from ._client import Paidwen

_lock = threading.Lock()
_installed = False


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(name)
    if value is None and isinstance(headers, Mapping):
        for key, candidate in headers.items():
            if isinstance(key, str) and key.lower() == name.lower():
                return str(candidate)
    return None if value is None else str(value)


def _pairs(headers: Any) -> list[tuple[Any, Any]]:
    if headers is None:
        return []
    return list(headers.items() if callable(getattr(headers, "items", None)) else headers)


def _names(*headers: Any) -> set[str]:
    return {str(name).lower() for given in headers for name, _ in _pairs(given)}


def _httpx(paidwen: Paidwen, ignore: tuple[str, ...]) -> Callable[[], None] | None:
    try:
        httpx = importlib.import_module("httpx")
    except ImportError:
        return None
    sync_transport = httpx.HTTPTransport
    async_transport = httpx.AsyncHTTPTransport
    original_sync = sync_transport.handle_request
    original_async = async_transport.handle_async_request

    def content(request: Any) -> bytes | None:
        try:
            data = request.content
        except httpx.RequestNotRead:
            return None
        return data if isinstance(data, bytes) else None

    def call(request: Any) -> tuple[Any, Any] | None:
        url = str(request.url)
        if paidwen.skips(server_of(url), ignore):
            return None
        return describe(
            request.method, url, body_of(content(request), request.headers.get("content-type"))
        )

    def propagated(request: Any, carrier: Carrier | None) -> Any:
        if carrier is None:
            return request
        added = missing(carrier, lambda name: name in request.headers)
        if not added:
            return request
        headers = request.headers.copy()
        headers.update(added)
        return httpx.Request(
            request.method,
            request.url,
            headers=headers,
            stream=request.stream,
            extensions=request.extensions,
        )

    def passed(request: Any) -> Any:
        return propagated(request, paidwen.passing(server_of(str(request.url)), ignore))

    def handle_request(transport: Any, request: Any) -> Any:
        described = call(request)
        if described is None:
            return original_sync(transport, passed(request))
        return paidwen._run(
            described[0],
            described[1],
            lambda carrier: original_sync(transport, propagated(request, carrier)),
            lambda response: response.status_code,
        )

    async def handle_async_request(transport: Any, request: Any) -> Any:
        described = call(request)
        if described is None:
            return await original_async(transport, passed(request))
        return await paidwen._run_async(
            described[0],
            described[1],
            lambda carrier: original_async(transport, propagated(request, carrier)),
            lambda response: response.status_code,
        )

    sync_transport.handle_request = handle_request
    async_transport.handle_async_request = handle_async_request

    def release() -> None:
        sync_transport.handle_request = original_sync
        async_transport.handle_async_request = original_async

    return release


def _urllib3(paidwen: Paidwen, ignore: tuple[str, ...]) -> Callable[[], None] | None:
    try:
        pools = importlib.import_module("urllib3.connectionpool")
    except ImportError:
        return None
    pool_class = pools.HTTPConnectionPool
    original = pool_class.urlopen

    def urlopen(
        pool: Any,
        method: str,
        url: str,
        body: Any = None,
        headers: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if url.startswith(("http://", "https://")):
            full = url
        else:
            port = pool.port
            default = (pool.scheme == "http" and port in (None, 80)) or (
                pool.scheme == "https" and port in (None, 443)
            )
            host = pool.host if default else f"{pool.host}:{port}"
            full = f"{pool.scheme}://{host}{url}"
        server = server_of(full)

        def sent(carrier: Carrier | None) -> Any:
            if carrier is None:
                return headers
            given = pool.headers if headers is None else headers
            names = _names(given)
            added = missing(carrier, lambda name: name in names)
            return {**given, **added} if added else headers

        if paidwen.skips(server, ignore):
            passing = paidwen.passing(server, ignore)
            return original(pool, method, url, body, sent(passing), *args, **kwargs)
        data = body if isinstance(body, (bytes, str)) else None
        tool, params = describe(method, full, body_of(data, _header(headers, "Content-Type")))
        return paidwen._run(
            tool,
            params,
            lambda carrier: original(pool, method, url, body, sent(carrier), *args, **kwargs),
            lambda response: response.status,
        )

    pool_class.urlopen = urlopen

    def release() -> None:
        pool_class.urlopen = original

    return release


def _aiohttp(paidwen: Paidwen, ignore: tuple[str, ...]) -> Callable[[], None] | None:
    try:
        aiohttp = importlib.import_module("aiohttp")
        yarl = importlib.import_module("yarl")
    except ImportError:
        return None
    session_class = aiohttp.ClientSession
    original = session_class._request

    def body(kwargs: dict[str, Any]) -> Any:
        if kwargs.get("json") is not None:
            encoded = json.dumps(kwargs["json"], default=str)
            return kwargs["json"] if len(encoded) <= MAX_BODY_BYTES else None
        data = kwargs.get("data")
        if isinstance(data, Mapping):
            return entries((str(key), str(value)) for key, value in data.items())
        if isinstance(data, (bytes, str)):
            return body_of(data, _header(kwargs.get("headers"), "Content-Type"))
        return None

    async def request(session: Any, method: str, str_or_url: Any, *args: Any, **kwargs: Any) -> Any:
        url = yarl.URL(str(str_or_url))
        if kwargs.get("params"):
            url = url.update_query(kwargs["params"])
        full = str(url)
        server = server_of(full)

        def sent(carrier: Carrier | None) -> dict[str, Any]:
            if carrier is None:
                return kwargs
            given = kwargs.get("headers")
            names = _names(given, getattr(session, "headers", None))
            added = missing(carrier, lambda name: name in names)
            if not added:
                return kwargs
            return {**kwargs, "headers": [*_pairs(given), *added.items()]}

        if paidwen.skips(server, ignore):
            passing = paidwen.passing(server, ignore)
            return await original(session, method, str_or_url, *args, **sent(passing))
        tool, params = describe(method, full, body(kwargs))
        return await paidwen._run_async(
            tool,
            params,
            lambda carrier: original(session, method, str_or_url, *args, **sent(carrier)),
            lambda response: response.status,
        )

    session_class._request = request

    def release() -> None:
        session_class._request = original

    return release


def install(paidwen: Paidwen, ignore: tuple[str, ...]) -> Callable[[], None]:
    global _installed
    with _lock:
        if _installed:
            raise PaidwenError("The HTTP calls of this process are already captured by Paidwen.")
        releases = [
            release
            for release in (
                _httpx(paidwen, ignore),
                _urllib3(paidwen, ignore),
                _aiohttp(paidwen, ignore),
            )
            if release is not None
        ]
        _installed = True

    def release_all() -> None:
        global _installed
        with _lock:
            for release in reversed(releases):
                release()
            _installed = False

    return release_all

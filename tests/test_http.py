from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import anyio
import httpx
import pytest
import requests
import urllib3
from fake import (
    AGENT_ID,
    CALLER_ID,
    CLIENT_TRACEPARENT,
    DISABLED,
    FakePaidwen,
    decision,
    parent,
    serve,
)

from paidwen import DeniedError, Paidwen, PaidwenError, RequestError


@dataclass
class Api:
    url: str = ""
    received: list[tuple[str, str, bytes]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)

    @property
    def host(self) -> str:
        return urlsplit(self.url).netloc


@pytest.fixture
def api() -> Iterator[Api]:
    state = Api()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _answer(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            state.received.append((self.command, self.path, body))
            state.headers.append({key.lower(): value for key, value in self.headers.items()})
            data = b'{"refund":"re_1"}'
            self.send_response(404 if self.path.startswith("/missing") else 200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = _answer
        do_POST = _answer
        do_DELETE = _answer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def fake() -> Iterator[FakePaidwen]:
    with serve() as server:
        yield server


@pytest.fixture
def capture(fake: FakePaidwen) -> Iterator[Callable[..., Paidwen]]:
    opened: list[Paidwen] = []

    def open_capture(ignore: tuple[str, ...] = (), journal: bool = False) -> Paidwen:
        paidwen = Paidwen("pw_test_key", base_url=fake.url, journal=journal)
        paidwen.http(ignore=ignore)
        opened.append(paidwen)
        return paidwen

    yield open_capture
    for paidwen in opened:
        paidwen.close()


def asked(fake: FakePaidwen) -> list[dict[str, Any]]:
    return [recorded.body for recorded in fake.decisions]


def propagated(api: Api) -> list[tuple[str, str | None, str | None]]:
    return [
        (path, parent(headers.get("traceparent")), headers.get("baggage"))
        for (_, path, _), headers in zip(api.received, api.headers, strict=True)
    ]


def test_httpx_asks_before_the_call_and_reads_it_as_a_tool(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    capture()
    response = httpx.post(
        f"{api.url}/v1/refunds?expand=charge&sig=abc&tag=a&tag=b",
        headers={"authorization": "Bearer sk_live_secret"},
        json={
            "amount": 2500,
            "charge": "ch_1",
            "api_key": "sk_live_secret",
            "nested": {"client_secret": "x"},
        },
    )
    assert response.status_code == 200
    assert response.json() == {"refund": "re_1"}
    assert len(api.received) == 1
    [question] = asked(fake)
    assert question["tool"] == {"name": "POST /v1/refunds", "kind": "http", "server": api.host}
    assert question["params"] == {
        "method": "POST",
        "url": f"{api.url}/v1/refunds",
        "query": {"expand": "charge", "sig": "REDACTED", "tag": ["a", "b"]},
        "body": {
            "amount": 2500,
            "charge": "ch_1",
            "api_key": "REDACTED",
            "nested": {"client_secret": "REDACTED"},
        },
    }
    assert "sk_live_secret" not in json.dumps(question)


def test_a_captured_call_carries_the_trace_and_the_baggage_of_its_tool(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    fake.answer = lambda request: decision(request, "allow", agent_id=AGENT_ID)
    capture()
    httpx.get(f"{api.url}/v1/httpx")
    requests.get(f"{api.url}/v1/requests")
    urllib3.request("GET", f"{api.url}/v1/urllib3")

    async def call() -> None:
        async with httpx.AsyncClient() as client:
            await client.get(f"{api.url}/v1/async")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api.url}/v1/aiohttp") as response:
                await response.read()
            async with session.get(
                f"{api.url}/v1/kept", headers=[("Baggage", "tenant=acme")]
            ) as response:
                await response.read()

    anyio.run(call)
    httpx.get(
        f"{api.url}/v1/kept",
        headers={"Traceparent": CLIENT_TRACEPARENT, "baggage": "tenant=acme"},
    )
    requests.get(f"{api.url}/v1/kept", headers={"BAGGAGE": "tenant=acme"})
    urllib3.request("GET", f"{api.url}/v1/kept", headers={"traceparent": CLIENT_TRACEPARENT})
    decided = [parent(recorded.headers["traceparent"]) for recorded in fake.decisions]
    client = parent(CLIENT_TRACEPARENT)
    own = f"paidwen.agent.id={AGENT_ID}"
    assert propagated(api) == [
        ("/v1/httpx", decided[0], own),
        ("/v1/requests", decided[1], own),
        ("/v1/urllib3", decided[2], own),
        ("/v1/async", decided[3], own),
        ("/v1/aiohttp", decided[4], own),
        ("/v1/kept", decided[5], "tenant=acme"),
        ("/v1/kept", client, "tenant=acme"),
        ("/v1/kept", decided[7], "tenant=acme"),
        ("/v1/kept", client, own),
    ]
    assert len(set(decided)) == 9


def test_a_call_inside_a_tool_carries_the_trace_and_the_baggage_of_the_tool(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    fake.answer = lambda request: decision(request, "allow", agent_id=AGENT_ID)
    paidwen = capture()

    @paidwen.tool("billing.ask")
    def ask() -> None:
        httpx.get(f"{api.url}/v1/httpx")
        requests.get(f"{api.url}/v1/requests")
        urllib3.request("GET", f"{api.url}/v1/urllib3")
        httpx.get(f"{api.url}/v1/kept", headers={"traceparent": CLIENT_TRACEPARENT})

    @paidwen.tool("billing.ask.async")
    async def ask_async() -> None:
        async with httpx.AsyncClient() as client:
            await client.get(f"{api.url}/v1/async")
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{api.url}/v1/aiohttp") as response,
        ):
            await response.read()

    ask()
    anyio.run(ask_async)
    tools = [parent(recorded.headers["traceparent"]) for recorded in fake.decisions]
    own = f"paidwen.agent.id={AGENT_ID}"
    assert len(tools) == 2
    assert propagated(api) == [
        ("/v1/httpx", tools[0], own),
        ("/v1/requests", tools[0], own),
        ("/v1/urllib3", tools[0], own),
        ("/v1/kept", parent(CLIENT_TRACEPARENT), own),
        ("/v1/async", tools[1], own),
        ("/v1/aiohttp", tools[1], own),
    ]


def test_a_call_inside_a_tool_toward_an_ignored_host_carries_nothing(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    fake.answer = lambda request: decision(request, "allow", agent_id=AGENT_ID)
    paidwen = capture(ignore=("127.0.0.*",))

    @paidwen.tool("search")
    def search() -> None:
        httpx.get(f"{api.url}/v1/httpx")
        requests.get(f"{api.url}/v1/requests")

    search()
    assert len(fake.decisions) == 1
    assert propagated(api) == [("/v1/httpx", None, None), ("/v1/requests", None, None)]


def test_an_agent_called_from_a_tool_of_another_agent_sends_its_caller(fake: FakePaidwen) -> None:
    agents = {"Bearer key-a": CALLER_ID, "Bearer key-b": AGENT_ID}
    fake.answer = lambda request: decision(
        request, "allow", agent_id=agents[fake.decisions[-1].headers["authorization"]]
    )
    billing = Paidwen("key-b", base_url=fake.url, journal=False)

    @billing.tool("refunds.create")
    def refund() -> str:
        return "re_1"

    class Service(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            with billing.context(headers=self.headers):
                data = refund().encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    service = ThreadingHTTPServer(("127.0.0.1", 0), Service)
    threading.Thread(target=service.serve_forever, daemon=True).start()
    triage = Paidwen("key-a", base_url=fake.url, journal=False)
    triage.http()

    @triage.tool("billing.ask")
    def ask() -> str:
        return httpx.get(f"http://127.0.0.1:{service.server_address[1]}/ask").text

    try:
        assert ask() == "re_1"
    finally:
        triage.close()
        billing.close()
        service.shutdown()
        service.server_close()
    asked, refunded = fake.decisions
    assert asked.headers["authorization"] == "Bearer key-a"
    assert "caller_agent_id" not in asked.body["context"]
    assert refunded.headers["authorization"] == "Bearer key-b"
    assert refunded.body["context"]["caller_agent_id"] == CALLER_ID
    assert refunded.body["context"]["trace_id"] == asked.body["context"]["trace_id"]


def test_a_refused_httpx_call_is_never_sent(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    fake.answer = lambda request: decision(request, "block", rule_id="no-refunds")
    capture()
    with pytest.raises(DeniedError, match="no-refunds"):
        httpx.post(f"{api.url}/v1/refunds")
    assert api.received == []


def test_a_refused_agent_never_sends_the_call_even_with_allow(api: Api) -> None:
    paidwen = Paidwen(
        "pw_test_key",
        base_url="https://api.example.test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(403, json=DISABLED))
        ),
        journal=False,
        on_unavailable="allow",
    )
    paidwen.http()
    try:
        with pytest.raises(RequestError, match=r"^Paidwen refused the agent"):
            httpx.post(f"{api.url}/v1/refunds", json={"amount": 2500})
        with pytest.raises(RequestError, match=r"^Paidwen refused the agent"):
            requests.post(f"{api.url}/v1/refunds", json={"amount": 2500})
    finally:
        paidwen.close()
    assert api.received == []


def test_an_async_httpx_call_asks_too(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    capture()

    async def call() -> int:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api.url}/v1/customers/42")
            return response.status_code

    assert anyio.run(call) == 200
    assert [question["tool"]["name"] for question in asked(fake)] == ["GET /v1/customers/42"]


def test_requests_and_urllib3_ask_and_a_refusal_raises_the_error_of_the_sdk(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    fake.answer = lambda request: decision(
        request, "block" if request["tool"]["name"].startswith("DELETE") else "allow"
    )
    capture()
    response = requests.post(
        f"{api.url}/v1/refunds", data={"amount": "2500", "password": "hunter2"}
    )
    assert response.status_code == 200
    with pytest.raises(DeniedError):
        requests.delete(f"{api.url}/v1/customers/42")
    with pytest.raises(DeniedError):
        urllib3.request("DELETE", f"{api.url}/v1/customers/43")
    assert urllib3.request("GET", f"{api.url}/v1/customers/44").status == 200
    form, first, second, read = asked(fake)
    assert form["params"]["body"] == {"amount": "2500", "password": "REDACTED"}
    assert form["tool"]["server"] == api.host
    assert first["tool"]["name"] == "DELETE /v1/customers/42"
    assert second["tool"]["name"] == "DELETE /v1/customers/43"
    assert read["params"]["url"] == f"{api.url}/v1/customers/44"
    assert [call[0] for call in api.received] == ["POST", "GET"]


def test_aiohttp_asks_before_the_call(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    capture()

    async def call() -> dict[str, Any]:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{api.url}/v1/refunds",
                params={"expand": "charge"},
                json={"amount": 2500, "token": "tok_1"},
            ) as response,
        ):
            return await response.json()

    assert anyio.run(call) == {"refund": "re_1"}
    [question] = asked(fake)
    assert question["tool"] == {"name": "POST /v1/refunds", "kind": "http", "server": api.host}
    assert question["params"]["query"] == {"expand": "charge"}
    assert question["params"]["body"] == {"amount": 2500, "token": "REDACTED"}


def test_an_unknown_or_large_body_stays_unread(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    capture()
    httpx.post(f"{api.url}/v1/files", content=b"binary", headers={"content-type": "image/png"})
    httpx.post(f"{api.url}/v1/large", json={"blob": "a" * 70_000})
    binary, large = asked(fake)
    assert "body" not in binary["params"]
    assert "body" not in large["params"]


def test_nothing_is_asked_inside_an_action_for_ignored_hosts_and_after_the_release(
    api: Api, fake: FakePaidwen
) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url, journal=False)
    release = paidwen.http(ignore=["*.example.test", "localhost"])

    @paidwen.tool("search")
    def search() -> int:
        return httpx.get(f"{api.url}/v1/inside").status_code

    assert search() == 200
    assert len(fake.decisions) == 1, "the nested call inside the action asks nothing"
    release()
    httpx.get(f"{api.url}/v1/after")
    assert len(fake.decisions) == 1, "the released capture asks nothing"
    assert [call[1] for call in api.received] == ["/v1/inside", "/v1/after"]
    second = Paidwen("pw_test_key", base_url=fake.url, journal=False)
    second.http(ignore=["127.0.0.*"])
    httpx.get(f"{api.url}/v1/ignored")
    assert len(fake.decisions) == 1, "the ignored host asks nothing"
    second.close()
    paidwen.close()


def test_each_captured_call_reaches_the_journal_with_its_status(
    api: Api, fake: FakePaidwen, capture: Callable[..., Paidwen]
) -> None:
    paidwen = capture(journal=True)
    httpx.get(f"{api.url}/missing/thing")
    requests.get(f"{api.url}/missing/other")
    paidwen.flush()
    spans = fake.spans()
    assert len(spans) == 2, "the export of the journal is not captured"
    assert [span.name for span in spans] == [
        "execute_tool GET /missing/thing",
        "execute_tool GET /missing/other",
    ]
    assert spans[0].attributes["paidwen.tool.kind"] == "http"
    assert spans[0].attributes["paidwen.tool.server"] == api.host
    assert spans[0].attributes["http.response.status_code"] == 404
    assert spans[0].status == 2
    assert len(fake.decisions) == 2


def test_a_second_capture_is_refused(capture: Callable[..., Paidwen]) -> None:
    paidwen = capture()
    with pytest.raises(PaidwenError, match="already captured"):
        paidwen.http()
    other = Paidwen("pw_test_key", base_url="http://127.0.0.1:9", journal=False)
    with pytest.raises(PaidwenError, match="already captured"):
        other.http()
    other.close()

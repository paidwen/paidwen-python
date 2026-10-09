from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

APPROVAL_ID = "0d9e8f7a-6b5c-4d3e-8f2a-1b0c9d8e7f6a"
AGENT_ID = "4b0f3a52-8c1d-4e7a-9f26-1d5c7b9e3a40"
CALLER_ID = "9e2d4c61-7a3b-4f58-8e10-6c2a9b4d7f13"
CLIENT_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
DISABLED = {
    "type": "https://paidwen.com/problems/agent-disabled",
    "title": "The agent is disabled.",
    "status": 403,
    "detail": "The agent is disabled until someone enables it.",
}


@dataclass
class Recorded:
    path: str
    headers: dict[str, str]
    body: Any
    at: float


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    attributes: dict[str, Any]
    events: list[str]
    status: int


def decision(request: dict[str, Any], verdict: str, **extra: Any) -> dict[str, Any]:
    return {
        "action_id": request["action_id"],
        "verdict": verdict,
        "rule_id": None,
        "observed": [],
        "decided_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def waiting(request: dict[str, Any], on_expiry: str = "deny") -> dict[str, Any]:
    return decision(
        request,
        "require_approval",
        rule_id="refund-above-500",
        approval={"id": APPROVAL_ID, "expires_in": 300, "interval": 1, "on_expiry": on_expiry},
    )


def status(state: str, on_expiry: str = "deny") -> tuple[int, dict[str, Any]]:
    return 200, {
        "id": APPROVAL_ID,
        "action_id": "6f1c2a7e-3b5d-4c8e-9a10-2f4b6d8e0a12",
        "status": state,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        "interval": 1,
        "on_expiry": on_expiry,
    }


def _value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    return getattr(value, kind) if kind else None


@dataclass
class FakePaidwen:
    decisions: list[Recorded] = field(default_factory=list)
    polls: list[Recorded] = field(default_factory=list)
    traces: list[Recorded] = field(default_factory=list)
    exchanges: list[dict[str, list[str]]] = field(default_factory=list)
    script: list[tuple[int, Any]] = field(default_factory=list)
    answer: Callable[[dict[str, Any]], dict[str, Any]] = lambda request: decision(request, "allow")
    url: str = ""

    def spans(self) -> list[Span]:
        spans: list[Span] = []
        for recorded in self.traces:
            request = ExportTraceServiceRequest()
            request.ParseFromString(recorded.body)
            for resource in request.resource_spans:
                for scope in resource.scope_spans:
                    for span in scope.spans:
                        spans.append(
                            Span(
                                name=span.name,
                                trace_id=span.trace_id.hex(),
                                span_id=span.span_id.hex(),
                                attributes={a.key: _value(a.value) for a in span.attributes},
                                events=[event.name for event in span.events],
                                status=span.status.code,
                            )
                        )
        return spans


@contextmanager
def serve() -> Iterator[FakePaidwen]:
    fake = FakePaidwen()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _record(self) -> Recorded:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            body: Any = raw
            if self.headers.get("content-type", "").startswith("application/json") and raw:
                body = json.loads(raw)
            return Recorded(
                self.path, {k.lower(): v for k, v in self.headers.items()}, body, time.time()
            )

        def _reply(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            recorded = self._record()
            if self.path == "/v1/decisions":
                fake.decisions.append(recorded)
                self._reply(200, fake.answer(recorded.body))
            elif self.path == "/v1/oauth/token":
                fake.exchanges.append(parse_qs(recorded.body.decode()))
                self._reply(
                    200,
                    {
                        "access_token": f"agent-token-{len(fake.exchanges)}",
                        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                        "token_type": "Bearer",
                        "expires_in": 900,
                    },
                )
            elif self.path == "/v1/traces":
                fake.traces.append(recorded)
                self.send_response(200)
                self.send_header("content-type", "application/x-protobuf")
                self.send_header("content-length", "0")
                self.end_headers()
            else:
                self._reply(404, {})

        def do_GET(self) -> None:
            recorded = self._record()
            if self.path.startswith("/v1/approvals/"):
                fake.polls.append(recorded)
                code, body = fake.script.pop(0) if fake.script else (503, {"title": "down"})
                self._reply(code, body or {"type": "x", "title": "x", "status": code})
            else:
                self._reply(404, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()


def parent(traceparent: str | None) -> str | None:
    return None if traceparent is None else "-".join(traceparent.split("-")[1:3])


@contextmanager
def carrying(entries: dict[str, str]) -> Iterator[None]:
    otel = otel_context.get_current()
    for key, value in entries.items():
        otel = baggage.set_baggage(key, value, otel)
    token = otel_context.attach(otel)
    try:
        yield
    finally:
        otel_context.detach(token)


def until(what: str, done: Callable[[], bool], limit: float = 10.0) -> None:
    started = time.time()
    while time.time() - started < limit:
        if done():
            return
        time.sleep(0.02)
    raise AssertionError(f"{what} did not happen within {limit} seconds")

import asyncio
import json
import logging
import re
import socket
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from paidwen import (
    Decision,
    DeniedError,
    Paidwen,
    PaidwenError,
    PendingError,
    RequestError,
)

ACTION_ID = "6f1c2a7e-3b5d-4c8e-9a10-2f4b6d8e0a12"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

Handler = Callable[[httpx.Request], httpx.Response]


def decision(verdict: str, rule_id: str | None = None) -> Decision:
    return {
        "action_id": ACTION_ID,
        "verdict": verdict,
        "rule_id": rule_id,
        "observed": [],
        "decided_at": "2026-10-05T09:00:00Z",
    }


def server(status: int, body: Any, requests: list[httpx.Request] | None = None) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return handle


def failing(error: type[httpx.HTTPError]) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        raise error("no answer", request=request)

    return handle


def client(handler: Handler, **options: Any) -> Paidwen:
    transport = httpx.MockTransport(handler)
    return Paidwen(
        "pw_test_key",
        base_url="https://api.example.test/",
        http_client=httpx.Client(transport=transport),
        journal=False,
        **options,
    )


def test_decide_asks_the_decision_endpoint_with_the_agent_key() -> None:
    requests: list[httpx.Request] = []
    answer = decision("allow")
    result = client(server(200, answer, requests)).decide(
        tool={"name": "payments.refund", "kind": "function"},
        params={"amount": 120},
        cost={"estimated_usd": 0.01},
    )
    assert result == answer
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.example.test/v1/decisions"
    assert request.headers["authorization"] == "Bearer pw_test_key"
    assert request.headers["content-type"] == "application/json"
    sent = json.loads(request.content)
    assert UUID.match(sent["action_id"])
    assert sent["tool"] == {"name": "payments.refund", "kind": "function"}
    assert sent["params"] == {"amount": 120}
    assert sent["cost"] == {"estimated_usd": 0.01}


def test_decide_keeps_the_action_identifier_given_by_the_agent() -> None:
    requests: list[httpx.Request] = []
    client(server(200, decision("allow"), requests)).decide(
        action_id=ACTION_ID, tool={"name": "x", "kind": "http"}
    )
    assert json.loads(requests[0].content)["action_id"] == ACTION_ID


def test_decide_blocks_by_default_when_paidwen_does_not_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="paidwen"):
        result = client(failing(httpx.ReadTimeout)).decide(
            action_id=ACTION_ID, tool={"name": "x", "kind": "function"}
        )
    assert result["verdict"] == "block"
    assert result["rule_id"] is None
    assert result["observed"] == []
    assert result["action_id"] == ACTION_ID
    assert 'applies on_unavailable "block"' in caplog.text


def test_decide_allows_when_the_agent_chose_so() -> None:
    result = client(failing(httpx.ConnectTimeout), on_unavailable="allow").decide(
        tool={"name": "x", "kind": "function"}
    )
    assert result["verdict"] == "allow"


@pytest.mark.parametrize(
    "handler",
    [
        failing(httpx.ConnectError),
        server(503, {"title": "down"}),
        server(200, "<html>"),
    ],
)
def test_decide_applies_the_fallback_on_any_failure(handler: Handler) -> None:
    assert client(handler).decide(tool={"name": "x", "kind": "function"})["verdict"] == "block"


def test_decide_gives_up_on_a_silent_server() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        started = time.monotonic()
        with Paidwen(
            "pw_test_key", base_url=f"http://127.0.0.1:{port}", timeout=0.2, journal=False
        ) as paidwen:
            result = paidwen.decide(tool={"name": "x", "kind": "function"})
        elapsed = time.monotonic() - started
    assert result["verdict"] == "block"
    assert 0.2 <= elapsed < 2


def test_decide_raises_a_request_error_when_paidwen_refuses_the_request() -> None:
    problem = {
        "type": "https://paidwen.com/problems/unauthorized",
        "title": "The agent key is missing, malformed or unknown.",
        "status": 401,
        "detail": "The agent key is unknown.",
    }
    with pytest.raises(RequestError) as raised:
        client(server(401, problem)).decide(tool={"name": "x", "kind": "function"})
    assert raised.value.status == 401
    assert str(raised.value) == "The agent key is unknown."
    assert raised.value.problem == problem


def test_tool_runs_the_function_when_paidwen_allows_it() -> None:
    requests: list[httpx.Request] = []
    paidwen = client(server(200, decision("allow"), requests))

    @paidwen.tool("payments.refund")
    def refund(order: str, amount: float, reason: str = "lost parcel", **extra: Any) -> str:
        return f"refunded {amount}"

    assert refund("4821", amount=120, channel="email") == "refunded 120"
    assert refund.__name__ == "refund"
    sent = json.loads(requests[0].content)
    assert sent["tool"] == {"name": "payments.refund", "kind": "function"}
    assert sent["params"] == {
        "order": "4821",
        "amount": 120,
        "reason": "lost parcel",
        "channel": "email",
    }


@pytest.mark.parametrize("verdict", ["block", "quarantine"])
def test_tool_never_runs_the_function_when_paidwen_denies_it(verdict: str) -> None:
    calls: list[str] = []
    paidwen = client(server(200, decision(verdict, "rule-1")))

    @paidwen.tool("x")
    def run() -> None:
        calls.append("run")

    with pytest.raises(DeniedError) as raised:
        run()
    assert isinstance(raised.value, PaidwenError)
    assert raised.value.decision["verdict"] == verdict
    assert str(raised.value) == f"Paidwen answered {verdict} for action {ACTION_ID} by rule rule-1."
    assert calls == []


def test_tool_says_when_the_permissions_of_the_agent_refuse_the_action() -> None:
    refused = {**decision("block"), "permission": {"allowed": False}}
    paidwen = client(server(200, refused))

    @paidwen.tool("payments.transfer")
    def transfer() -> None:
        raise AssertionError("the function must not run")

    with pytest.raises(DeniedError) as raised:
        transfer()
    assert str(raised.value) == (
        f"Paidwen answered block for action {ACTION_ID} "
        "because the permissions of the agent do not allow it."
    )


@pytest.mark.parametrize("verdict", ["require_approval", "escalate"])
def test_tool_waits_when_paidwen_asks_for_a_human_answer(verdict: str) -> None:
    paidwen = client(server(200, decision(verdict, "rule-1")))

    @paidwen.tool("x")
    def run() -> None:
        raise AssertionError("the function must not run")

    with pytest.raises(PendingError) as raised:
        run()
    assert raised.value.decision["verdict"] == verdict


def test_tool_wraps_a_coroutine_function() -> None:
    requests: list[httpx.Request] = []
    paidwen = client(server(200, decision("allow"), requests))

    @paidwen.tool("reports.generate")
    async def generate(pages: int) -> str:
        await asyncio.sleep(0)
        return f"{pages} pages"

    assert asyncio.run(generate(pages=4)) == "4 pages"
    assert json.loads(requests[0].content)["params"] == {"pages": 4}

    denied = client(server(200, decision("block", "rule-1")))

    @denied.tool("reports.generate")
    async def refused() -> None:
        raise AssertionError("the function must not run")

    with pytest.raises(DeniedError):
        asyncio.run(refused())


def test_the_constructor_needs_an_agent_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAIDWEN_API_KEY", raising=False)
    with pytest.raises(PaidwenError):
        Paidwen()


def test_the_constructor_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAIDWEN_API_KEY", "pw_test_from_env")
    monkeypatch.delenv("PAIDWEN_BASE_URL", raising=False)
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(server(200, decision("allow"), requests))
    with Paidwen(http_client=httpx.Client(transport=transport), journal=False) as paidwen:
        paidwen.decide(tool={"name": "x", "kind": "function"})
    assert str(requests[0].url) == "https://api.paidwen.com/v1/decisions"
    assert requests[0].headers["authorization"] == "Bearer pw_test_from_env"


def test_the_constructor_checks_its_options() -> None:
    with pytest.raises(PaidwenError):
        Paidwen("k", on_unavailable="maybe")
    with pytest.raises(PaidwenError):
        Paidwen("k", timeout=0)

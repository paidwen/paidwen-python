import json
from collections.abc import Callable
from typing import Any

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.client import Client
from mcp.server.mcpserver import MCPServer

from paidwen import DeniedError, Paidwen, PaidwenError, PendingError


def payments(served: list[str]) -> MCPServer:
    server = MCPServer("payments")

    @server.tool()
    def refund(order: str = "", amount: float = 0) -> str:
        served.append(order)
        return f"refunded {amount:g}"

    return server


def answering(verdict: str, asked: list[dict[str, Any]]) -> Paidwen:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append({"tool": body["tool"], "params": body.get("params")})
        return httpx.Response(
            200,
            json={
                "action_id": body["action_id"],
                "verdict": verdict,
                "rule_id": None if verdict == "allow" else "rule-1",
                "observed": [],
                "decided_at": "2026-10-05T09:00:00Z",
            },
        )

    return Paidwen(
        "pw_test_key",
        base_url="https://api.example.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        journal=False,
    )


def run(work: Callable[[], Any]) -> Any:
    return anyio.run(work)


def test_mcp_asks_before_each_tool_call_of_every_session() -> None:
    asked: list[dict[str, Any]] = []
    served: list[str] = []
    release = answering("allow", asked).mcp(ClientSession)
    try:

        async def call() -> Any:
            async with Client(payments(served)) as client:
                await client.list_tools()
                return await client.call_tool("refund", {"order": "4821", "amount": 120})

        result = run(call)
    finally:
        release()
    assert result.content[0].text == "refunded 120"
    assert asked == [
        {
            "tool": {"name": "refund", "kind": "mcp", "server": "payments"},
            "params": {"order": "4821", "amount": 120},
        }
    ]
    assert served == ["4821"]


@pytest.mark.parametrize(
    ("verdict", "error"),
    [
        ("block", DeniedError),
        ("quarantine", DeniedError),
        ("require_approval", PendingError),
        ("escalate", PendingError),
    ],
)
def test_mcp_never_sends_a_refused_call(verdict: str, error: type[Exception]) -> None:
    served: list[str] = []
    release = answering(verdict, []).mcp(ClientSession)
    try:

        async def call() -> None:
            async with Client(payments(served)) as client:
                with pytest.raises(error):
                    await client.call_tool("refund", {"order": "4822", "amount": 2500})

        run(call)
    finally:
        release()
    assert served == []


def test_mcp_captures_one_session_without_touching_the_others() -> None:
    asked: list[dict[str, Any]] = []
    served: list[str] = []
    paidwen = answering("block", asked)

    async def call() -> None:
        async with Client(payments(served)) as captured, Client(payments(served)) as other:
            release = paidwen.mcp(captured.session)
            try:
                with pytest.raises(DeniedError):
                    await captured.call_tool("refund", {"order": "captured"})
                await other.call_tool("refund", {"order": "other"})
            finally:
                release()

    run(call)
    assert len(asked) == 1
    assert served == ["other"]


def test_mcp_removes_the_capture_and_refuses_a_second_one() -> None:
    asked: list[dict[str, Any]] = []
    paidwen = answering("allow", asked)
    original = ClientSession.send_request
    release = paidwen.mcp(ClientSession)
    with pytest.raises(PaidwenError):
        paidwen.mcp(ClientSession)
    release()
    assert ClientSession.send_request is original
    assert not hasattr(ClientSession, "_paidwen_captured")

    async def call() -> None:
        async with Client(payments([])) as client:
            await client.call_tool("refund", {"order": "free"})

    run(call)
    assert asked == []


@pytest.mark.parametrize("target", [object(), 42, type("Empty", (), {})])
def test_mcp_refuses_a_target_that_is_not_an_mcp_session(target: object) -> None:
    with pytest.raises(PaidwenError):
        answering("allow", []).mcp(target)

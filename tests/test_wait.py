from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import anyio
import pytest
from fake import APPROVAL_ID, FakePaidwen, serve, status, waiting

from paidwen import DeniedError, Paidwen, PendingError, Wait


@pytest.fixture
def fake() -> Iterator[FakePaidwen]:
    with serve() as server:
        server.answer = waiting
        yield server


def client(fake: FakePaidwen, **options: Any) -> Paidwen:
    return Paidwen("pw_test_key", base_url=fake.url, journal=False, **options)


def test_the_function_waits_and_runs_once_approved(fake: FakePaidwen) -> None:
    fake.script = [status("pending"), status("approved")]
    waits: list[Wait] = []
    runs: list[int] = []
    paidwen = client(fake, on_wait=waits.append)

    @paidwen.tool("payments.refund")
    def refund(amount: int) -> str:
        runs.append(amount)
        return "refunded"

    started = time.time()
    assert refund(2500) == "refunded"
    assert time.time() - started >= 1.0
    assert runs == [2500]
    assert len(waits) == 1
    assert waits[0].approval_id == APPROVAL_ID
    assert waits[0].interval == 1
    assert waits[0].on_expiry == "deny"
    assert len(fake.polls) == 2
    assert fake.polls[0].path == f"/v1/approvals/{APPROVAL_ID}"
    assert fake.polls[0].headers["authorization"] == "Bearer pw_test_key"


def test_a_denied_action_never_runs(fake: FakePaidwen) -> None:
    fake.script = [status("denied")]
    paidwen = client(fake)
    runs: list[int] = []

    @paidwen.tool("payments.refund")
    def refund(amount: int) -> None:
        runs.append(amount)

    with pytest.raises(DeniedError) as raised:
        refund(2500)
    assert raised.value.approval is not None
    assert raised.value.approval["status"] == "denied"
    assert "the approver denied it" in str(raised.value)
    assert runs == []


def test_an_expired_wait_applies_the_choice_of_the_customer(fake: FakePaidwen) -> None:
    paidwen = client(fake)

    @paidwen.tool("payments.refund")
    def refund() -> str:
        return "ran"

    fake.script = [status("expired", "allow")]
    assert refund() == "ran"
    fake.script = [status("expired", "deny")]
    with pytest.raises(DeniedError, match="expired without an answer"):
        refund()


def test_slow_down_adds_five_seconds(fake: FakePaidwen) -> None:
    fake.script = [(429, None), status("approved")]
    paidwen = client(fake)

    @paidwen.tool("payments.refund")
    def refund() -> None:
        return None

    refund()
    gap = fake.polls[1].at - fake.polls[0].at
    assert 6.0 <= gap < 8.0


def test_an_unavailable_poll_is_tried_again(fake: FakePaidwen) -> None:
    fake.script = [(503, None), status("approved")]
    paidwen = client(fake)

    @paidwen.tool("payments.refund")
    def refund() -> str:
        return "ran"

    assert refund() == "ran"
    assert len(fake.polls) == 2


def test_on_pending_raise_returns_the_hand(fake: FakePaidwen) -> None:
    paidwen = client(fake, on_pending="raise")

    @paidwen.tool("payments.refund")
    def refund() -> str:
        return "ran"

    with pytest.raises(PendingError) as raised:
        refund()
    assert raised.value.decision["approval"]["id"] == APPROVAL_ID
    assert len(fake.polls) == 1


def test_on_pending_raise_runs_at_once_when_the_answer_is_already_there(
    fake: FakePaidwen,
) -> None:
    fake.script = [status("approved")]
    paidwen = client(fake, on_pending="raise")

    @paidwen.tool("payments.refund")
    def refund() -> str:
        return "ran"

    started = time.time()
    assert refund() == "ran"
    assert time.time() - started < 1.0
    assert len(fake.polls) == 1


def test_the_same_action_resumes_with_its_identifier(fake: FakePaidwen) -> None:
    action_id = "6f1c2a7e-3b5d-4c8e-9a10-2f4b6d8e0a12"
    first = client(fake, on_pending="raise")

    @first.tool("payments.refund")
    def refund(amount: int) -> str:
        return "ran"

    with first.context(action_id=action_id), pytest.raises(PendingError):
        refund(2500)
    fake.script = [status("approved")]
    second = client(fake)

    @second.tool("payments.refund")
    def resumed(amount: int) -> str:
        return "ran"

    with second.context(action_id=action_id):
        assert resumed(2500) == "ran"
    assert [recorded.body["action_id"] for recorded in fake.decisions] == [action_id, action_id]


def test_an_async_function_waits_too(fake: FakePaidwen) -> None:
    fake.script = [status("pending"), status("approved")]
    waits: list[Wait] = []

    async def notify(wait: Wait) -> None:
        waits.append(wait)

    paidwen = client(fake, on_wait=notify)

    @paidwen.tool("payments.refund")
    async def refund(amount: int) -> str:
        return "refunded"

    assert anyio.run(refund, 2500) == "refunded"
    assert len(waits) == 1


def test_the_context_gives_its_action_to_the_first_decision_and_its_session_to_all(
    fake: FakePaidwen,
) -> None:
    fake.answer = lambda request: {
        "action_id": request["action_id"],
        "verdict": "allow",
        "rule_id": None,
        "observed": [],
        "decided_at": "2026-10-05T09:00:00Z",
    }
    paidwen = client(fake)
    with paidwen.context(action_id="6f1c2a7e-3b5d-4c8e-9a10-2f4b6d8e0a12", session_id="s-1"):
        paidwen.decide(tool={"name": "a", "kind": "function"})
        paidwen.decide(tool={"name": "b", "kind": "function"})
    first, second = (recorded.body for recorded in fake.decisions)
    assert first["action_id"] == "6f1c2a7e-3b5d-4c8e-9a10-2f4b6d8e0a12"
    assert second["action_id"] != first["action_id"]
    assert first["context"]["session_id"] == "s-1"
    assert second["context"]["session_id"] == "s-1"


def test_a_tool_call_sends_its_trace(fake: FakePaidwen) -> None:
    fake.answer = lambda request: {
        "action_id": request["action_id"],
        "verdict": "allow",
        "rule_id": None,
        "observed": [],
        "decided_at": "2026-10-05T09:00:00Z",
    }
    paidwen = client(fake)

    @paidwen.tool("search")
    def search() -> None:
        return None

    search()
    recorded = fake.decisions[0]
    header = recorded.headers["traceparent"]
    assert header.startswith("00-") and header.endswith("-01") and len(header) == 55
    assert recorded.body["context"]["trace_id"] == header.split("-")[1]

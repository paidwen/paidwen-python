from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from fake import FakePaidwen, serve, until

from paidwen import Paidwen, PaidwenError, RequestError

ACTION_ID = "6f1c2a7e-3b5d-4c8e-9a10-2f4b6d8e0a12"
BASE = "https://api.example.test"
TOOL = {"name": "payments.refund", "kind": "function"}


class Platform:
    def __init__(self) -> None:
        self.exchanges: list[dict[str, list[str]]] = []
        self.authorizations: list[str] = []
        self.issued = 0
        self.expires_in = 900
        self.exchange_status = 200
        self.exchange_body: Any = None
        self.decision_status: list[int] = []
        self.down = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/v1/oauth/token":
            self.exchanges.append(parse_qs(request.content.decode()))
            if self.exchange_status != 200:
                return httpx.Response(
                    self.exchange_status, json=self.exchange_body or {"error": "invalid_request"}
                )
            self.issued += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"agent-token-{self.issued}",
                    "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    "token_type": "Bearer",
                    "expires_in": self.expires_in,
                },
            )
        self.authorizations.append(request.headers.get("authorization", ""))
        status = self.decision_status.pop(0) if self.decision_status else 200
        if status != 200:
            return httpx.Response(
                status, json={"type": "x", "title": "The agent token is invalid.", "status": status}
            )
        return httpx.Response(
            200,
            json={
                "action_id": ACTION_ID,
                "verdict": "allow",
                "rule_id": None,
                "observed": [],
                "decided_at": datetime.now(timezone.utc).isoformat(),
            },
        )


@pytest.fixture
def platform() -> Platform:
    return Platform()


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.write_text("platform-token-1\n")
    return path


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("PAIDWEN_API_KEY", raising=False)
    monkeypatch.delenv("PAIDWEN_IDENTITY_TOKEN_FILE", raising=False)
    yield


def client(platform: Platform, **options: Any) -> Paidwen:
    return Paidwen(
        base_url=BASE,
        journal=False,
        http_client=httpx.Client(transport=httpx.MockTransport(platform.handle)),
        **options,
    )


def ask(paidwen: Paidwen) -> str:
    return paidwen.decide(tool=TOOL, action_id=ACTION_ID)["verdict"]


def test_the_token_of_the_file_is_exchanged_once_and_the_agent_token_is_sent(
    platform: Platform, token_file: Path
) -> None:
    paidwen = client(platform, identity_token_file=str(token_file))
    ask(paidwen)
    ask(paidwen)
    assert len(platform.exchanges) == 1
    form = platform.exchanges[0]
    assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:token-exchange"]
    assert form["subject_token"] == ["platform-token-1"]
    assert form["subject_token_type"] == ["urn:ietf:params:oauth:token-type:jwt"]
    assert platform.authorizations == ["Bearer agent-token-1", "Bearer agent-token-1"]


def test_less_than_five_minutes_left_exchanges_again_and_reads_the_file_again(
    platform: Platform, token_file: Path
) -> None:
    platform.expires_in = 240
    paidwen = client(platform, identity_token_file=str(token_file))
    ask(paidwen)
    token_file.write_text("platform-token-2")
    ask(paidwen)
    assert [form["subject_token"] for form in platform.exchanges] == [
        ["platform-token-1"],
        ["platform-token-2"],
    ]
    assert platform.authorizations == ["Bearer agent-token-1", "Bearer agent-token-2"]


def test_simultaneous_decisions_share_one_exchange(platform: Platform, token_file: Path) -> None:
    paidwen = client(platform, identity_token_file=str(token_file))
    with ThreadPoolExecutor(max_workers=4) as pool:
        verdicts = list(pool.map(lambda _: ask(paidwen), range(4)))
    assert verdicts == ["allow"] * 4
    assert len(platform.exchanges) == 1


def test_a_valid_token_stays_when_its_renewal_finds_paidwen_down(
    platform: Platform, token_file: Path
) -> None:
    platform.expires_in = 240
    paidwen = client(platform, identity_token_file=str(token_file))
    ask(paidwen)
    platform.exchange_status = 503
    assert ask(paidwen) == "allow"
    assert platform.authorizations == ["Bearer agent-token-1", "Bearer agent-token-1"]


def test_a_refused_exchange_raises_and_the_action_never_runs(
    platform: Platform, token_file: Path
) -> None:
    platform.exchange_status = 400
    platform.exchange_body = {
        "error": "invalid_request",
        "error_description": "No enabled trust rule of the organization accepts the subject token.",
    }
    paidwen = client(platform, identity_token_file=str(token_file))
    ran = []

    @paidwen.tool("payments.refund")
    def refund(amount: int) -> None:
        ran.append(amount)

    with pytest.raises(RequestError, match="No enabled trust rule") as raised:
        refund(10)
    assert raised.value.status == 400
    assert ran == []
    assert platform.authorizations == []


def test_an_exchange_without_paidwen_applies_on_unavailable(
    platform: Platform, token_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    platform.exchange_status = 503
    with caplog.at_level(logging.WARNING, logger="paidwen"):
        assert ask(client(platform, identity_token_file=str(token_file))) == "block"
        allowing = client(platform, identity_token_file=str(token_file), on_unavailable="allow")
        assert ask(allowing) == "allow"
    platform.down = True
    assert ask(client(platform, identity_token_file=str(token_file))) == "block"
    assert "token exchange" in caplog.text


def test_a_refused_agent_token_is_exchanged_again_once(
    platform: Platform, token_file: Path
) -> None:
    platform.decision_status = [401]
    assert ask(client(platform, identity_token_file=str(token_file))) == "allow"
    assert len(platform.exchanges) == 2
    assert platform.authorizations == ["Bearer agent-token-1", "Bearer agent-token-2"]
    platform.decision_status = [401, 401]
    with pytest.raises(RequestError) as raised:
        ask(client(platform, identity_token_file=str(token_file)))
    assert raised.value.status == 401


def test_the_identity_token_comes_from_a_function(platform: Platform) -> None:
    calls: list[int] = []

    def token() -> str:
        calls.append(1)
        return f"from-function-{len(calls)}"

    ask(client(platform, identity_token=token))
    assert platform.exchanges[0]["subject_token"] == ["from-function-1"]


def test_two_ways_to_prove_the_identity_are_refused(
    platform: Platform, token_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(PaidwenError, match="only one"):
        client(platform, api_key="pw_test_key", identity_token_file=str(token_file))
    monkeypatch.setenv("PAIDWEN_API_KEY", "pw_test_key")
    monkeypatch.setenv("PAIDWEN_IDENTITY_TOKEN_FILE", str(token_file))
    with pytest.raises(PaidwenError, match="Keep only one"):
        client(platform)
    monkeypatch.delenv("PAIDWEN_API_KEY")
    ask(client(platform))
    assert len(platform.exchanges) == 1


def test_an_unreadable_identity_token_raises(
    platform: Platform, token_file: Path, tmp_path: Path
) -> None:
    with pytest.raises(PaidwenError, match="could not read the identity token"):
        ask(client(platform, identity_token_file=str(tmp_path / "missing")))
    token_file.write_text("   ")
    with pytest.raises(PaidwenError, match="identity token is empty"):
        ask(client(platform, identity_token_file=str(token_file)))


def test_the_journal_sends_the_spans_with_the_agent_token() -> None:
    with serve() as fake:
        journal(fake)


def journal(fake: FakePaidwen) -> None:
    paidwen = Paidwen(identity_token=lambda: "platform-token", base_url=fake.url)

    @paidwen.tool("payments.refund")
    def refund(amount: int) -> str:
        return "ok"

    assert refund(12) == "ok"
    paidwen.flush()
    until("the spans", lambda: len(fake.spans()) == 1)
    assert fake.decisions[0].headers["authorization"] == "Bearer agent-token-1"
    assert fake.traces[0].headers["authorization"] == "Bearer agent-token-1"
    assert fake.exchanges[0]["subject_token"] == ["platform-token"]
    paidwen.close()

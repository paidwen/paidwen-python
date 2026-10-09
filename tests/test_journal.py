from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anyio
import pytest
from fake import FakePaidwen, Span, decision, serve, until
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from paidwen import DeniedError, Paidwen

CAPTURE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
RESULT = "gen_ai.tool.call.result"


@pytest.fixture
def fake() -> Iterator[FakePaidwen]:
    with serve() as server:
        yield server


def capturing(mode: str | None, fake: FakePaidwen, monkeypatch: pytest.MonkeyPatch) -> Paidwen:
    if mode is None:
        monkeypatch.delenv(CAPTURE, raising=False)
    else:
        monkeypatch.setenv(CAPTURE, mode)
    paidwen = Paidwen("pw_test_key", base_url=fake.url)
    monkeypatch.delenv(CAPTURE, raising=False)
    return paidwen


def sent(paidwen: Paidwen, fake: FakePaidwen, count: int) -> list[Span]:
    paidwen.flush()
    until("the spans", lambda: len(fake.spans()) == count)
    paidwen.close()
    return fake.spans()


def test_each_tool_call_is_an_execute_tool_span_sent_with_the_key_of_the_agent(
    fake: FakePaidwen,
) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)

    @paidwen.tool("payments.refund")
    def refund(amount: int) -> str:
        return "ok"

    assert refund(12) == "ok"
    fake.answer = lambda request: decision(request, "block", rule_id="no-refunds")
    with pytest.raises(DeniedError):
        refund(13)
    paidwen.flush()
    until("the spans", lambda: len(fake.spans()) == 2)
    assert fake.traces[0].headers["authorization"] == "Bearer pw_test_key"
    assert fake.traces[0].headers["content-type"] == "application/x-protobuf"
    allowed, denied = fake.spans()
    assert allowed.name == "execute_tool payments.refund"
    assert allowed.attributes["gen_ai.operation.name"] == "execute_tool"
    assert allowed.attributes["gen_ai.tool.name"] == "payments.refund"
    assert allowed.attributes["paidwen.tool.kind"] == "function"
    assert allowed.attributes["paidwen.action.id"] == fake.decisions[0].body["action_id"]
    assert allowed.events == ["paidwen.action.started"]
    assert allowed.status == 1
    assert denied.events == []
    assert denied.status == 2
    assert denied.attributes["error.type"] == "DeniedError"
    assert fake.decisions[0].headers["traceparent"].split("-")[2] == allowed.span_id
    paidwen.close()


def test_an_async_tool_call_is_journaled_too(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)

    @paidwen.tool("search")
    async def search() -> str:
        return "found"

    assert anyio.run(search) == "found"
    paidwen.flush()
    until("the span", lambda: len(fake.spans()) == 1)
    assert fake.spans()[0].events == ["paidwen.action.started"]
    paidwen.close()


def test_a_model_call_leaves_at_once_and_the_next_question_waits_for_it(
    fake: FakePaidwen,
) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)
    provider = TracerProvider()
    provider.add_span_processor(paidwen.span_processor())
    tracer = provider.get_tracer("an-instrumentation")
    tracer.start_span(
        "chat gpt-4o-mini",
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 10_000,
            "gen_ai.usage.output_tokens": 1_000,
        },
    ).end()
    tracer.start_span("GET /health", attributes={"http.request.method": "GET"}).end()
    paidwen.decide(tool={"name": "search", "kind": "function"})
    assert fake.traces, "the model call left before the question"
    assert fake.traces[-1].at <= fake.decisions[0].at
    spans = fake.spans()
    assert len(spans) == 1, "only the model call is sent"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4o-mini"
    provider.shutdown()
    paidwen.close()


def test_a_model_call_after_the_close_is_dropped(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)
    provider = TracerProvider()
    provider.add_span_processor(paidwen.span_processor())
    paidwen.close()
    provider.get_tracer("late").start_span(
        "chat gpt-4o-mini", attributes={"gen_ai.operation.name": "chat"}
    ).end()
    assert fake.traces == []
    provider.shutdown()


def test_the_tool_call_carries_the_session_of_the_context(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)

    @paidwen.tool("search")
    def search() -> None:
        return None

    with paidwen.context(session_id="session-4821"):
        search()
    paidwen.flush()
    until("the span", lambda: len(fake.spans()) == 1)
    assert fake.spans()[0].attributes["gen_ai.conversation.id"] == "session-4821"
    paidwen.close()


def test_the_copy_of_a_model_call_carries_the_session_of_the_context(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)
    own = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(paidwen.span_processor())
    provider.add_span_processor(SimpleSpanProcessor(own))
    with paidwen.context(session_id="session-4821"):
        provider.get_tracer("an-instrumentation").start_span(
            "chat gpt-4o-mini",
            attributes={"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o-mini"},
        ).end()
    provider.force_flush()
    until("the model call", lambda: len(fake.spans()) == 1)
    [sent] = fake.spans()
    [kept] = own.get_finished_spans()
    assert sent.attributes["gen_ai.conversation.id"] == "session-4821"
    assert sent.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert sent.name == "chat gpt-4o-mini"
    assert sent.trace_id == f"{kept.context.trace_id:032x}"
    assert sent.span_id == f"{kept.context.span_id:016x}"
    assert "gen_ai.conversation.id" not in (kept.attributes or {})
    provider.shutdown()
    paidwen.close()


def test_a_model_call_keeps_the_session_it_already_carries(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)
    provider = TracerProvider()
    provider.add_span_processor(paidwen.span_processor())
    with paidwen.context(session_id="session-4821"):
        provider.get_tracer("an-instrumentation").start_span(
            "chat gpt-4o-mini",
            attributes={"gen_ai.operation.name": "chat", "gen_ai.conversation.id": "thread-17"},
        ).end()
    provider.force_flush()
    until("the model call", lambda: len(fake.spans()) == 1)
    assert fake.spans()[0].attributes["gen_ai.conversation.id"] == "thread-17"
    provider.shutdown()
    paidwen.close()


def test_no_span_carries_a_session_outside_a_context(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)
    provider = TracerProvider()
    provider.add_span_processor(paidwen.span_processor())

    @paidwen.tool("search")
    def search() -> None:
        return None

    provider.get_tracer("an-instrumentation").start_span(
        "chat gpt-4o-mini", attributes={"gen_ai.operation.name": "chat"}
    ).end()
    search()
    paidwen.flush()
    until("the spans", lambda: len(fake.spans()) == 2)
    assert all("gen_ai.conversation.id" not in span.attributes for span in fake.spans())
    provider.shutdown()
    paidwen.close()


def test_the_tool_call_is_a_child_of_the_active_span_of_the_agent(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url)
    provider = TracerProvider()

    @paidwen.tool("search")
    def search() -> None:
        return None

    with provider.get_tracer("agent").start_as_current_span("conversation") as parent:
        search()
    paidwen.flush()
    until("the span", lambda: len(fake.spans()) == 1)
    assert fake.spans()[0].trace_id == f"{parent.get_span_context().trace_id:032x}"
    provider.shutdown()
    paidwen.close()


def test_journal_false_sends_nothing(fake: FakePaidwen) -> None:
    paidwen = Paidwen("pw_test_key", base_url=fake.url, journal=False)

    @paidwen.tool("search")
    def search() -> None:
        return None

    search()
    paidwen.flush()
    paidwen.close()
    assert fake.traces == []


def test_the_tool_call_carries_what_the_tool_returns_when_the_content_capture_is_on(
    fake: FakePaidwen, monkeypatch: pytest.MonkeyPatch
) -> None:
    paidwen = capturing("true", fake, monkeypatch)
    refunded = {"refund": "re_1", "amount": 120, "currency": "EUR", "note": "remboursé"}

    @paidwen.tool("search")
    async def search() -> str:
        return "3 orders"

    @paidwen.tool("payments.refund")
    def refund(amount: int) -> dict[str, Any]:
        return refunded

    assert anyio.run(search) == "3 orders"
    assert refund(120) is refunded
    first, second = sent(paidwen, fake, 2)
    assert first.attributes[RESULT] == "3 orders"
    assert second.attributes[RESULT] == (
        '{"refund":"re_1","amount":120,"currency":"EUR","note":"remboursé"}'
    )


@pytest.mark.parametrize("mode", ["SPAN_ONLY", "span_and_event"])
def test_span_only_and_span_and_event_capture_the_result_in_upper_or_lower_case(
    fake: FakePaidwen, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    paidwen = capturing(mode, fake, monkeypatch)

    @paidwen.tool("search")
    def search() -> str:
        return "3 orders"

    search()
    [span] = sent(paidwen, fake, 1)
    assert span.attributes[RESULT] == "3 orders"


@pytest.mark.parametrize("mode", [None, "false", "EVENT_ONLY"])
def test_the_tool_call_carries_no_result_when_the_content_capture_is_off(
    fake: FakePaidwen, monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    paidwen = capturing(mode, fake, monkeypatch)

    @paidwen.tool("search")
    def search() -> str:
        return "3 orders"

    search()
    [span] = sent(paidwen, fake, 1)
    assert RESULT not in span.attributes


def test_the_tool_call_carries_no_result_when_the_tool_raises_or_paidwen_refuses_it(
    fake: FakePaidwen, monkeypatch: pytest.MonkeyPatch
) -> None:
    paidwen = capturing("true", fake, monkeypatch)

    @paidwen.tool("search")
    def search(fail: bool) -> str:
        if fail:
            raise TimeoutError("The search timed out.")
        return "3 orders"

    with pytest.raises(TimeoutError):
        search(True)
    fake.answer = lambda request: decision(request, "block", rule_id="no-search")
    with pytest.raises(DeniedError):
        search(False)
    spans = sent(paidwen, fake, 2)
    assert [(span.status, RESULT in span.attributes) for span in spans] == [(2, False), (2, False)]


def test_a_result_that_json_cannot_carry_is_not_sent_and_the_tool_still_returns_it(
    fake: FakePaidwen, monkeypatch: pytest.MonkeyPatch
) -> None:
    paidwen = capturing("true", fake, monkeypatch)
    opaque = object()

    @paidwen.tool("search")
    def search(empty: bool) -> object:
        return None if empty else opaque

    assert search(False) is opaque
    assert search(True) is None
    spans = sent(paidwen, fake, 2)
    assert [(span.status, RESULT in span.attributes) for span in spans] == [(1, False), (1, False)]


def test_a_result_longer_than_65_536_characters_is_cut(
    fake: FakePaidwen, monkeypatch: pytest.MonkeyPatch
) -> None:
    paidwen = capturing("true", fake, monkeypatch)
    pages = "x" * 70_000

    @paidwen.tool("search")
    def search() -> str:
        return pages

    assert search() is pages
    [span] = sent(paidwen, fake, 1)
    assert span.attributes[RESULT] == pages[:65_536]

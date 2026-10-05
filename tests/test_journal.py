from __future__ import annotations

from collections.abc import Iterator

import anyio
import pytest
from fake import FakePaidwen, decision, serve, until
from opentelemetry.sdk.trace import TracerProvider

from paidwen import DeniedError, Paidwen


@pytest.fixture
def fake() -> Iterator[FakePaidwen]:
    with serve() as server:
        yield server


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

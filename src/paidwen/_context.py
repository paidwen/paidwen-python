from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from opentelemetry import baggage, trace
from opentelemetry import context as otel_context
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace import Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

AGENT_BAGGAGE = "paidwen.agent.id"
CALLER_MAX_LENGTH = 64
TRACEPARENT = "traceparent"
TRACESTATE = "tracestate"

Carrier: TypeAlias = dict[str, str]

_W3C = CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])


@dataclass
class _Pending:
    id: str
    used: bool = False


@dataclass(frozen=True)
class Scope:
    pending: _Pending | None = None
    session_id: str | None = None
    in_action: bool = False
    silenced: bool = False
    carrier: Carrier | None = None


_EMPTY = Scope()
_scope: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("paidwen_scope", default=None)


def scope() -> Scope:
    return _scope.get() or _EMPTY


@contextmanager
def within(**changes: Any) -> Iterator[None]:
    token = _scope.set(replace(scope(), **changes))
    try:
        yield
    finally:
        _scope.reset(token)


@contextmanager
def attached(otel: Context) -> Iterator[None]:
    token = otel_context.attach(otel)
    try:
        yield
    finally:
        otel_context.detach(token)


def _received(headers: Mapping[str, Any]) -> Carrier:
    values: dict[str, list[str]] = {}
    for name, value in headers.items():
        if isinstance(value, str):
            values.setdefault(str(name).lower(), []).append(value)
    return {name: ",".join(found) for name, found in values.items()}


@contextmanager
def context(
    action_id: str | None = None,
    session_id: str | None = None,
    headers: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    changes: dict[str, Any] = {}
    if action_id is not None:
        changes["pending"] = _Pending(action_id)
    if session_id is not None:
        changes["session_id"] = session_id
    with within(**changes):
        if headers is None:
            yield
            return
        with attached(_W3C.extract(_received(headers), otel_context.get_current())):
            yield


def take_action_id() -> str | None:
    pending = scope().pending
    if pending is None or pending.used:
        return None
    pending.used = True
    return pending.id


def silenced() -> AbstractContextManager[None]:
    return within(silenced=True)


@contextmanager
def inside_action(tool: Context, carrier: Carrier) -> Iterator[None]:
    with within(in_action=True, carrier=carrier), attached(tool):
        yield


def tool_context(agent_id: str | None) -> Context:
    current = otel_context.get_current()
    if agent_id is None:
        return current
    return baggage.set_baggage(AGENT_BAGGAGE, agent_id, current)


def caller(agent_id: str | None) -> str | None:
    value = baggage.get_baggage(AGENT_BAGGAGE)
    if isinstance(value, str) and value and value != agent_id and len(value) <= CALLER_MAX_LENGTH:
        return value
    return None


def propagated(tool: Context, span: Span) -> Carrier:
    carrier: Carrier = {}
    _W3C.inject(carrier, trace.set_span_in_context(span, tool))
    return carrier


def missing(carrier: Carrier, present: Callable[[str], bool]) -> Carrier:
    traced = present(TRACEPARENT)
    return {
        name: value
        for name, value in carrier.items()
        if not present(name) and not (traced and name == TRACESTATE)
    }

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any

import requests
from opentelemetry import context as otel_context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from ._context import scope, silenced

ACTION_STARTED = "paidwen.action.started"
CONVERSATION_ID = "gen_ai.conversation.id"
SETTLE_LIMIT = 1.0
MODEL_OPERATIONS = frozenset({"chat", "text_completion", "generate_content", "embeddings"})
TOOL_RESULT = "gen_ai.tool.call.result"
ATTRIBUTE_LENGTH_LIMIT = 65_536

logger = logging.getLogger("paidwen")


def quiet_call(function: Any, *args: Any) -> Any:
    token = otel_context.attach(otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        with silenced():
            return function(*args)
    finally:
        otel_context.detach(token)


class _SilentExporter(SpanExporter):
    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result: SpanExportResult = quiet_call(self._inner.export, spans)
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def is_model_call(span: ReadableSpan) -> bool:
    operation = (span.attributes or {}).get("gen_ai.operation.name")
    return isinstance(operation, str) and operation in MODEL_OPERATIONS


def _session_attributes() -> dict[str, Any]:
    session = scope().session_id
    return {} if session is None else {CONVERSATION_ID: session}


def _in_session(span: ReadableSpan) -> ReadableSpan:
    session = scope().session_id
    attributes = span.attributes or {}
    if session is None or CONVERSATION_ID in attributes:
        return span
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes={**attributes, CONVERSATION_ID: session},
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def dumped(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump(by_alias=True, mode="json", exclude_none=True) if callable(dump) else value


def _serialized(result: Any) -> str | None:
    if result is None or isinstance(result, str):
        return result
    try:
        return json.dumps(dumped(result), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return None


class _ModelProcessor(SpanProcessor):
    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    def on_end(self, span: ReadableSpan) -> None:
        self._journal.report(span)

    def shutdown(self) -> None:
        self._journal.settle()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._journal.settle()
        return True


class _Authorization(requests.auth.AuthBase):
    def __init__(self, authorization: Callable[[], str]) -> None:
        self._authorization = authorization

    def __call__(self, request: Any) -> Any:
        try:
            request.headers["authorization"] = self._authorization()
        except Exception as error:
            logger.warning("Paidwen could not prove the identity of the agent: %s", error)
        return request


class Journal:
    def __init__(
        self,
        endpoint: str,
        authorization: Callable[[], str],
        enabled: bool,
        capture_content: bool,
        version: str,
    ) -> None:
        self._exporter: SpanExporter | None = None
        self._capture_content = capture_content
        self._provider = TracerProvider(
            resource=Resource.create(
                {"service.name": "paidwen-sdk", "telemetry.sdk.language": "python"}
            ),
            span_limits=SpanLimits(max_attribute_length=ATTRIBUTE_LENGTH_LIMIT),
        )
        if enabled:
            session = requests.Session()
            session.auth = _Authorization(authorization)
            self._exporter = _SilentExporter(
                OTLPSpanExporter(endpoint=endpoint, timeout=10, session=session)
            )
            self._provider.add_span_processor(
                BatchSpanProcessor(self._exporter, schedule_delay_millis=1000)
            )
        self._tracer = self._provider.get_tracer("paidwen", version)
        self._sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paidwen-journal")
        self._pending: set[Future[Any]] = set()
        self._lock = threading.Lock()
        self._closed = False

    def start_tool(self, name: str, attributes: dict[str, Any]) -> Span:
        return self._tracer.start_span(
            f"execute_tool {name}",
            context=otel_context.get_current(),
            kind=SpanKind.INTERNAL,
            attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
            | _session_attributes()
            | attributes,
        )

    def set_result(self, span: Span, result: Any) -> None:
        if not self._capture_content:
            return
        value = _serialized(result)
        if value is not None:
            span.set_attribute(TOOL_RESULT, value)

    def report(self, span: ReadableSpan) -> None:
        exporter = self._exporter
        if exporter is None or not is_model_call(span):
            return
        with self._lock:
            if self._closed:
                return
            future = self._sender.submit(exporter.export, [_in_session(span)])
            self._pending.add(future)
        future.add_done_callback(self._done)

    def _done(self, future: Future[Any]) -> None:
        with self._lock:
            self._pending.discard(future)
        if future.exception() is not None or future.result() != SpanExportResult.SUCCESS:
            logger.warning("Paidwen could not send a model call to the journal.")

    def settle(self) -> None:
        with self._lock:
            pending = list(self._pending)
        if pending:
            wait(pending, timeout=SETTLE_LIMIT)

    def processor(self) -> SpanProcessor:
        return _ModelProcessor(self)

    def flush(self) -> None:
        self.settle()
        self._provider.force_flush()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
        self.settle()
        self._provider.shutdown()
        self._sender.shutdown(wait=True)


def started(span: Span) -> None:
    span.add_event(ACTION_STARTED)


def finish(span: Span, error: BaseException | None = None) -> None:
    if error is None:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_attribute("error.type", type(error).__name__)
        span.set_status(Status(StatusCode.ERROR, str(error)))
    span.end()


def traceparent(span: Span) -> str:
    context = span.get_span_context()
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-01"


def trace_id(span: Span) -> str:
    return f"{span.get_span_context().trace_id:032x}"

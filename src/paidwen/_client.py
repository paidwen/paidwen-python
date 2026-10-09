from __future__ import annotations

import functools
import inspect
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any, Literal, TypeVar, cast

import anyio
import anyio.to_thread
import httpx
from opentelemetry import context as otel_context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span

from . import _context
from ._context import Carrier
from ._contract import (
    ActionContext,
    ApprovalStatus,
    Cost,
    Decision,
    DecisionRequest,
    Problem,
    Tool,
)
from ._credentials import (
    Credentials,
    FederatedCredentials,
    IdentityToken,
    KeyCredentials,
    Unavailable,
    identity_token_file,
)
from ._errors import DeniedError, PaidwenError, PendingError, RequestError, reason_of
from ._journal import Journal, finish, quiet_call, started, trace_id, traceparent

Fallback = Literal["allow", "block"]
OnPending = Literal["wait", "raise"]

VERSION = "0.9.0"
DEFAULT_BASE_URL = "https://api.paidwen.com"
QUOTA_EXCEEDED = "https://paidwen.com/problems/quota-exceeded"
DEFAULT_TIMEOUT = 3.0
SLOW_DOWN_SECONDS = 5
WAIT_GRACE_SECONDS = 60
MISSING_IDENTITY = (
    "Paidwen needs an agent key or an identity token: pass api_key, identity_token_file or "
    "identity_token, or set PAIDWEN_API_KEY or PAIDWEN_IDENTITY_TOKEN_FILE."
)

_CAPTURED = "_paidwen_captured"

logger = logging.getLogger("paidwen")

F = TypeVar("F", bound=Callable[..., Any])
R = TypeVar("R")


@dataclass(frozen=True)
class Wait:
    decision: Decision
    approval_id: str
    expires_at: datetime
    interval: int
    on_expiry: Literal["deny", "allow"]


class _HttpStatusError(Exception):
    pass


def _runs(status: ApprovalStatus) -> bool:
    return status["status"] == "approved" or (
        status["status"] == "expired" and status["on_expiry"] == "allow"
    )


def _decided_at(decision: Decision) -> datetime:
    try:
        return datetime.fromisoformat(decision["decided_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return datetime.now(timezone.utc)


class Paidwen:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        identity_token_file: str | None = None,
        identity_token: IdentityToken | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        on_unavailable: Fallback = "block",
        on_pending: OnPending = "wait",
        on_wait: Callable[[Wait], object] | None = None,
        journal: bool = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        if on_unavailable not in ("allow", "block"):
            raise PaidwenError('on_unavailable must be "allow" or "block".')
        if on_pending not in ("wait", "raise"):
            raise PaidwenError('on_pending must be "wait" or "raise".')
        if not timeout > 0:
            raise PaidwenError("timeout must be a positive number of seconds.")
        base = (base_url or os.environ.get("PAIDWEN_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._base = base
        self._timeout = timeout
        self._on_unavailable: Fallback = on_unavailable
        self._on_pending: OnPending = on_pending
        self._on_wait = on_wait
        self._client = http_client or httpx.Client()
        self._owns_client = http_client is None
        self._credentials = _credentials_of(
            api_key, identity_token_file, identity_token, base, self._client, timeout
        )
        self._journal = Journal(
            f"{base}/v1/traces", self._credentials.authorization, journal, VERSION
        )
        self._release_http: Callable[[], None] | None = None
        self._agent_id: str | None = None

    def decide(
        self,
        *,
        tool: Tool,
        params: Mapping[str, Any] | None = None,
        cost: Cost | None = None,
        context: ActionContext | None = None,
        action_id: str | None = None,
    ) -> Decision:
        request = self._request(tool, params, cost, context, action_id, None)
        return self._send(request, None)

    def _request(
        self,
        tool: Tool,
        params: Mapping[str, Any] | None,
        cost: Cost | None,
        context: ActionContext | None,
        action_id: str | None,
        span: Span | None,
    ) -> DecisionRequest:
        request: DecisionRequest = {
            "action_id": action_id or _context.take_action_id() or str(uuid.uuid4()),
            "tool": tool,
        }
        if params is not None:
            request["params"] = dict(params)
        if cost is not None:
            request["cost"] = cost
        known: dict[str, Any] = {}
        session = _context.scope().session_id
        if session is not None:
            known["session_id"] = session
        if span is not None:
            known["trace_id"] = trace_id(span)
        caller = _context.caller(self._agent_id)
        if caller is not None:
            known["caller_agent_id"] = caller
        if known or context is not None:
            request["context"] = cast(ActionContext, {**known, **(context or {})})
        return request

    def _send(self, request: DecisionRequest, span: Span | None, retry: bool = True) -> Decision:
        self._journal.settle()
        try:
            authorization = self._credentials.authorization()
        except Unavailable as error:
            return self._fallback(request, error.message, error.reason)
        headers = {"authorization": authorization, "accept": "application/json"}
        if span is not None:
            headers["traceparent"] = traceparent(span)
        try:
            response = quiet_call(
                functools.partial(
                    self._client.post,
                    f"{self._base}/v1/decisions",
                    json=request,
                    headers=headers,
                    timeout=self._timeout,
                )
            )
        except httpx.HTTPError as error:
            return self._fallback(request, "Paidwen did not answer", str(error))
        if response.status_code >= 500:
            problem = _problem(response)
            return self._fallback(
                request, _unavailable(response.status_code, problem), reason_of(problem)
            )
        if response.status_code == 401 and retry and self._credentials.federated:
            self._credentials.reject(authorization)
            return self._send(request, span, False)
        if not response.is_success:
            raise RequestError(response.status_code, _problem(response))
        try:
            body = response.json()
        except ValueError as error:
            return self._fallback(request, "Paidwen did not answer", str(error))
        if isinstance(body, dict) and isinstance(body.get("agent_id"), str) and body["agent_id"]:
            self._agent_id = body["agent_id"]
        return cast(Decision, body)

    def _poll(self, approval_id: str, retry: bool = True) -> tuple[str, ApprovalStatus | None]:
        try:
            authorization = self._credentials.authorization()
        except Unavailable:
            return "unavailable", None
        try:
            response = quiet_call(
                functools.partial(
                    self._client.get,
                    f"{self._base}/v1/approvals/{approval_id}",
                    headers={"authorization": authorization, "accept": "application/json"},
                    timeout=self._timeout,
                )
            )
        except httpx.HTTPError:
            return "unavailable", None
        if response.status_code == 401 and retry and self._credentials.federated:
            self._credentials.reject(authorization)
            return self._poll(approval_id, False)
        if response.status_code == 429:
            return "slow_down", None
        if response.status_code >= 500:
            return "unavailable", None
        if not response.is_success:
            raise RequestError(response.status_code, _problem(response))
        try:
            body = response.json()
        except ValueError:
            return "unavailable", None
        return "status", cast(ApprovalStatus, body)

    def _deadline(self, decision: Decision) -> tuple[dict[str, Any], datetime]:
        approval = decision.get("approval")
        if approval is None:
            raise PaidwenError("The decision carries no approval to wait for.")
        deadline = _decided_at(decision) + timedelta(
            seconds=approval["expires_in"] + WAIT_GRACE_SECONDS
        )
        return cast(dict[str, Any], approval), deadline

    def _expired(
        self,
        decision: Decision,
        approval: dict[str, Any],
        deadline: datetime,
        interval: int,
        last: ApprovalStatus | None,
    ) -> ApprovalStatus:
        on_expiry = (last or {}).get("on_expiry") or approval.get("on_expiry") or "deny"
        return {
            "id": approval["id"],
            "action_id": decision["action_id"],
            "status": "expired",
            "expires_at": deadline.isoformat(),
            "interval": interval,
            "on_expiry": on_expiry,
        }

    def wait_for_approval(self, decision: Decision) -> ApprovalStatus:
        return self._wait(decision, lambda: None)

    def _wait(self, decision: Decision, pending: Callable[[], None]) -> ApprovalStatus:
        approval, deadline = self._deadline(decision)
        interval = int(approval["interval"])
        last: ApprovalStatus | None = None
        first = True
        while True:
            if not first:
                time.sleep(interval)
            kind, status = self._poll(approval["id"])
            if kind == "status" and status is not None:
                last = status
                if status["status"] != "pending":
                    return status
                interval = max(interval, status["interval"])
            elif kind == "slow_down":
                interval += SLOW_DOWN_SECONDS
            if datetime.now(timezone.utc) >= deadline:
                return self._expired(decision, approval, deadline, interval, last)
            if first:
                first = False
                pending()

    async def _wait_async(
        self, decision: Decision, pending: Callable[[], Awaitable[None]]
    ) -> ApprovalStatus:
        approval, deadline = self._deadline(decision)
        interval = int(approval["interval"])
        last: ApprovalStatus | None = None
        first = True
        while True:
            if not first:
                await anyio.sleep(interval)
            kind, status = await anyio.to_thread.run_sync(self._poll, approval["id"])
            if kind == "status" and status is not None:
                last = status
                if status["status"] != "pending":
                    return status
                interval = max(interval, status["interval"])
            elif kind == "slow_down":
                interval += SLOW_DOWN_SECONDS
            if datetime.now(timezone.utc) >= deadline:
                return self._expired(decision, approval, deadline, interval, last)
            if first:
                first = False
                await pending()

    def _wait_of(self, decision: Decision) -> Wait:
        approval = decision["approval"]
        return Wait(
            decision=decision,
            approval_id=approval["id"],
            expires_at=_decided_at(decision) + timedelta(seconds=approval["expires_in"]),
            interval=approval["interval"],
            on_expiry=approval.get("on_expiry", "deny"),
        )

    def _verdict(self, decision: Decision, span: Span) -> bool:
        span.set_attribute("paidwen.action.id", decision["action_id"])
        verdict = decision["verdict"]
        if decision.get("incident") is not None or verdict in ("block", "quarantine"):
            raise DeniedError(decision)
        if verdict == "allow":
            return True
        if decision.get("approval") is None:
            raise PendingError(decision)
        return False

    def _authorize(
        self, tool: Tool, params: Mapping[str, Any] | None, cost: Cost | None, span: Span
    ) -> None:
        decision = self._send(self._request(tool, params, cost, None, None, span), span)
        if self._verdict(decision, span):
            return

        def pending() -> None:
            if self._on_pending == "raise":
                raise PendingError(decision)
            if self._on_wait is not None:
                notice = self._on_wait(self._wait_of(decision))
                if inspect.isawaitable(notice):
                    cast(Any, notice).close()
                    raise PaidwenError("on_wait returned a coroutine in a synchronous call.")

        status = self._wait(decision, pending)
        if not _runs(status):
            raise DeniedError(decision, status)

    async def _authorize_async(
        self, tool: Tool, params: Mapping[str, Any] | None, cost: Cost | None, span: Span
    ) -> None:
        request = self._request(tool, params, cost, None, None, span)
        decision = await anyio.to_thread.run_sync(self._send, request, span)
        if self._verdict(decision, span):
            return

        async def pending() -> None:
            if self._on_pending == "raise":
                raise PendingError(decision)
            if self._on_wait is not None:
                notice = self._on_wait(self._wait_of(decision))
                if inspect.isawaitable(notice):
                    await notice

        status = await self._wait_async(decision, pending)
        if not _runs(status):
            raise DeniedError(decision, status)

    def _close(self, span: Span, result: Any, status_of: Callable[[Any], int] | None) -> None:
        if status_of is None:
            finish(span)
            return
        code = status_of(result)
        span.set_attribute("http.response.status_code", code)
        finish(span, _HttpStatusError(str(code)) if code >= 400 else None)

    @contextmanager
    def _inside(self, span: Span) -> Iterator[Carrier]:
        tool = _context.tool_context(self._agent_id)
        carrier = _context.propagated(tool, span)
        with _context.inside_action(tool, carrier):
            yield carrier

    def _run(
        self,
        tool: Tool,
        params: Mapping[str, Any] | None,
        action: Callable[[Carrier], R],
        status_of: Callable[[Any], int] | None = None,
        cost: Cost | None = None,
    ) -> R:
        span = self._journal.start_tool(tool["name"], _attributes(tool))
        try:
            self._authorize(tool, params, cost, span)
            started(span)
            with self._inside(span) as carrier:
                result = action(carrier)
        except BaseException as error:
            finish(span, error)
            raise
        self._close(span, result, status_of)
        return result

    async def _run_async(
        self,
        tool: Tool,
        params: Mapping[str, Any] | None,
        action: Callable[[Carrier], Awaitable[R]],
        status_of: Callable[[Any], int] | None = None,
        cost: Cost | None = None,
    ) -> R:
        span = self._journal.start_tool(tool["name"], _attributes(tool))
        try:
            await self._authorize_async(tool, params, cost, span)
            started(span)
            with self._inside(span) as carrier:
                result = await action(carrier)
        except BaseException as error:
            finish(span, error)
            raise
        self._close(span, result, status_of)
        return result

    def skips(self, server: str, ignore: Sequence[str]) -> bool:
        from ._http import ignored

        current = _context.scope()
        if current.silenced or current.in_action:
            return True
        if otel_context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return True
        return ignored(ignore, server)

    def passing(self, server: str, ignore: Sequence[str]) -> Carrier | None:
        from ._http import ignored, server_of

        current = _context.scope()
        if not current.in_action or current.silenced:
            return None
        if otel_context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return None
        if server == server_of(self._base) or ignored(ignore, server):
            return None
        return current.carrier

    def tool(
        self,
        name: str,
        *,
        cost: Cost | Callable[[dict[str, Any]], Cost | None] | None = None,
    ) -> Callable[[F], F]:
        def estimate(params: dict[str, Any]) -> Cost | None:
            return cost(params) if callable(cost) else cost

        def decorate(function: F) -> F:
            signature = inspect.signature(function)
            tool: Tool = {"name": name, "kind": "function"}

            if inspect.iscoroutinefunction(function):

                @functools.wraps(function)
                async def run_async(*args: Any, **kwargs: Any) -> Any:
                    params = _params(signature, args, kwargs)
                    return await self._run_async(
                        tool,
                        params,
                        lambda carrier: function(*args, **kwargs),
                        cost=estimate(params),
                    )

                return cast(F, run_async)

            @functools.wraps(function)
            def run(*args: Any, **kwargs: Any) -> Any:
                params = _params(signature, args, kwargs)
                return self._run(
                    tool, params, lambda carrier: function(*args, **kwargs), cost=estimate(params)
                )

            return cast(F, run)

        return decorate

    def context(
        self,
        action_id: str | None = None,
        session_id: str | None = None,
        headers: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[None]:
        return _context.context(action_id=action_id, session_id=session_id, headers=headers)

    def mcp(self, target: Any) -> Callable[[], None]:
        send = getattr(target, "send_request", None)
        if not callable(send):
            raise PaidwenError(
                "mcp() takes the ClientSession class of the MCP SDK, or one of its sessions."
            )
        if getattr(target, _CAPTURED, False):
            raise PaidwenError("This MCP session is already captured by Paidwen.")
        own = vars(target).get("send_request")
        run = self._run_async

        if isinstance(target, type):

            async def send_class(session: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                call = _tool_call(session, request)
                if call is None:
                    return await send(session, request, *args, **kwargs)
                return await run(
                    call[0],
                    call[1],
                    lambda carrier: send(session, _with_meta(request, carrier), *args, **kwargs),
                )

            wrapped = functools.wraps(send)(send_class)
        else:

            async def send_session(request: Any, *args: Any, **kwargs: Any) -> Any:
                call = _tool_call(target, request)
                if call is None:
                    return await send(request, *args, **kwargs)
                return await run(
                    call[0],
                    call[1],
                    lambda carrier: send(_with_meta(request, carrier), *args, **kwargs),
                )

            wrapped = functools.wraps(send)(send_session)
        target.send_request = wrapped
        setattr(target, _CAPTURED, True)

        def release() -> None:
            if own is None:
                delattr(target, "send_request")
            else:
                target.send_request = own
            delattr(target, _CAPTURED)

        return release

    def http(self, ignore: Sequence[str] = ()) -> Callable[[], None]:
        from ._capture import install

        if self._release_http is not None:
            raise PaidwenError("The HTTP calls of this process are already captured by Paidwen.")
        release = install(self, tuple(ignore))

        def undo() -> None:
            release()
            self._release_http = None

        self._release_http = undo
        return undo

    def span_processor(self) -> SpanProcessor:
        return self._journal.processor()

    def flush(self) -> None:
        self._journal.flush()

    def close(self) -> None:
        if self._release_http is not None:
            self._release_http()
        self._journal.shutdown()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Paidwen:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _fallback(self, request: DecisionRequest, cause: str, reason: str | None) -> Decision:
        logger.warning(
            '%s, so the SDK applies on_unavailable "%s"%s',
            cause,
            self._on_unavailable,
            f": {reason}" if reason else ".",
        )
        return {
            "action_id": request["action_id"],
            "verdict": self._on_unavailable,
            "rule_id": None,
            "observed": [],
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }


def _credentials_of(
    api_key: str | None,
    token_file: str | None,
    identity_token: IdentityToken | None,
    base: str,
    client: httpx.Client,
    timeout: float,
) -> Credentials:
    given = [value for value in (api_key, token_file, identity_token) if value is not None]
    if len(given) > 1:
        raise PaidwenError("Pass only one of api_key, identity_token_file and identity_token.")

    def federated(source: IdentityToken) -> Credentials:
        return FederatedCredentials(source, base, client, timeout)

    if token_file is not None:
        if not token_file:
            raise PaidwenError("identity_token_file must name the file of the identity token.")
        return federated(identity_token_file(token_file))
    if identity_token is not None:
        if not callable(identity_token):
            raise PaidwenError("identity_token must be a function that returns the identity token.")
        return federated(identity_token)
    if api_key is not None:
        if not api_key:
            raise PaidwenError(MISSING_IDENTITY)
        return KeyCredentials(api_key)
    key = os.environ.get("PAIDWEN_API_KEY")
    path = os.environ.get("PAIDWEN_IDENTITY_TOKEN_FILE")
    if key and path:
        raise PaidwenError(
            "Both PAIDWEN_API_KEY and PAIDWEN_IDENTITY_TOKEN_FILE are set. Keep only one of them."
        )
    if key:
        return KeyCredentials(key)
    if path:
        return federated(identity_token_file(path))
    raise PaidwenError(MISSING_IDENTITY)


def _problem(response: httpx.Response) -> Problem | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return cast(Problem, body) if isinstance(body, dict) else None


def _unavailable(status: int, problem: Problem | None) -> str:
    answered = f"Paidwen answered HTTP {status}"
    if problem is not None and problem.get("type") == QUOTA_EXCEEDED:
        return f"{answered} because the organization reached the events its plan allows this month"
    return answered


def _attributes(tool: Tool) -> dict[str, Any]:
    attributes: dict[str, Any] = {"paidwen.tool.kind": tool["kind"]}
    server = tool.get("server")
    if server is not None:
        attributes["paidwen.tool.server"] = server
    return attributes


def _params(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    params: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            params.update(value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            params[name] = list(value)
        else:
            params[name] = value
    return params


def _dumped(request: Any) -> Any:
    dump = getattr(request, "model_dump", None)
    return dump(by_alias=True, mode="json", exclude_none=True) if callable(dump) else request


def _with_meta(request: Any, carrier: Carrier) -> Any:
    data = _dumped(request)
    params = data.get("params") if isinstance(data, Mapping) else None
    if not isinstance(params, Mapping):
        return request
    meta = params.get("_meta")
    present: Mapping[str, Any] = meta if isinstance(meta, Mapping) else {}
    added = _context.missing(carrier, lambda name: name in present)
    if not added:
        return request
    changed = {**data, "params": {**params, "_meta": {**present, **added}}}
    validate = getattr(type(request), "model_validate", None)
    return validate(changed) if callable(validate) else changed


def _tool_call(session: Any, request: Any) -> tuple[Tool, dict[str, Any]] | None:
    data = _dumped(request)
    if not isinstance(data, Mapping) or data.get("method") != "tools/call":
        return None
    params = data.get("params")
    call: Mapping[str, Any] = params if isinstance(params, Mapping) else {}
    name = call.get("name")
    arguments = call.get("arguments")
    tool: Tool = {"name": name if isinstance(name, str) else "", "kind": "mcp"}
    server = getattr(getattr(session, "server_info", None), "name", None)
    if isinstance(server, str) and 1 <= len(server) <= 256:
        tool["server"] = server
    return tool, dict(arguments) if isinstance(arguments, Mapping) else {}

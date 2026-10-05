from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Literal, TypeVar, cast

import httpx

from ._contract import ActionContext, Cost, Decision, DecisionRequest, Problem, Tool
from ._errors import DeniedError, PaidwenError, PendingError, RequestError

Fallback = Literal["allow", "block"]

DEFAULT_BASE_URL = "https://api.paidwen.com"
DEFAULT_TIMEOUT = 3.0

logger = logging.getLogger("paidwen")

F = TypeVar("F", bound=Callable[..., Any])


class Paidwen:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        on_unavailable: Fallback = "block",
        http_client: httpx.Client | None = None,
    ) -> None:
        key = api_key or os.environ.get("PAIDWEN_API_KEY")
        if not key:
            raise PaidwenError("Paidwen needs an agent key: pass api_key or set PAIDWEN_API_KEY.")
        if on_unavailable not in ("allow", "block"):
            raise PaidwenError('on_unavailable must be "allow" or "block".')
        if not timeout > 0:
            raise PaidwenError("timeout must be a positive number of seconds.")
        base = base_url or os.environ.get("PAIDWEN_BASE_URL") or DEFAULT_BASE_URL
        self._api_key = key
        self._endpoint = f"{base.rstrip('/')}/v1/decisions"
        self._timeout = timeout
        self._on_unavailable: Fallback = on_unavailable
        self._client = http_client or httpx.Client()
        self._owns_client = http_client is None

    def decide(
        self,
        *,
        tool: Tool,
        params: Mapping[str, Any] | None = None,
        cost: Cost | None = None,
        context: ActionContext | None = None,
        action_id: str | None = None,
    ) -> Decision:
        request: DecisionRequest = {"action_id": action_id or str(uuid.uuid4()), "tool": tool}
        if params is not None:
            request["params"] = dict(params)
        if cost is not None:
            request["cost"] = cost
        if context is not None:
            request["context"] = context
        try:
            response = self._client.post(
                self._endpoint,
                json=request,
                headers={
                    "authorization": f"Bearer {self._api_key}",
                    "accept": "application/json",
                },
                timeout=self._timeout,
            )
        except httpx.HTTPError as error:
            return self._fallback(request, error)
        if response.status_code >= 500:
            return self._fallback(
                request, PaidwenError(f"Paidwen answered HTTP {response.status_code}.")
            )
        try:
            body = response.json()
        except ValueError as error:
            if response.is_success:
                return self._fallback(request, error)
            body = None
        if not response.is_success:
            problem = cast(Problem, body) if isinstance(body, dict) else None
            raise RequestError(response.status_code, problem)
        return cast(Decision, body)

    def tool(self, name: str) -> Callable[[F], F]:
        def decorate(function: F) -> F:
            signature = inspect.signature(function)

            def ask(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Decision:
                return self.decide(
                    tool={"name": name, "kind": "function"},
                    params=_params(signature, args, kwargs),
                )

            if inspect.iscoroutinefunction(function):

                @functools.wraps(function)
                async def run_async(*args: Any, **kwargs: Any) -> Any:
                    _enforce(await asyncio.to_thread(ask, args, kwargs))
                    return await function(*args, **kwargs)

                return cast(F, run_async)

            @functools.wraps(function)
            def run(*args: Any, **kwargs: Any) -> Any:
                _enforce(ask(args, kwargs))
                return function(*args, **kwargs)

            return cast(F, run)

        return decorate

    def close(self) -> None:
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

    def _fallback(self, request: DecisionRequest, cause: BaseException) -> Decision:
        logger.warning(
            'Paidwen did not answer, so the SDK applies on_unavailable "%s": %s',
            self._on_unavailable,
            cause,
        )
        return {
            "action_id": request["action_id"],
            "verdict": self._on_unavailable,
            "rule_id": None,
            "observed": [],
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }


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


def _enforce(decision: Decision) -> None:
    if decision["verdict"] in ("block", "quarantine"):
        raise DeniedError(decision)
    if decision["verdict"] in ("require_approval", "escalate"):
        raise PendingError(decision)

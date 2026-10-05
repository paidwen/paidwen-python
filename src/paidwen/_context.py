from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from typing import Any


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


def context(
    action_id: str | None = None, session_id: str | None = None
) -> AbstractContextManager[None]:
    changes: dict[str, Any] = {}
    if action_id is not None:
        changes["pending"] = _Pending(action_id)
    if session_id is not None:
        changes["session_id"] = session_id
    return within(**changes)


def take_action_id() -> str | None:
    pending = scope().pending
    if pending is None or pending.used:
        return None
    pending.used = True
    return pending.id


def silenced() -> AbstractContextManager[None]:
    return within(silenced=True)


def inside_action() -> AbstractContextManager[None]:
    return within(in_action=True)

from __future__ import annotations

from typing import Any

from ._contract import Decision, Problem


class PaidwenError(Exception):
    pass


class DeniedError(PaidwenError):
    def __init__(self, decision: Decision) -> None:
        super().__init__(_describe(decision))
        self.decision = decision


class PendingError(PaidwenError):
    def __init__(self, decision: Decision) -> None:
        super().__init__(_describe(decision))
        self.decision = decision


class RequestError(PaidwenError):
    def __init__(self, status: int, problem: Problem | None) -> None:
        details: dict[str, Any] = dict(problem) if problem else {}
        super().__init__(
            details.get("detail")
            or details.get("title")
            or f"Paidwen refused the request with HTTP {status}."
        )
        self.status = status
        self.problem = problem


def _describe(decision: Decision) -> str:
    rule = "" if decision["rule_id"] is None else f" by rule {decision['rule_id']}"
    return f"Paidwen answered {decision['verdict']} for action {decision['action_id']}{rule}."

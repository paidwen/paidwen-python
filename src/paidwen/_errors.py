from __future__ import annotations

from typing import Any

from ._contract import ApprovalStatus, Decision, Problem


class PaidwenError(Exception):
    pass


class DeniedError(PaidwenError):
    def __init__(self, decision: Decision, approval: ApprovalStatus | None = None) -> None:
        super().__init__(_describe(decision, approval))
        self.decision = decision
        self.approval = approval


class PendingError(PaidwenError):
    def __init__(self, decision: Decision) -> None:
        super().__init__(_describe(decision, None))
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


def _describe(decision: Decision, approval: ApprovalStatus | None) -> str:
    answer = f"Paidwen answered {decision['verdict']} for action {decision['action_id']}"
    if approval is not None and approval["status"] == "denied":
        return f"{answer}, and the approver denied it."
    if approval is not None and approval["status"] == "expired":
        return f"{answer}, and the wait expired without an answer."
    if decision["rule_id"] is not None:
        return f"{answer} by rule {decision['rule_id']}."
    budget = decision.get("budget")
    if budget is not None:
        return f"{answer} because the {budget['exceeded']} budget of the agent is spent."
    permission = decision.get("permission")
    if permission is not None and not permission["allowed"]:
        return f"{answer} because the permissions of the agent do not allow it."
    return f"{answer}."

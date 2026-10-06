from __future__ import annotations

from ._contract import ApprovalStatus, Decision, Problem


class PaidwenError(Exception):
    pass


class DeniedError(PaidwenError):
    def __init__(self, decision: Decision, approval: ApprovalStatus | None = None) -> None:
        super().__init__(_describe(decision, approval))
        self.decision = decision
        self.approval = approval
        self.incident = decision.get("incident")


class PendingError(PaidwenError):
    def __init__(self, decision: Decision) -> None:
        super().__init__(_describe(decision, None))
        self.decision = decision


class RequestError(PaidwenError):
    def __init__(self, status: int, problem: Problem | None) -> None:
        super().__init__(_refusal(status, problem))
        self.status = status
        self.problem = problem


def reason_of(problem: Problem | None) -> str | None:
    if problem is None:
        return None
    for text in (problem.get("detail"), problem.get("title")):
        if isinstance(text, str) and text:
            return text
    return None


def _refusal(status: int, problem: Problem | None) -> str:
    reason = reason_of(problem)
    if status == 403:
        if reason is None:
            return "Paidwen refused the agent with HTTP 403."
        return f"Paidwen refused the agent: {reason}"
    return reason or f"Paidwen refused the request with HTTP {status}."


def _describe(decision: Decision, approval: ApprovalStatus | None) -> str:
    answer = f"Paidwen answered {decision['verdict']} for action {decision['action_id']}"
    if approval is not None and approval["status"] == "denied":
        return f"{answer}, and the approver denied it."
    if approval is not None and approval["status"] == "expired":
        return f"{answer}, and the wait expired without an answer."
    incident = decision.get("incident")
    if incident is not None:
        quarantine = f"the agent is in quarantine (incident {incident['id']})"
        if decision["rule_id"] is None:
            return f"{answer} because {quarantine}."
        return f"{answer} by rule {decision['rule_id']}, and {quarantine}."
    if decision["rule_id"] is not None:
        return f"{answer} by rule {decision['rule_id']}."
    budget = decision.get("budget")
    if budget is not None:
        return f"{answer} because the {budget['exceeded']} budget of the agent is spent."
    permission = decision.get("permission")
    if permission is not None and not permission["allowed"]:
        return f"{answer} because the permissions of the agent do not allow it."
    return f"{answer}."

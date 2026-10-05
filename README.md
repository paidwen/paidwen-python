# Paidwen SDK for Python

Paidwen controls what AI agents do. Before each action, the agent asks Paidwen whether it may act, then applies the answer. The agent keeps its own tool keys and performs the action itself.

```python
from paidwen import Paidwen

paidwen = Paidwen()


@paidwen.tool("payments.refund")
def refund(order: str, amount: float) -> None:
    payments.refund(order, amount)
```

`Paidwen()` reads the agent key from `PAIDWEN_API_KEY`. A call to `refund` asks Paidwen first. It runs when Paidwen allows it, waits when the action needs a human approval, and raises `DeniedError` when Paidwen blocks it or puts the agent in quarantine. When the permissions or the budget of the agent refuse the action, the message of `DeniedError` says so. An async function works the same way.

## Human approval

```python
paidwen = Paidwen(on_wait=lambda wait: notify_user(wait.approval_id, wait.expires_at))
```

When a rule asks for a human approval, the call waits for the answer of the approver. The SDK calls `on_wait` once when the wait starts, so the agent can tell its user. The call runs once the approver approves it, and raises `DeniedError` when the approver denies it. When the wait expires, the SDK applies the `on_expiry` choice of the rule: `"deny"` raises `DeniedError`, `"allow"` runs the call.

With `on_pending="raise"`, the call raises `PendingError` at once and the action does not run. The agent retries later with the same action identifier, and Paidwen gives back the same wait:

```python
with paidwen.context(action_id=action_id):
    refund("ord_42", 2500)
```

## Context

`with paidwen.context(action_id=..., session_id=...)` gives its action identifier to the first decision of the block, and its session to every decision and journal line of the block.

## MCP tool calls

```python
from mcp import ClientSession

paidwen.mcp(ClientSession)
```

`paidwen.mcp(ClientSession)` asks Paidwen before each `tools/call` request of every session of the official MCP SDK in the process, including the sessions that an agent framework creates. The question carries the tool name, its arguments and the name that the MCP server announces. A refused call raises `DeniedError` and never reaches the server.

`paidwen.mcp(session)` captures one session only, for a process that runs several agents with their own keys. `mcp()` returns a function that removes the capture.

## HTTP calls

```python
release = paidwen.http(ignore=["*.internal.example.com"])
```

`paidwen.http()` asks Paidwen before each outgoing HTTP call that httpx, requests, urllib3 or aiohttp makes in the process. The tool name is the method and the path, for example `POST /v1/refunds`, and the server is the host. The parameters carry `method`, `url`, `query`, and `body` when the body is JSON or a form of at most 64 KiB. The SDK never sends the headers, and replaces the value of each sensitive key, such as `password`, `token` or `api_key`, by `REDACTED`.

A refused call raises `DeniedError` and never leaves the process. The SDK asks nothing for its own calls, for the calls made inside an action already decided, and for the hosts that match `ignore`. A process has one HTTP capture. `http()` returns a function that removes it.

## Journal

Each tool call goes to the journal of Paidwen as an OpenTelemetry span, with its decision, its start and its result. To journal the model calls that your OpenTelemetry instrumentations describe, add the span processor of Paidwen to your tracer provider:

```python
provider.add_span_processor(paidwen.span_processor())
```

A model call leaves as soon as it ends, and the next question of the agent waits for it, one second at most, so that the budget sees its cost. `journal=False` turns off every journal export. `paidwen.close()` sends what remains and removes the captures.

## When Paidwen does not answer

When Paidwen does not answer within 3 seconds, or answers with a server error, the SDK applies `on_unavailable`: `"block"` by default, or `"allow"`.

The SDK is open source under the Apache 2.0 license.

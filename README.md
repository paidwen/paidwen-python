# Paidwen SDK for Python

Paidwen controls what AI agents do. Before each action, the agent asks Paidwen whether it may act, then applies the answer. The agent keeps its own tool keys and performs the action itself.

```python
from paidwen import Paidwen

paidwen = Paidwen()


@paidwen.tool("payments.refund")
def refund(order: str, amount: float) -> None:
    payments.refund(order, amount)
```

`Paidwen()` reads the agent key from `PAIDWEN_API_KEY`. A call to `refund` asks Paidwen first. It runs when Paidwen allows it, raises `DeniedError` when Paidwen blocks it or puts the agent in quarantine, and raises `PendingError` when the action waits for a human answer. When the permissions of the agent refuse the action, the message of `DeniedError` says so.

## MCP tool calls

```python
from mcp import ClientSession

paidwen.mcp(ClientSession)
```

`paidwen.mcp(ClientSession)` asks Paidwen before each `tools/call` request of every session of the official MCP SDK in the process, including the sessions that an agent framework creates. The question carries the tool name, its arguments and the name that the MCP server announces. A refused call raises `DeniedError` or `PendingError` and never reaches the server.

`paidwen.mcp(session)` captures one session only, for a process that runs several agents with their own keys. `mcp()` returns a function that removes the capture.

## When Paidwen does not answer

When Paidwen does not answer within 3 seconds, or answers with a server error, the SDK applies `on_unavailable`: `"block"` by default, or `"allow"`.

The SDK is open source under the Apache 2.0 license.

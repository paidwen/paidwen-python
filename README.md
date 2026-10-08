# Paidwen SDK for Python

Paidwen controls what AI agents do. Before each action, the agent asks Paidwen whether it may act, then applies the answer. The agent keeps its own tool keys and performs the action itself.

## The agent installs Paidwen in two lines

```sh
pip install paidwen
```

```python
from paidwen import Paidwen

paidwen = Paidwen()
paidwen.http()
```

`Paidwen()` reads the agent key from the `PAIDWEN_API_KEY` environment variable. Its first argument, `api_key`, passes the key in the code instead. The SDK runs on Python 3.10 or later.

`paidwen.http()` asks Paidwen before each outgoing HTTP call that httpx, requests, urllib3 or aiohttp makes in the process. Call it once when the process starts. The capture does not see the clients that open their own connections, such as `urllib.request`, `http.client`, pycurl and gRPC.

The tool name of an HTTP call is its method and its path, for example `POST /v1/refunds`, and its server is the host. The parameters carry `method`, `url`, `query`, and `body` when the body is JSON or a form of at most 64 KiB. The SDK never sends the headers, and replaces the value of each sensitive key, such as `password`, `token` or `api_key`, by `REDACTED`.

A refused call raises `DeniedError` and never leaves the process. The SDK asks nothing for its own calls, for the calls made inside an action already decided, and for the hosts that match the `ignore` option, such as `paidwen.http(ignore=["*.internal.example.com"])`. A process has one HTTP capture, and `http()` returns a function that removes it.

## An agent on Kubernetes, AWS or GitHub Actions needs no key

```python
paidwen = Paidwen(identity_token_file="/var/run/secrets/paidwen/token")
```

When the platform of the agent signs a token for it, the SDK exchanges this token for an agent token valid for 15 minutes, under a trust rule of your organization, and renews it before it expires. `identity_token_file` reads the token from a file at each exchange, such as the projected token of a Kubernetes service account. `identity_token` takes a function that returns the token, such as a call to `get_web_identity_token` on AWS. Without option, the SDK reads `PAIDWEN_IDENTITY_TOKEN_FILE`. The guide is at https://paidwen.com/docs/workload-identity.

## A wrapped function asks Paidwen before it runs

```python
@paidwen.tool("payments.refund")
def refund(order: str, amount: float) -> None:
    payments.refund(order, amount)


@paidwen.tool("search.web", cost={"estimated_usd": 0.001})
async def search(query: str) -> list[str]: ...


refund("ord_42", 2500)
```

A call to `refund` asks Paidwen first, with the tool name and the arguments of the call as parameters. It runs when Paidwen allows it, waits when the action needs a human approval, and raises `DeniedError` when Paidwen refuses it. An async function works the same way.

The `cost` option gives the cost that the agent estimates for the tool, as a fixed value or as a function that receives the parameters as a dictionary. The budget of the agent counts it, and the rules read it in `cost.estimated_usd`.

## The MCP capture asks Paidwen before each tool call

```python
from mcp import ClientSession

paidwen.mcp(ClientSession)
```

`paidwen.mcp(ClientSession)` asks Paidwen before each `tools/call` request of every session of the official MCP SDK in the process, including the sessions that an agent framework creates. The question carries the tool name, its arguments and the name that the MCP server announces. A refused call raises `DeniedError` and never reaches the server.

`paidwen.mcp(session)` captures one session only, for a process that runs several agents with their own keys. `mcp()` returns a function that removes the capture.

## An action can wait for a human approval

```python
paidwen = Paidwen(on_wait=lambda wait: notify_user(wait.approval_id, wait.expires_at))
```

When a rule asks for a human approval, the call waits for the answer of the approver. The SDK calls `on_wait` once when the wait starts, so the agent can tell its user. In an async function, `on_wait` can also be a coroutine function. The call runs once the approver approves it, and raises `DeniedError` when the approver denies it. When the wait expires, the SDK applies the `on_expiry` choice of the rule: `deny` raises `DeniedError`, and `allow` runs the call.

With `on_pending="raise"`, the call raises `PendingError` as soon as the wait is open, and the action does not run. The agent calls again later with the same action identifier, found in `error.decision["action_id"]`. The action runs if the approver has approved it:

```python
with paidwen.context(action_id=action_id):
    refund("ord_42", 2500)
```

`paidwen.context(action_id=..., session_id=...)` gives its action identifier to the first decision of the block, and its session to every decision and journal line of the block.

## The SDK applies your choice when Paidwen does not answer

The SDK waits 3 seconds at most for each step of the exchange with Paidwen, and the `timeout` option changes this delay, in seconds. Without an answer in time, on a network error or on a server error, the SDK applies `on_unavailable`: `"block"` by default, or `"allow"`. It writes a warning with the cause to the `paidwen` logger. With `"block"`, the call raises `DeniedError`.

When the organization has used all the events its plan allows this month, Paidwen answers 503 with the problem type `quota-exceeded` until the first day of the next month in UTC. The SDK applies `on_unavailable` in the same way, and its warning names the cause with the detail of the problem.

## The journal receives each tool call and each model call

Each tool call goes to the journal of Paidwen as an OpenTelemetry span, with its decision, its start and its result. The span is a child of the active OpenTelemetry span of the agent.

To journal the model calls that your OpenTelemetry instrumentations describe, add the span processor of Paidwen to your tracer provider:

```python
provider.add_span_processor(paidwen.span_processor())
```

A model call leaves as soon as it ends, and the next question of the agent waits for it, one second at most, so that the budget sees its cost. The `journal=False` option turns off every journal export. `paidwen.flush()` sends the waiting spans, and `paidwen.close()` sends what remains and removes the HTTP capture. `with Paidwen() as paidwen:` closes the client at the end of the block.

## Three errors stop an action

- `DeniedError` says that the action does not run: a rule, the permissions or the budget of the agent refused it, the approver denied it, the wait expired with `on_expiry` at `deny`, the agent is in quarantine, or Paidwen did not answer and `on_unavailable` is `"block"`. Its message says why. `error.decision` carries the decision, `error.approval` the state of the wait, and `error.incident` the incident that put the agent in quarantine, with its `id` and its `kind`.
- `PendingError` says that the action waits for a human approval and that the agent chose `on_pending="raise"`. `error.decision` carries the decision and its `approval`.
- `RequestError` says that Paidwen refused the request itself, with an HTTP status from 400 to 499. A disabled agent receives 403, and the message then reads `Paidwen refused the agent: The agent is disabled.` with the detail of the problem. An unknown key receives 401. `error.status` carries the status, and `error.problem` the problem in the format of RFC 9457. The action never runs, even with `on_unavailable="allow"`, and the SDK does not ask again.

The three errors extend `PaidwenError`, which the constructor also raises for a missing key or a wrong option.

The SDK is open source under the Apache 2.0 license. The documentation lives at https://paidwen.com/docs.

# Paidwen SDK for Python

Paidwen controls what AI agents do. Before each action, the agent asks Paidwen whether it may act, then applies the answer. The agent keeps its own tool keys and performs the action itself.

```python
from paidwen import Paidwen

paidwen = Paidwen()


@paidwen.tool("payments.refund")
def refund(order: str, amount: float) -> None:
    payments.refund(order, amount)
```

`Paidwen()` reads the agent key from `PAIDWEN_API_KEY`. A call to `refund` asks Paidwen first. It runs when Paidwen allows it, raises `DeniedError` when Paidwen blocks it or puts the agent in quarantine, and raises `PendingError` when the action waits for a human answer.

When Paidwen does not answer within 3 seconds, or answers with a server error, the SDK applies `on_unavailable`: `"block"` by default, or `"allow"`.

The SDK is open source under the Apache 2.0 license.

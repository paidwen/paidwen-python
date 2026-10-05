from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from typing import Any

import httpx
from opentelemetry.sdk.trace import TracerProvider

from paidwen import DeniedError, Paidwen, Wait

MODEL = "gpt-4o-mini"


def now() -> float:
    return time.time() * 1000


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def fake_model(provider: TracerProvider, turn: int) -> str:
    span = provider.get_tracer("reference-agent").start_span(
        f"chat {MODEL}",
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": MODEL,
        },
    )
    span.set_attributes(
        {
            "gen_ai.response.model": MODEL,
            "gen_ai.usage.input_tokens": 10_000,
            "gen_ai.usage.output_tokens": 1_000,
        }
    )
    span.end()
    return f"search the web for turn {turn}"


def refund() -> None:
    departed: list[float] = []

    def waiting(wait: Wait) -> None:
        emit(
            {
                "event": "waiting",
                "approval_id": wait.approval_id,
                "interval": wait.interval,
                "at": now(),
            }
        )

    paidwen = Paidwen(on_wait=waiting)

    @paidwen.tool("payments.refund", cost={"estimated_usd": 0.3})
    def payments(order: str, amount: int, reason: str) -> dict[str, str]:
        departed.append(now())
        return {"refund": f"re_{order}"}

    action_id = str(uuid.uuid4())
    emit({"event": "asking", "action_id": action_id, "at": now()})
    try:
        with paidwen.context(action_id=action_id, session_id=f"proof-{action_id}"):
            result = payments("ord_4821", 2500, "lost parcel")
        emit(
            {"event": "done", "action_id": action_id, "departed_at": departed[0], "result": result}
        )
    except DeniedError as error:
        emit({"event": "denied", "action_id": action_id, "message": str(error)})
    paidwen.close()


async def loop(parallel: int) -> None:
    paidwen = Paidwen()
    provider = TracerProvider()
    provider.add_span_processor(paidwen.span_processor())

    @paidwen.tool("search.web", cost={"estimated_usd": 0.001})
    async def search(query: str) -> list[str]:
        return [query]

    async def run(worker: int) -> dict[str, Any]:
        turns = 0
        while True:
            query = fake_model(provider, turns)
            try:
                await search(query)
                turns += 1
            except DeniedError as error:
                return {"worker": worker, "turns": turns, "budget": error.decision.get("budget")}

    workers = await asyncio.gather(*(run(worker) for worker in range(parallel)))
    emit({"event": "stopped", "workers": list(workers)})
    provider.shutdown()
    paidwen.close()


def http_calls(target: str, calls: int, warmup: int, captured: bool) -> None:
    paidwen = Paidwen() if captured else None
    release = paidwen.http() if paidwen else None
    body = {"amount": 2500, "currency": "usd", "order": "ord_4821"}
    client = httpx.Client()

    def series(count: int) -> list[float]:
        durations: list[float] = []
        for _ in range(count):
            started = time.perf_counter()
            client.post(f"{target}/v1/refunds", json=body).read()
            durations.append((time.perf_counter() - started) * 1000)
        return durations

    series(warmup)
    durations = series(calls)
    client.close()
    if release is not None and paidwen is not None:
        release()
        paidwen.close()
    emit({"event": "measured", "durations": durations})


def main(arguments: list[str]) -> None:
    mode = arguments[0] if arguments else ""
    if mode == "refund":
        refund()
    elif mode == "loop":
        asyncio.run(loop(int(arguments[1]) if len(arguments) > 1 else 1))
    elif mode == "http" and len(arguments) > 1:
        calls = int(arguments[2]) if len(arguments) > 2 else 1000
        captured = len(arguments) > 3 and arguments[3] == "captured"
        http_calls(arguments[1], calls, 100, captured)
    else:
        raise SystemExit(
            "usage: python proof/agent.py refund | loop [parallel]"
            " | http <url> [calls] [plain|captured]"
        )


if __name__ == "__main__":
    main(sys.argv[1:])

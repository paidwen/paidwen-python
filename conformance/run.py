import json
import sys
import time
from typing import Any

from paidwen import Paidwen, RequestError


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as file:
        plan = json.load(file)
    results: list[dict[str, Any]] = []
    for step in plan["steps"]:
        options: dict[str, Any] = {"base_url": plan["base_url"]}
        if "timeout_ms" in step:
            options["timeout"] = step["timeout_ms"] / 1000
        if "on_unavailable" in step:
            options["on_unavailable"] = step["on_unavailable"]
        if "api_key" in step:
            options["api_key"] = step["api_key"]
        if "identity_token_file" in step:
            options["identity_token_file"] = step["identity_token_file"]
        with Paidwen(**options) as paidwen:
            started = time.perf_counter()
            try:
                decision = paidwen.decide(**step["request"])
                results.append({"name": step["name"], "decision": decision})
            except RequestError as error:
                problem = error.problem or {}
                results.append(
                    {
                        "name": step["name"],
                        "error": {"status": error.status, "type": problem.get("type")},
                    }
                )
            results[-1]["elapsed_ms"] = (time.perf_counter() - started) * 1000
    sys.stdout.write(json.dumps({"results": results}) + "\n")


if __name__ == "__main__":
    main()

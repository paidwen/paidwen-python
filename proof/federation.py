import json
import os
import sys
import time

import httpx

from paidwen import Paidwen


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set.")
    return value


AUDIENCE = required("PAIDWEN_AUDIENCE")


def github_token() -> str:
    response = httpx.get(
        required("ACTIONS_ID_TOKEN_REQUEST_URL"),
        params={"audience": AUDIENCE},
        headers={"authorization": f"Bearer {required('ACTIONS_ID_TOKEN_REQUEST_TOKEN')}"},
    )
    response.raise_for_status()
    return str(response.json()["value"])


paidwen = Paidwen(base_url=required("PAIDWEN_BASE_URL"), identity_token=github_token)
started = time.perf_counter()
decision = paidwen.decide(
    tool={"name": "release.publish", "kind": "function"},
    params={
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "run": os.environ.get("GITHUB_RUN_ID"),
    },
)
elapsed = round((time.perf_counter() - started) * 1000)


@paidwen.tool("release.publish")
def release(version: str) -> str:
    return version


release("0.0.0")
paidwen.close()
print(json.dumps({"language": "Python", "verdict": decision["verdict"], "elapsed_ms": elapsed}))
if decision["verdict"] != "allow":
    sys.exit(1)

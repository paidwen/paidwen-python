import json
import sys

from paidwen import Paidwen


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as file:
        plan = json.load(file)
    results = []
    for entry in plan["cases"]:
        with Paidwen(entry["api_key"], base_url=plan["base_url"]) as paidwen:
            results.append({"name": entry["name"], "decision": paidwen.decide(**entry["request"])})
    sys.stdout.write(json.dumps({"results": results}) + "\n")


if __name__ == "__main__":
    main()

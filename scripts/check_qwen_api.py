"""Make a small Qwen request and validate the SayCan scoring contract."""
from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saycan import Belief, Scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="Pick up the red cube and place it near the blue cube",
    )
    args = parser.parse_args()

    belief = Belief()
    belief.objects = {
        "red_cube": {"position": [0.43, -0.16, 0.385], "confidence": 1.0},
        "blue_cube": {"position": [0.55, 0.16, 0.385], "confidence": 1.0},
    }
    calls = [
        {"id": "PickUp:0", "name": "PickUp", "args": {"target": "red_cube"}},
        {"id": "PickUp:1", "name": "PickUp", "args": {"target": "blue_cube"}},
        {"id": "Done:2", "name": "Done", "args": {}},
    ]
    scorer = Scorer("qwen")
    scores = scorer.score(args.task, calls, belief)
    report = {
        "success": True,
        "model": scorer.model,
        "base_url": scorer.base_url,
        "latency_s": scorer.last_latency_s,
        "scores": scores,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nQWEN API CHECK PASSED (the API key was not printed)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"QWEN API CHECK FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


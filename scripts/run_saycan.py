from pathlib import Path
import sys

# Allow direct execution as: python scripts/run_saycan.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from config import Config
from mujoco_env import build_model, Runtime
from context import Context
from saycan import SayCan, Scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=["qwen", "keyword"], default="qwen")
    parser.add_argument(
        "--task",
        default="Pick up the red cube and place it near the blue cube",
    )
    args = parser.parse_args()

    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    context = Context(runtime, args.task)
    try:
        context.robot.home()
        context.observe()
        result = SayCan(context, Scorer(args.scorer)).run(args.task)
        cfg.output.mkdir(exist_ok=True, parents=True)
        (cfg.output / "last_run.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["success"]:
            raise SystemExit(1)
    finally:
        runtime.r.close()


if __name__ == "__main__":
    main()

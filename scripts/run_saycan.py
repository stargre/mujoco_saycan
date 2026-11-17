from pathlib import Path
import sys

# Allow direct execution as: python scripts/run_saycan.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
import platform

from config import Config
from mujoco_env import build_model, Runtime
from context import Context
from saycan import SayCan, Scorer
from visualization import RunVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Run closed-loop SayCan planning in the MuJoCo Panda scene."
    )
    parser.add_argument("--scorer", choices=["qwen", "keyword"], default="keyword")
    parser.add_argument(
        "--task",
        default="Pick up the red cube and place it near the blue cube",
    )
    parser.add_argument(
        "--visualize",
        choices=["record", "viewer", "both", "none"],
        default="record",
        help="record GIF/HTML, open the native viewer, do both, or disable visualization",
    )
    parser.add_argument("--output", default="outputs/saycan_run")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--cameras",
        default="overhead",
        help="comma-separated recording cameras: overhead,side_view",
    )
    parser.add_argument("--max-rounds", type=int, default=12)
    args = parser.parse_args()

    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    context = Context(runtime, args.task)
    output = Path(args.output)
    recorder = None
    try:
        if args.visualize in ("viewer", "both"):
            runtime.enable_viewer()
        if args.visualize in ("record", "both"):
            cameras = tuple(
                item.strip() for item in args.cameras.split(",") if item.strip()
            )
            invalid = set(cameras) - {"overhead", "side_view"}
            if invalid:
                raise ValueError(f"Unknown cameras: {sorted(invalid)}")
            recorder = RunVisualizer(
                runtime, output, args.task, fps=args.fps, cameras=cameras
            )
            context.on_stage = recorder.on_stage
            context.on_plan = recorder.on_plan

        context.robot.home()
        context.observe()
        scorer = Scorer(args.scorer)
        result = SayCan(context, scorer).run(args.task, max_rounds=args.max_rounds)
        result["run_metadata"] = {
            "task": args.task,
            "scorer": args.scorer,
            "qwen_model": scorer.model if args.scorer == "qwen" else None,
            "qwen_base_url": scorer.base_url if args.scorer == "qwen" else None,
            "qwen_request_count": scorer.request_count,
            "qwen_last_latency_s": scorer.last_latency_s,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco_gl": os.getenv("MUJOCO_GL"),
        }
        if recorder is not None:
            result["visualization"] = recorder.finalize(result)
        output.mkdir(exist_ok=True, parents=True)
        (output / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nArtifacts: {output.resolve()}")
        if not result["success"]:
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

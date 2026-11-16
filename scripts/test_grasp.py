"""Run one physical PickUp skill and export diagnostic PNG/GIF artifacts.

Run from the repository root:
  python scripts/test_grasp.py --target red_cube
"""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import mujoco
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from mujoco_env import build_model, Runtime
from context import Context


def annotate(frame, label):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 27), fill=(0, 0, 0))
    draw.text((9, 7), label, fill=(255, 255, 255))
    return image


def save_gif(frames, path, duration_ms=650):
    if not frames:
        return
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["red_cube", "blue_cube"], default="red_cube")
    parser.add_argument("--output", default="outputs/grasp_test")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    frames = {"overhead": [], "side_view": []}
    stage_reports = []
    try:
        context = Context(runtime, f"pick up the {args.target.replace('_', ' ')}")
        robot = context.robot
        robot.home()
        context.observe()

        body_id = mujoco.mj_name2id(
            runtime.m, mujoco.mjtObj.mjOBJ_BODY, args.target
        )
        if body_id < 0:
            raise RuntimeError(f"Missing object body: {args.target}")
        initial_object_position = runtime.d.xpos[body_id].copy()

        def capture(stage):
            stage_reports.append({
                "stage": stage,
                "simulation_time_s": float(runtime.d.time),
                "tcp_position_world": robot.tcp_pos().round(6).tolist(),
                "object_position_world": runtime.d.xpos[body_id].round(6).tolist(),
            })
            for camera in ("overhead", "side_view"):
                frame = annotate(
                    runtime.render(camera=camera),
                    f"{args.target} | {stage} | t={runtime.d.time:.2f}s",
                )
                frames[camera].append(frame)
                frame.save(output / f"{len(stage_reports):02d}_{stage}_{camera}.png")

        context.on_stage = capture
        capture("home")
        result = context.execute({
            "name": "PickUp",
            "args": {"target": args.target},
        })
        capture("skill_result")

        final_object_position = runtime.d.xpos[body_id].copy()
        lift_delta = float(final_object_position[2] - initial_object_position[2])
        report = {
            "target": args.target,
            "success": bool(result["success"]),
            "status": result["status"],
            "result": result,
            "initial_object_position_world": initial_object_position.tolist(),
            "final_object_position_world": final_object_position.tolist(),
            "simulator_object_lift_m": lift_delta,
            "final_tcp_position_world": robot.tcp_pos().tolist(),
            "final_joint_position_rad": robot.q().tolist(),
            "stage_trace": stage_reports,
            "artifacts": [
                "grasp_overhead.gif",
                "grasp_side_view.gif",
            ],
        }
        (output / "grasp_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        save_gif(frames["overhead"], output / "grasp_overhead.gif")
        save_gif(frames["side_view"], output / "grasp_side_view.gif")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nGrasp artifacts saved to: {output.resolve()}")

        if not result["success"]:
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

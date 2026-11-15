"""Export headless MuJoCo visualization artifacts (PNG + GIF).

Run with MUJOCO_GL=egl on a headless Linux server:
  python scripts/visualize_scene.py

Outputs are written to outputs/visualization/.
The motion animation is a real simulated joint-space trajectory from home to
the camera-estimated pre-grasp waypoint; it is not a teleport animation.
"""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from mujoco_env import build_model, Runtime
from context import Context


def render_frame(runtime, camera, label):
    frame = Image.fromarray(runtime.render(camera=camera))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 28), fill=(0, 0, 0))
    draw.text((10, 7), label, fill=(255, 255, 255))
    return frame


def save_gif(frames, path, fps):
    if not frames:
        raise RuntimeError(f"No frames generated for {path}")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, int(1000 / fps)),
        loop=0,
        optimize=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/visualization")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    try:
        context = Context(runtime, "")
        robot = context.robot
        robot.home()
        belief = context.observe()

        # Save static overview images at the home pose.
        for camera in ("overhead", "side_view"):
            frame = render_frame(runtime, camera, f"Franka Panda | HOME | {camera}")
            frame.save(out / f"home_{camera}.png")

        if "red_cube" not in belief.objects:
            raise RuntimeError("RGB-D camera did not localize red_cube")

        cube = np.asarray(belief.objects["red_cube"]["position"], dtype=float)
        target = cube + np.array([0.0, 0.0, 0.15])
        q_start = robot.q()
        q_target = robot.ik(target)

        # Render synchronized overhead and side-view frames while the simulator
        # follows a quintic joint-space trajectory.
        overhead_frames = []
        side_frames = []
        n_steps = max(2, int(args.duration / runtime.m.opt.timestep))
        stride = max(1, n_steps // max(1, args.fps * args.duration))
        for step in range(n_steps):
            tau = (step + 1) / n_steps
            blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            runtime.d.ctrl[robot.ai] = q_start + blend * (q_target - q_start)
            runtime.step()
            if step % stride == 0 or step == n_steps - 1:
                label = f"HOME -> PRE-GRASP | t={runtime.d.time:.2f}s"
                overhead_frames.append(render_frame(runtime, "overhead", label))
                side_frames.append(render_frame(runtime, "side_view", label))

        # Hold the final command briefly so the GIF shows the achieved endpoint.
        for _ in range(max(1, int(0.5 / runtime.m.opt.timestep))):
            runtime.step()
        for camera, frames in (
            ("overhead", overhead_frames),
            ("side_view", side_frames),
        ):
            frames.append(
                render_frame(runtime, camera, f"PRE-GRASP HOLD | {camera}")
            )
            save_gif(frames, out / f"reach_{camera}.gif", args.fps)

        for camera in ("overhead", "side_view"):
            render_frame(
                runtime, camera, f"FINAL SIM STATE | {camera}"
            ).save(out / f"final_{camera}.png")

        result = {
            "target_hand_position_world": target.tolist(),
            "ik_joint_target_rad": q_target.tolist(),
            "actual_joint_position_rad": robot.q().tolist(),
            "actual_hand_position_world": robot.pos().tolist(),
            "cartesian_endpoint_error_m": float(
                np.linalg.norm(robot.pos() - target)
            ),
            "simulation_time_s": float(runtime.d.time),
            "artifacts": [
                "home_overhead.png",
                "home_side_view.png",
                "reach_overhead.gif",
                "reach_side_view.gif",
                "final_overhead.png",
                "final_side_view.png",
            ],
        }
        (out / "visualization_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        print(f"\nVisualization files saved to: {out.resolve()}")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

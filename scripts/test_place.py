"""Run PickUp -> PlaceNear -> Done and export physical diagnostics.

Run from the repository root:
  python scripts/test_place.py
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


def save_gif(frames, path, duration_ms=600):
    if frames:
        frames[0].save(
            path, save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0, optimize=True,
        )


def geom_contacting_body_names(model, data, target_geom_id):
    names = set()
    pairs = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if target_geom_id not in (g1, g2):
            continue
        other = g2 if g1 == target_geom_id else g1
        body_id = int(model.geom_bodyid[other])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        names.add(body_name)
        pairs.append({
            "other_body": body_name,
            "distance_m": float(contact.dist),
            "position_world": np.asarray(contact.pos).round(6).tolist(),
        })
    return names, pairs


def object_table_contact(model, data, object_geom_id):
    table_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "table"
    )
    return any(
        {int(data.contact[i].geom1), int(data.contact[i].geom2)}
        == {int(table_id), int(object_geom_id)}
        for i in range(data.ncon)
    )


def robot_table_contacts(model, data):
    table_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "table"
    )
    found = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if table_id not in (g1, g2):
            continue
        other = g2 if g1 == table_id else g1
        body_id = int(model.geom_bodyid[other])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        if body_name not in ("red_cube", "blue_cube"):
            found.append({
                "body": body_name,
                "distance_m": float(contact.dist),
                "position_world": np.asarray(contact.pos).round(6).tolist(),
            })
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/place_test")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    frames = {"overhead": [], "side_view": []}
    stage_trace = []

    try:
        task = "Pick up the red cube and place it near the blue cube"
        context = Context(runtime, task)
        robot = context.robot
        robot.home()
        context.observe()

        model, data = runtime.m, runtime.d
        red_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "red_cube"
        )
        blue_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube"
        )
        red_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom"
        )
        table_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table"
        )
        if min(red_body, blue_body, red_geom, table_geom) < 0:
            raise RuntimeError("Required red_cube, blue_cube, or table geometry is missing")

        initial_red = data.xpos[red_body].copy()
        initial_blue = data.xpos[blue_body].copy()
        table_top = float(
            model.geom_pos[table_geom][2] + model.geom_size[table_geom][2]
        )
        red_half_height = float(model.geom_size[red_geom][2])

        def capture(stage):
            stage_trace.append({
                "stage": stage,
                "simulation_time_s": float(data.time),
                "tcp_position_world": robot.tcp_pos().round(6).tolist(),
                "red_cube_position_world": data.xpos[red_body].round(6).tolist(),
                "blue_cube_position_world": data.xpos[blue_body].round(6).tolist(),
                "robot_table_contacts": robot_table_contacts(model, data),
            })
            for camera in ("overhead", "side_view"):
                frame = annotate(
                    runtime.render(camera=camera),
                    f"PlaceNear | {stage} | t={data.time:.2f}s",
                )
                frames[camera].append(frame)
                frame.save(
                    output / f"{len(stage_trace):02d}_{stage}_{camera}.png"
                )

        context.on_stage = capture
        capture("home")

        pickup = context.execute({
            "name": "PickUp",
            "args": {"target": "red_cube"},
        })
        if pickup["success"]:
            capture("pickup_complete")
            place = context.execute({
                "name": "PlaceNear",
                "args": {
                    "object": "red_cube",
                    "target": "blue_cube",
                    "relation": "near",
                },
            })
        else:
            place = {
                "skill": "PlaceNear",
                "success": False,
                "status": "skipped_pickup_failed",
                "reason": "PlaceNear was not attempted because PickUp failed",
            }

        if place["success"]:
            done = context.execute({"name": "Done", "args": {}})
        else:
            done = {
                "skill": "Done",
                "success": False,
                "status": "skipped_place_failed",
                "reason": "Done was not attempted because PlaceNear failed",
            }
        capture("task_end")

        before_settle = data.xpos[red_body].copy()
        for _ in range(max(1, int(0.5 / model.opt.timestep))):
            runtime.step()
        final_red = data.xpos[red_body].copy()
        final_blue = data.xpos[blue_body].copy()

        finger_bodies, finger_pairs = geom_contacting_body_names(
            model, data, red_geom
        )
        table_contact = object_table_contact(model, data, red_geom)
        robot_contacts = robot_table_contacts(model, data)
        true_target_position = final_blue + np.array([0.075, 0.0, 0.0])
        true_target_position[2] = table_top + red_half_height
        goal_xy_error = float(
            np.linalg.norm(final_red[:2] - true_target_position[:2])
        )
        target_distance = float(
            np.linalg.norm(final_red[:2] - final_blue[:2])
        )
        height_error = float(abs(final_red[2] - true_target_position[2]))
        settle_displacement = float(np.linalg.norm(final_red - before_settle))

        physical_checks = {
            "both_skills_succeeded": bool(pickup["success"] and place["success"]),
            "done_succeeded": bool(done["success"]),
            "red_cube_on_table": bool(table_contact),
            "red_cube_not_in_finger_contact": not bool(
                {"left_finger", "right_finger"} & finger_bodies
            ),
            "robot_clear_of_table": not bool(robot_contacts),
            "target_xy_error_m": goal_xy_error,
            "target_distance_m": target_distance,
            "height_error_m": height_error,
            "settle_displacement_m": settle_displacement,
        }
        evaluation_pass = (
            physical_checks["both_skills_succeeded"]
            and physical_checks["done_succeeded"]
            and physical_checks["red_cube_on_table"]
            and physical_checks["red_cube_not_in_finger_contact"]
            and physical_checks["robot_clear_of_table"]
            and goal_xy_error <= 0.04
            and 0.05 <= target_distance <= 0.13
            and height_error <= 0.01
            and settle_displacement <= 0.01
        )

        report = {
            "success": bool(evaluation_pass),
            "status": "place_near_verified" if evaluation_pass else "place_near_not_verified",
            "pickup_result": pickup,
            "place_result": place,
            "done_result": done,
            "initial_red_cube_position_world": initial_red.tolist(),
            "initial_blue_cube_position_world": initial_blue.tolist(),
            "final_red_cube_position_world": final_red.tolist(),
            "final_blue_cube_position_world": final_blue.tolist(),
            "simulator_evaluation_target_position_world": true_target_position.tolist(),
            "physical_checks": physical_checks,
            "final_finger_contact_bodies": sorted(finger_bodies),
            "final_finger_contact_pairs": finger_pairs,
            "final_robot_table_contacts": robot_contacts,
            "stage_trace": stage_trace,
            "artifacts": [
                "place_overhead.gif",
                "place_side_view.gif",
            ],
        }
        (output / "place_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        save_gif(frames["overhead"], output / "place_overhead.gif")
        save_gif(frames["side_view"], output / "place_side_view.gif")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nPlaceNear artifacts saved to: {output.resolve()}")

        if not evaluation_pass:
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

"""Diagnose Panda IK endpoint tracking, collision contacts and gripper actuation.

Run:
  python scripts/diagnose_manipulation.py
This is an evaluation/debug utility; simulator ground truth is used only for
diagnostics, never by the SayCan planner.
"""
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from mujoco_env import build_model, Runtime
from context import Context


def name(model, obj_type, idx):
    value = mujoco.mj_id2name(model, obj_type, int(idx))
    return value if value is not None else f"<unnamed:{idx}>"


def contact_report(model, data):
    rows = []
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        n1 = name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        n2 = name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        b1 = name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1])
        b2 = name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2])
        text = f"{n1} [{b1}] <-> {n2} [{b2}]"
        if any(k in text.lower() for k in ("finger", "cube", "table", "hand")):
            rows.append(text)
    return rows


def main():
    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    try:
        ctx = Context(runtime, "")
        robot = ctx.robot
        robot.home()
        belief = ctx.observe()

        print("\n=== Model / actuator inventory ===")
        for joint_name in [f"joint{i}" for i in range(1, 8)] + [
            "finger_joint1", "finger_joint2"
        ]:
            jid = mujoco.mj_name2id(
                runtime.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            adr = runtime.m.jnt_qposadr[jid]
            print(f"{joint_name}: qpos={runtime.d.qpos[adr]:.5f}")
        for actuator_name in [f"actuator{i}" for i in range(1, 9)]:
            aid = mujoco.mj_name2id(
                runtime.m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            print(
                f"{actuator_name}: ctrl={runtime.d.ctrl[aid]:.5f}, "
                f"ctrlrange={runtime.m.actuator_ctrlrange[aid].tolist()}"
            )

        if "red_cube" not in belief.objects:
            raise RuntimeError("Camera did not detect red_cube")

        estimated = np.asarray(
            belief.objects["red_cube"]["position"], dtype=float
        )
        target = estimated + np.array([0.0, 0.0, 0.15])
        q_before = robot.q()
        q_solution = robot.ik(target)

        # Independently evaluate the FK of the IK solution without leaving a
        # changed simulation state behind.
        saved_q = robot.q()
        runtime.d.qpos[robot.qa] = q_solution
        mujoco.mj_forward(runtime.m, runtime.d)
        fk_at_solution = robot.pos().copy()
        fk_error = float(np.linalg.norm(fk_at_solution - target))
        runtime.d.qpos[robot.qa] = saved_q
        mujoco.mj_forward(runtime.m, runtime.d)

        print("\n=== IK solution forward-kinematics check ===")
        print("Estimated red cube:", estimated.round(5).tolist())
        print("Hand target:", target.round(5).tolist())
        print("FK at IK solution:", fk_at_solution.round(5).tolist())
        print("IK FK residual:", f"{fk_error:.5f} m")
        print("Start joints:", q_before.round(5).tolist())
        print("IK joints:", q_solution.round(5).tolist())

        print("\n=== Execute 3-second joint trajectory ===")
        robot.joints(q_solution, dur=3.0)
        q_after_trajectory = robot.q()
        ee_after_trajectory = robot.pos().copy()
        print("Actual joints:", q_after_trajectory.round(5).tolist())
        print("Joint tracking error:", (q_after_trajectory - q_solution).round(5).tolist())
        print(
            "Joint tracking error norm:",
            f"{np.linalg.norm(q_after_trajectory - q_solution):.5f} rad"
        )
        print("EE after trajectory:", ee_after_trajectory.round(5).tolist())
        print(
            "Cartesian endpoint error:",
            f"{np.linalg.norm(ee_after_trajectory - target):.5f} m"
        )

        print("\n=== Hold final setpoint for 1.5 seconds ===")
        for _ in range(max(1, int(1.5 / runtime.m.opt.timestep))):
            runtime.step()
        q_after_hold = robot.q()
        ee_after_hold = robot.pos().copy()
        print("Actual joints after hold:", q_after_hold.round(5).tolist())
        print(
            "Joint tracking error after hold:",
            f"{np.linalg.norm(q_after_hold - q_solution):.5f} rad"
        )
        print("EE after hold:", ee_after_hold.round(5).tolist())
        print(
            "Cartesian endpoint error after hold:",
            f"{np.linalg.norm(ee_after_hold - target):.5f} m"
        )

        print("\n=== Robot / table / cube contact pairs ===")
        rows = contact_report(runtime.m, runtime.d)
        if rows:
            for row in rows:
                print("-", row)
        else:
            print("No matching contacts in current contact manifold")

        print("\n=== Finger world positions ===")
        for body_name in ("hand", "left_finger", "right_finger"):
            bid = mujoco.mj_name2id(
                runtime.m, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            print(f"{body_name}: {runtime.d.xpos[bid].round(5).tolist()}")

        print("\n=== Gripper open/close actuator check (no object grasp) ===")
        for opened in (False, True):
            robot.grip(opened, dur=1.0)
            values = []
            for joint_name in ("finger_joint1", "finger_joint2"):
                jid = mujoco.mj_name2id(
                    runtime.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                values.append(float(runtime.d.qpos[runtime.m.jnt_qposadr[jid]]))
            print(
                f"{'OPEN' if opened else 'CLOSE'}: "
                f"finger qpos={np.asarray(values).round(5).tolist()}, "
                f"actuator8 ctrl={runtime.d.ctrl[robot.g]:.3f}"
            )

        print("\nDIAGNOSTIC COMPLETE")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

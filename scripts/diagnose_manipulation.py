"""Diagnose Panda TCP tracking, table contacts, finger geometry and gripper."""
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


def object_name(model, obj_type, idx):
    value = mujoco.mj_id2name(model, obj_type, int(idx))
    return value if value is not None else f"<unnamed:{idx}>"


def table_contacts(model, data):
    rows = []
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if table_id not in (g1, g2):
            continue
        other = g2 if g1 == table_id else g1
        body_id = int(model.geom_bodyid[other])
        body = object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom = object_name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        if body in ("red_cube", "blue_cube"):
            continue
        rows.append({
            "body": body,
            "geom": geom,
            "contact_xyz": np.asarray(con.pos).round(5).tolist(),
            "distance_m": float(con.dist),
        })
    return rows


def print_table_contacts(model, data, stage):
    rows = table_contacts(model, data)
    print(f"\n=== Robot-table contacts: {stage} ({len(rows)}) ===")
    if not rows:
        print("No robot-table contacts detected.")
    for row in rows:
        print(
            f"{row['body']} / {row['geom']}: "
            f"point={row['contact_xyz']}, dist={row['distance_m']:.6f} m"
        )


def find_main_pad_geom(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    matches = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if np.allclose(model.geom_size[gid], [0.0085, 0.004, 0.0085], atol=1e-5):
            if np.allclose(model.geom_pos[gid], [0.0, 0.0055, 0.0445], atol=1e-5):
                matches.append(gid)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one main fingertip pad geom for {body_name}, found {matches}"
        )
    return matches[0]


def main():
    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    try:
        ctx = Context(runtime, "")
        robot = ctx.robot
        robot.home()
        belief = ctx.observe()

        table_id = mujoco.mj_name2id(
            runtime.m, mujoco.mjtObj.mjOBJ_GEOM, "table"
        )
        print("Table center:", runtime.d.geom_xpos[table_id].round(5).tolist())
        print("Table half-size:", runtime.m.geom_size[table_id].round(5).tolist())
        print_table_contacts(runtime.m, runtime.d, "after HOME")

        print("\n=== TCP geometry audit ===")
        left_pad = find_main_pad_geom(runtime.m, "left_finger")
        right_pad = find_main_pad_geom(runtime.m, "right_finger")
        left_pos = runtime.d.geom_xpos[left_pad].copy()
        right_pos = runtime.d.geom_xpos[right_pad].copy()
        pad_midpoint = 0.5 * (left_pos + right_pos)
        tcp = robot.tcp_pos()
        print("Hand origin:", robot.hand_pos().round(5).tolist())
        print("Left main pad center:", left_pos.round(5).tolist())
        print("Right main pad center:", right_pos.round(5).tolist())
        print("Pad midpoint:", pad_midpoint.round(5).tolist())
        print("Computed TCP:", tcp.round(5).tolist())
        print("TCP / pad-midpoint difference:",
              f"{np.linalg.norm(tcp - pad_midpoint):.6f} m")

        estimated = np.asarray(
            belief.objects["red_cube"]["position"], dtype=float
        )
        target = estimated + np.array([0.0, 0.0, 0.15])
        q_before = robot.q()
        q_solution = robot.ik(target)

        saved_q = robot.q()
        runtime.d.qpos[robot.qa] = q_solution
        mujoco.mj_forward(runtime.m, runtime.d)
        fk_tcp = robot.tcp_pos().copy()
        fk_error = float(np.linalg.norm(fk_tcp - target))
        runtime.d.qpos[robot.qa] = saved_q
        mujoco.mj_forward(runtime.m, runtime.d)

        print("\n=== TCP IK forward-kinematics check ===")
        print("Estimated red cube:", estimated.round(5).tolist())
        print("Target TCP:", target.round(5).tolist())
        print("TCP FK at IK solution:", fk_tcp.round(5).tolist())
        print("IK TCP residual:", f"{fk_error:.5f} m")
        print("Start joints:", q_before.round(5).tolist())
        print("IK joints:", q_solution.round(5).tolist())

        print("\n=== Execute 3-second TCP trajectory ===")
        robot.joints(q_solution, dur=3.0)
        q_after = robot.q()
        tcp_after = robot.tcp_pos().copy()
        print("Actual joints:", q_after.round(5).tolist())
        print("Joint tracking error:", (q_after - q_solution).round(5).tolist())
        print("Joint tracking error norm:",
              f"{np.linalg.norm(q_after - q_solution):.5f} rad")
        print("Hand origin after trajectory:", robot.hand_pos().round(5).tolist())
        print("TCP after trajectory:", tcp_after.round(5).tolist())
        print("TCP Cartesian endpoint error:",
              f"{np.linalg.norm(tcp_after - target):.5f} m")
        print_table_contacts(runtime.m, runtime.d, "after trajectory")

        print("\n=== Hold final setpoint for 1.5 seconds ===")
        for _ in range(max(1, int(1.5 / runtime.m.opt.timestep))):
            runtime.step()
        q_hold = robot.q()
        tcp_hold = robot.tcp_pos().copy()
        print("Actual joints after hold:", q_hold.round(5).tolist())
        print("Joint tracking error after hold:",
              f"{np.linalg.norm(q_hold - q_solution):.5f} rad")
        print("Hand origin after hold:", robot.hand_pos().round(5).tolist())
        print("TCP after hold:", tcp_hold.round(5).tolist())
        print("TCP endpoint error after hold:",
              f"{np.linalg.norm(tcp_hold - target):.5f} m")
        print_table_contacts(runtime.m, runtime.d, "after hold")

        print("\n=== Finger body world positions ===")
        for body_name in ("left_finger", "right_finger"):
            bid = mujoco.mj_name2id(
                runtime.m, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            print(f"{body_name}: {runtime.d.xpos[bid].round(5).tolist()}")

        print("\n=== Gripper open/close actuator check ===")
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
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

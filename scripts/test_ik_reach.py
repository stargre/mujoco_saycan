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


def main():
    cfg = Config()
    runtime = Runtime(build_model(cfg.resolve()), cfg.width, cfg.height)
    try:
        context = Context(runtime, "")
        robot = context.robot
        robot.home()
        belief = context.observe()

        print("\n=== Camera localization audit (ground truth is evaluation-only) ===")
        for name, estimate in belief.objects.items():
            geom_id = mujoco.mj_name2id(
                runtime.m, mujoco.mjtObj.mjOBJ_GEOM, name + "_geom"
            )
            truth = runtime.d.geom_xpos[geom_id].copy()
            estimated = np.asarray(estimate["position"], dtype=float)
            error = float(np.linalg.norm(estimated - truth))
            print(
                f"{name}: estimated={estimated.round(4).tolist()}, "
                f"truth={truth.round(4).tolist()}, error={error:.4f} m"
            )

        if "red_cube" not in belief.objects:
            raise RuntimeError("red_cube was not detected")

        target = (
            np.asarray(belief.objects["red_cube"]["position"], dtype=float)
            + np.array([0.0, 0.0, 0.15])
        )
        q_before = robot.q()
        ee_before = robot.pos().copy()
        q_solution = robot.ik(target)
        state_delta = float(np.linalg.norm(robot.q() - q_before))

        print("\n=== IK state-preservation audit ===")
        print("EE before:", ee_before.round(4).tolist())
        print("Pre-grasp target:", target.round(4).tolist())
        print("IK solution:", q_solution.round(4).tolist())
        print("Joint-state change caused by IK:", f"{state_delta:.9f} rad")
        if state_delta > 1e-8:
            raise RuntimeError("IK changed the live robot state; it must be a pure calculation")

        robot.move(target, dur=2.0)
        ee_after = robot.pos().copy()
        final_error = float(np.linalg.norm(ee_after - target))
        joint_delta = float(np.linalg.norm(robot.q() - q_before))

        print("\n=== Physical trajectory audit ===")
        print("EE after:", ee_after.round(4).tolist())
        print("Final Cartesian error:", f"{final_error:.4f} m")
        print("Joint displacement:", f"{joint_delta:.4f} rad")

        if joint_delta < 0.02:
            raise RuntimeError("No measurable physical joint motion was executed")
        if final_error > 0.04:
            raise RuntimeError(
                f"End effector did not reach the target tolerance: {final_error:.4f} m"
            )

        print("\nIK REACH TEST PASSED")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

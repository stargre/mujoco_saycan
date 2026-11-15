"""Launch the interactive MuJoCo GUI viewer.

Requires a graphical display (local desktop or working X11 forwarding).
For headless servers use scripts/visualize_scene.py instead.
"""
from pathlib import Path
import sys
import time
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from mujoco_env import build_model


def main():
    cfg = Config()
    model = build_model(cfg.resolve())
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("Interactive MuJoCo viewer started. Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

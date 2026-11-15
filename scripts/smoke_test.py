from pathlib import Path
import sys

# Allow direct execution as: python scripts/smoke_test.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from mujoco_env import build_model, Runtime
from context import Context


def main():
    cfg = Config()
    model = build_model(cfg.resolve())
    runtime = Runtime(model, cfg.width, cfg.height)
    try:
        context = Context(runtime, "")
        context.robot.home()
        observation = context.observe()
        detected = list(observation.objects)
        print("Detected objects:", detected)
        assert "red_cube" in observation.objects, "RGB-D failed to detect red_cube"
        assert "blue_cube" in observation.objects, "RGB-D failed to detect blue_cube"
        print("SMOKE TEST PASSED")
    finally:
        runtime.r.close()


if __name__ == "__main__":
    main()

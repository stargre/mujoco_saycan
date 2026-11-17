"""Project configuration and Panda model discovery."""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


class Config:
    def __init__(self):
        configured = os.getenv("PANDA_MODEL_DIR", "").strip()
        self.model_dir = Path(configured).expanduser() if configured else None
        self.width = int(os.getenv("SAYCAN_RENDER_WIDTH", "640"))
        self.height = int(os.getenv("SAYCAN_RENDER_HEIGHT", "480"))
        self.output = PROJECT_ROOT / "outputs"

    def resolve(self):
        """Return a Menagerie Franka directory containing ``scene.xml``."""
        candidates = []
        if self.model_dir is not None:
            candidates.append(self.model_dir)
        candidates.extend([
            PROJECT_ROOT / "assets" / "franka_emika_panda",
            PROJECT_ROOT.parent / "mujoco_menagerie" / "franka_emika_panda",
        ])
        for candidate in candidates:
            if (candidate / "scene.xml").is_file():
                return candidate.resolve()
        checked = "\n  - ".join(str(path.resolve()) for path in candidates)
        raise FileNotFoundError(
            "Franka Panda MJCF was not found. Run "
            "`python scripts/setup_assets.py` or set PANDA_MODEL_DIR to the "
            "mujoco_menagerie/franka_emika_panda directory.\nChecked:\n  - "
            + checked
        )

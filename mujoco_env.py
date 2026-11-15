"""MuJoCo scene composition and runtime."""
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco


def build_model(folder):
    folder = Path(folder).resolve()
    source = folder / "scene.xml"
    if not source.is_file():
        raise FileNotFoundError(f"Missing Panda scene XML: {source}")
    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"No <worldbody> found in {source}")

    additions = [
        '<geom name="table" type="box" pos="0.48 0 0.32" size="0.42 0.48 0.04" rgba="0.55 0.48 0.38 1" friction="1.2 0.02 0.002"/>',
        '<camera name="overhead" pos="0.48 -0.72 1.05" xyaxes="1 0 0 0 0.82 0.57" fovy="55"/>',
        '<camera name="side_view" pos="1.2 -1.2 1.0" xyaxes="0.848 0.530 0 -0.222 0.355 0.914" fovy="50"/>',
        '<body name="red_cube" pos="0.43 -0.16 0.385"><freejoint/><geom name="red_cube_geom" type="box" size="0.025 0.025 0.025" mass="0.08" rgba="0.9 0.08 0.06 1" friction="1.5 0.02 0.002"/></body>',
        '<body name="blue_cube" pos="0.55 0.16 0.385"><freejoint/><geom name="blue_cube_geom" type="box" size="0.025 0.025 0.025" mass="0.08" rgba="0.05 0.2 0.95 1" friction="1.5 0.02 0.002"/></body>',
        '<geom name="target_marker" type="cylinder" pos="0.55 0.16 0.362" size="0.065 0.002" rgba="0.1 0.85 0.2 0.45" contype="0" conaffinity="0"/>',
    ]
    for item in additions:
        world.append(ET.fromstring(item))

    # from_xml_string's second argument is an asset byte mapping, not a path.
    # Save beside scene.xml so panda.xml and assets/ relative paths still resolve.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".xml",
            prefix=".saycan_scene_", dir=folder, delete=False
        ) as f:
            f.write(ET.tostring(root, encoding="unicode"))
            temp_path = Path(f.name)
        return mujoco.MjModel.from_xml_path(str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class Runtime:
    def __init__(self, model, w=640, h=480):
        self.m = model
        self.d = mujoco.MjData(model)
        self.r = mujoco.Renderer(model, height=h, width=w)
        mujoco.mj_resetData(model, self.d)
        mujoco.mj_forward(model, self.d)

    def step(self):
        mujoco.mj_step(self.m, self.d)

    def render(self, mode="rgb", camera="overhead"):
        self.r.update_scene(self.d, camera=camera)
        if mode == "depth":
            self.r.enable_depth_rendering()
        elif mode == "seg":
            self.r.enable_segmentation_rendering()
        image = self.r.render().copy()
        if mode == "depth":
            self.r.disable_depth_rendering()
        elif mode == "seg":
            self.r.disable_segmentation_rendering()
        return image

    def close(self):
        self.r.close()

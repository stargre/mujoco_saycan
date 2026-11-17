"""MuJoCo scene composition and runtime."""
import os
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

    # Keep the tabletop under both cubes but move its front edge away from the
    # Panda shoulder/link2 region. Bounds are configurable for clearance tests.
    x_min = float(os.getenv("SAYCAN_TABLE_X_MIN", "0.34"))
    x_max = float(os.getenv("SAYCAN_TABLE_X_MAX", "0.90"))
    y_half = float(os.getenv("SAYCAN_TABLE_Y_HALF", "0.48"))
    top = float(os.getenv("SAYCAN_TABLE_TOP", "0.36"))
    thickness = float(os.getenv("SAYCAN_TABLE_THICKNESS", "0.08"))
    if not (x_min < 0.405 and x_max > 0.575 and y_half >= 0.20):
        raise ValueError(
            "Table bounds must support both cubes: require x_min < 0.405, "
            "x_max > 0.575 and y_half >= 0.20"
        )
    if thickness <= 0 or y_half <= 0:
        raise ValueError("Table thickness and half-width must be positive")

    table_x = 0.5 * (x_min + x_max)
    table_half_x = 0.5 * (x_max - x_min)
    table_z = top - 0.5 * thickness
    table_half_z = 0.5 * thickness

    additions = [
        (
            f'<geom name="table" type="box" pos="{table_x} 0 {table_z}" '
            f'size="{table_half_x} {y_half} {table_half_z}" '
            'rgba="0.55 0.48 0.38 1" friction="1.2 0.02 0.002"/>'
        ),
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
        self._step_callbacks = []
        self._viewer = None
        self._closed = False
        mujoco.mj_resetData(model, self.d)
        mujoco.mj_forward(model, self.d)

    def step(self):
        mujoco.mj_step(self.m, self.d)
        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.sync()
            else:
                self._viewer = None
        for callback in tuple(self._step_callbacks):
            callback(self)

    def add_step_callback(self, callback):
        if callback not in self._step_callbacks:
            self._step_callbacks.append(callback)

    def remove_step_callback(self, callback):
        if callback in self._step_callbacks:
            self._step_callbacks.remove(callback)

    def enable_viewer(self):
        """Open a passive native viewer and synchronize it during motion."""
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.m, self.d)
        return self._viewer

    def render(self, mode="rgb", camera="overhead"):
        self.r.update_scene(self.d, camera=camera)
        try:
            if mode == "depth":
                self.r.enable_depth_rendering()
            elif mode == "seg":
                self.r.enable_segmentation_rendering()
            elif mode != "rgb":
                raise ValueError(f"Unknown render mode: {mode}")
            return self.r.render().copy()
        finally:
            if mode == "depth":
                self.r.disable_depth_rendering()
            elif mode == "seg":
                self.r.disable_segmentation_rendering()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.r.close()

import xml.etree.ElementTree as ET,mujoco
def build_model(folder):
 root=ET.parse(folder/"scene.xml").getroot();w=root.find("worldbody")
 for x in [
 '<geom name="table" type="box" pos="0.48 0 0.32" size="0.42 0.48 0.04"/>',
 '<camera name="overhead" pos="0.48 -0.72 1.05" xyaxes="1 0 0 0 0.82 0.57" fovy="55"/>',
 '<body name="red_cube" pos="0.43 -0.16 0.405"><freejoint/><geom name="red_cube_geom" type="box" size="0.025 0.025 0.025" mass="0.08" rgba="0.9 0.08 0.06 1"/></body>',
 '<body name="blue_cube" pos="0.55 0.16 0.405"><freejoint/><geom name="blue_cube_geom" type="box" size="0.025 0.025 0.025" mass="0.08" rgba="0.05 0.2 0.95 1"/></body>']:
  w.append(ET.fromstring(x))
 return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"),str(folder)+"/")
class Runtime:
 def __init__(self,m,w=640,h=480):
  self.m=m;self.d=mujoco.MjData(m);self.r=mujoco.Renderer(m,height=h,width=w);mujoco.mj_resetData(m,self.d);mujoco.mj_forward(m,self.d)
 def step(self):mujoco.mj_step(self.m,self.d)
 def render(self,mode="rgb"):
  self.r.update_scene(self.d,camera="overhead")
  if mode=="depth":self.r.enable_depth_rendering()
  if mode=="seg":self.r.enable_segmentation_rendering()
  x=self.r.render().copy()
  if mode=="depth":self.r.disable_depth_rendering()
  if mode=="seg":self.r.disable_segmentation_rendering()
  return x

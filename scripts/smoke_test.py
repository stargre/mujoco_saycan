from config import Config
from mujoco_env import build_model,Runtime
from context import Context
c=Config();r=Runtime(build_model(c.resolve()),c.width,c.height);x=Context(r,"");x.robot.home();o=x.observe()
print("Detected:",list(o.objects));assert "red_cube" in o.objects and "blue_cube" in o.objects
print("SMOKE TEST PASSED");r.r.close()

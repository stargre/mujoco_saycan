import numpy as np,mujoco
class Skills:
 def __init__(self,ctx):self.c=ctx
 def execute(self,c):
  n=c["name"];a=c["args"];x=a.get("target")
  if n in ("Explore","Find"):
   self.c.observe();ok=n=="Explore" or x in self.c.belief.objects
   return self.r(n,ok,"skill_complete" if ok else "target_not_found","RGB-D observation refreshed")
  if n=="PickUp":
   self.c.observe()
   if self.c.belief.held:return self.r(n,False,"gripper_not_empty","Gripper occupied")
   if x not in self.c.belief.objects:return self.r(n,False,"target_not_found","Target not observed")
   p=np.array(self.c.belief.objects[x]["position"]);self.c.robot.move(p+[0,0,.13]);self.c.robot.move(p+[0,0,.025]);self.c.robot.grip(False);self.c.robot.move(p+[0,0,.16])
   gid=mujoco.mj_name2id(self.c.rt.m,mujoco.mjtObj.mjOBJ_GEOM,x+"_geom")
   contact=any(gid in (self.c.rt.d.contact[j].geom1,self.c.rt.d.contact[j].geom2) for j in range(self.c.rt.d.ncon))
   if contact:self.c.belief.held=x;return self.r(n,True,"skill_complete","Contact verified")
   return self.r(n,False,"pickup_not_confirmed","No target contact detected")
  if n=="PlaceNear":
   if self.c.belief.held!=a["object"]:return self.r(n,False,"not_holding_object","Object not held")
   d=np.array(self.c.belief.objects[a["target"]]["position"]);p=d+np.array([.075,0,.055]);self.c.robot.move(p+[0,0,.13]);self.c.robot.move(p);self.c.robot.grip(True);self.c.robot.move(p+[0,0,.15]);self.c.observe();self.c.belief.held=None;self.c.belief.completed.append("place")
   return self.r(n,True,"skill_complete","Release sequence executed")
  if n=="Done":
   ok=not self.c.belief.held and (not self.c.task_place or "place" in self.c.belief.completed)
   return self.r(n,ok,"skill_complete" if ok else "task_not_complete","Task postcondition")
  if n=="NavigateTo":
   p=np.array(self.c.belief.objects[x]["position"]);self.c.robot.move(p+[0,0,.18]);return self.r(n,True,"skill_complete","Pre-grasp reached")
  if n=="Reach":
   p=np.array(self.c.belief.objects[x]["position"]);self.c.robot.move(p+[0,0,.04]);return self.r(n,True,"skill_complete","Approach reached")
  return self.r(n,False,"unknown_skill",n)
 def r(self,n,ok,s,msg):return {"skill":n,"success":ok,"status":s,"reason":msg}

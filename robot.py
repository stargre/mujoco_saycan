import mujoco,numpy as np
JOINTS=[f"joint{i}" for i in range(1,8)];ACT=[f"actuator{i}" for i in range(1,8)]
HOME=np.array([0,0,0,-1.57079,0,1.57079,-.7853])
class Panda:
 def __init__(self,rt):
  self.r=rt;m=rt.m;self.d=rt.d
  self.qa=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in JOINTS]
  self.da=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in JOINTS]
  self.ai=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ACT]
  self.g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,"actuator8");self.ee=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"hand")
 def q(self):return self.d.qpos[self.qa].copy()
 def pos(self):return self.d.xpos[self.ee].copy()
 def joints(self,target,dur=1.3):
  a=self.q();n=max(2,int(dur/self.r.m.opt.timestep))
  for i in range(n):
   t=(i+1)/n;s=10*t**3-15*t**4+6*t**5;self.d.ctrl[self.ai]=a+s*(target-a);self.r.step()
 def ik(self,x):
  m=self.r.m;x=np.array(x,float)
  for _ in range(200):
   mujoco.mj_forward(m,self.d);e=x-self.pos()
   if np.linalg.norm(e)<.007:return self.q()
   jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jacBody(m,self.d,jp,jr,self.ee);J=jp[:,self.da]
   dq=J.T@np.linalg.solve(J@J.T+.003*np.eye(3),e);q=self.q()+np.clip(dq,-.05,.05)
   for j,n in enumerate(JOINTS):
    k=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n);lo,hi=m.jnt_range[k];q[j]=np.clip(q[j],lo+.01,hi-.01)
   self.d.qpos[self.qa]=q
  raise RuntimeError("IK did not converge")
 def move(self,x,dur=1.2):self.joints(self.ik(x),dur)
 def grip(self,opened,dur=.6):
  self.d.ctrl[self.g]=255 if opened else 0
  for _ in range(int(dur/self.r.m.opt.timestep)):self.r.step()
 def home(self):self.grip(True);self.joints(HOME,2)

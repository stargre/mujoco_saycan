import mujoco,numpy as np
class Vision:
 def __init__(self,rt):
  self.r=rt;self.m=rt.m;self.d=rt.d;self.ids={n:mujoco.mj_name2id(rt.m,mujoco.mjtObj.mjOBJ_GEOM,n+"_geom") for n in ("red_cube","blue_cube")}
 def observe(self):
  rgb=self.r.render();dep=self.r.render("depth");seg=self.r.render("seg");cid=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_CAMERA,"overhead")
  cp=self.d.cam_xpos[cid].copy();R=self.d.cam_xmat[cid].reshape(3,3).copy();f=np.deg2rad(self.m.cam_fovy[cid]);fy=rgb.shape[0]/(2*np.tan(f/2));fx=fy;out={}
  for name,gid in self.ids.items():
   mask=(seg[:,:,0]==gid)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM));ys,xs=np.where(mask)
   if len(xs)<5:continue
   z=float(np.median(dep[ys,xs]));u=float(np.median(xs));v=float(np.median(ys));p=cp+R@np.array([(u-rgb.shape[1]/2)*z/fx,-(v-rgb.shape[0]/2)*z/fy,z])
   out[name]={"position":p.tolist(),"confidence":min(1,len(xs)/100)}
  return out

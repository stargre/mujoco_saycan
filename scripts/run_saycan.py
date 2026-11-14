import argparse,json
from config import Config
from mujoco_env import build_model,Runtime
from context import Context
from saycan import SayCan,Scorer
def main():
 p=argparse.ArgumentParser();p.add_argument("--scorer",choices=["qwen","keyword"],default="qwen");p.add_argument("--task",default="Pick up the red cube and place it near the blue cube");a=p.parse_args()
 cfg=Config();rt=Runtime(build_model(cfg.resolve()),cfg.width,cfg.height);ctx=Context(rt,a.task);ctx.robot.home();ctx.observe()
 result=SayCan(ctx,Scorer(a.scorer)).run(a.task);cfg.output.mkdir(exist_ok=True);(cfg.output/"last_run.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));rt.r.close()
 if not result["success"]:raise SystemExit(1)
if __name__=="__main__":main()

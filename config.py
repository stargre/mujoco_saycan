import os
from pathlib import Path
class Config:
 model_dir=Path(os.getenv("PANDA_MODEL_DIR",""));width=640;height=480;output=Path("outputs")
 def resolve(self):
  for p in (self.model_dir,Path("assets/franka_emika_panda")):
   if p and (p/"scene.xml").exists():return p.resolve()
  raise FileNotFoundError("Set PANDA_MODEL_DIR to the Menagerie franka_emika_panda directory.")
